from __future__ import annotations
import math, platform, torch, json, csv, pathlib, time, warnings, re
import torch.nn.functional as F
from torch.cuda.amp import autocast
from dataclasses import dataclass

from dataset   import build_loaders, LoaderCfg, FEATURE_VARS
from informer  import Informer


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.deterministic    = False

#---- Function ----
#Calculate quantiles
def compute_quantile(loader, q=0.95, subsample_ratio=0.02, max_batches=50):
    import numpy as np
    vals = []   # Store sampled values
    with torch.no_grad():
        for bi, (_, y) in enumerate(loader):
            y_flat = y.to(torch.float32).view(-1)
            k = (max(1, int(y_flat.numel() * subsample_ratio))
                 if 0 < subsample_ratio < 1 else y_flat.numel())
            idx = torch.randint(0, y_flat.numel(), (k,), device=y_flat.device)
            vals.append(y_flat[idx].cpu().numpy())
            if max_batches and (bi + 1) >= max_batches:
                break
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, q)) # Calculate and return the specified percentile value

# Single-pass forward parallel processing with simultaneous statistics for P95/P99/17ms
def eval_extreme_triple(model, loader,
                        thr95: float, thr99: float, thr17: float = 17.0):
    stats = {k: dict(n=0, se=0.0, ae=0.0, TP=0, FP=0, FN=0, TN=0)
             for k in ("95", "99", "17")}
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            with autocast(enabled=False):
                yhat = model(X)                                                 # Model Prediction
            yhat32, y32 = yhat.float(), y.float()                               # Convert predicted values and actual values to float type
            err = yhat32 - y32                                                  # Calculate errors

            for tag, th in (("95", thr95), ("99", thr99), ("17", thr17)):
                s  = stats[tag]                                                 # Retrieve statistics for the current threshold
                tail = (y32 >= th)                                              # Determine whether the actual value exceed the threshold
                pred_tail = (yhat32 >= th)                                      # Determine whether the predicted value exceed the threshold
                s["n"]  += tail.sum().item()                                    # Cumulative number of actual values exceeding the threshold
                s["se"] += (err[tail]**2).sum().item()                          # Cumulative squared error exceeding the threshold
                s["ae"] += err[tail].abs().sum().item()                         # Cumulative absolute error exceeding the threshold
                s["TP"] += ( pred_tail &  tail).sum().item()
                s["FP"] += ( pred_tail & ~tail).sum().item()
                s["FN"] += (~pred_tail &  tail).sum().item()
                s["TN"] += (~pred_tail & ~tail).sum().item()

    def fin(s):
        n = max(s["n"], 1)
        rmse = math.sqrt(s["se"]/n); mae = s["ae"]/n                           # Calculate N,RMSE,MAE
        TP, FP, FN = s["TP"], s["FP"], s["FN"]
        recall    = TP / max(TP+FN, 1)                                          # Calculate Recall, Precision, CSI, FAR
        precision = TP / max(TP+FP, 1)
        csi       = TP / max(TP+FP+FN, 1)
        far       = FP / max(TP+FP, 1)
        return dict(n_tail=int(n), rmse_tail=rmse, mae_tail=mae,
                    TP=int(TP), FP=int(FP), FN=int(FN), TN=int(s["TN"]),
                    recall=recall, precision=precision, csi=csi, far=far)

    return fin(stats["95"]), fin(stats["99"]), fin(stats["17"])


def main():
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"CUDA available: {torch.cuda.is_available()}\n")

    out_dir = pathlib.Path("results_informer")
    out_dir.mkdir(exist_ok=True)

    # 1) Estimated P95/P99 Thresholds (CPU Multi-Process)
    thr_t0 = time.perf_counter()
    cpu_cfg = LoaderCfg(batch=16, device="cpu", use_fp16=False, num_workers=8)
    _, val_cpu, _ = build_loaders(cpu_cfg)

    q95 = compute_quantile(val_cpu, q=0.95, subsample_ratio=0.01, max_batches=30) # Calculate P95 threshold
    q99 = compute_quantile(val_cpu, q=0.99, subsample_ratio=0.03, max_batches=60) # Calculate P99 threshold
    thr_time = time.perf_counter() - thr_t0
    print(f"P95 = {q95:.3f}, P99 = {q99:.3f}  (Time {thr_time:.1f}s)\n")

    # 2) Build a GPU loader + Informer model
    gpu_cfg = LoaderCfg(batch=8,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        use_fp16=False, num_workers=4)
    _, val_dl, test_dl = build_loaders(gpu_cfg)
    device = torch.device(gpu_cfg.device)

    #same as former model
    model = Informer(                                                        # Building the ConvLSTM Model(Same Parameters)
        C=len(FEATURE_VARS), H=145, W=253,
        T_out=gpu_cfg.win_out,
        d_model=256, n_heads=8,
        e_layers=2, d_layers=3
    ).to(device)

    ckpt = "best_informer.pt"                                                # Load model weight
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()
    print(f"Load {ckpt}\n")

    # Extreme Evaluation: P95/P99/17m/s
    def _fmt(tag, d):
        return (f"{tag}: n={d['n_tail']:>7}, RMSE={d['rmse_tail']:.3f}, "
                f"MAE={d['mae_tail']:.3f}, Recall={d['recall']:.3f}, "
                f"Precision={d['precision']:.3f}, CSI={d['csi']:.3f}, "
                f"FAR={d['far']:.3f}")

    t1 = time.perf_counter()
    v95, v99, v17 = eval_extreme_triple(model, val_dl, q95, q99)           # Evaluate P95/P99/17ms on the validation set
    t2 = time.perf_counter()                                               # Evaluate P95/P99/17ms on the testing set
    t95, t99, t17 = eval_extreme_triple(model, test_dl, q95, q99)
    t3 = time.perf_counter()
    # Print results
    print("==== Extreme segment (Informer) ====")
    print("Val ", _fmt("@P95", v95))
    print("Val ", _fmt("@P99", v99))
    print("Val ", _fmt("@17ms", v17))
    print("Test", _fmt("@P95", t95))
    print("Test", _fmt("@P99", t99))
    print("Test", _fmt("@17ms", t17))
    print(f"\nTime：val forward {t2 - t1:.1f}s, test forward {t3 - t2:.1f}s, Whole Time: {t3 - thr_t0:.1f}s")

    # Saving results
    results = {
        "thresholds": {"P95": q95, "P99": q99, "time_sec": thr_time},
        "extreme": {
            "val":  {"P95": v95,  "P99": v99,  "17ms": v17,  "time_sec": t2 - t1},
            "test": {"P95": t95,  "P99": t99,  "17ms": t17,  "time_sec": t3 - t2},
        },
        "total_time_sec": t3 - thr_t0
    }
    with open(out_dir / "extreme_metrics_informer.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "extreme_metrics_informer_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "P", "n_tail", "RMSE", "MAE", "Recall",
                    "Precision", "CSI", "FAR"])
        for split, d in (("val", v95), ("val", v99), ("val", v17),
                         ("test", t95), ("test", t99), ("test", t17)):
            P = {"P95": "P95", "P99": "P99", "17ms": "17ms"}[
                    "P95" if d is v95 or d is t95 else
                    "P99" if d is v99 or d is t99 else
                    "17ms"]
            w.writerow([split, P, d["n_tail"], d["rmse_tail"], d["mae_tail"],
                        d["recall"], d["precision"], d["csi"], d["far"]])

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
