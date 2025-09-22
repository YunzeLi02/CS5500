"""
evaluate_convlstm_extreme.py
— 同时评估 P95 / P99 / 固定阈值 17 m s⁻¹
"""

from __future__ import annotations
import math, platform, torch, json, csv, pathlib, time
import torch.nn.functional as F
from torch.cuda.amp import autocast
from dataclasses import dataclass

from dataset  import build_loaders, LoaderCfg, FEATURE_VARS
from CovLSTM  import ConvLSTM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

# ---- Function ----
# Calculate quantiles
def compute_quantile(loader, q=0.95, subsample_ratio=0.02, max_batches=50):
    import numpy as np
    vals = [] # Store sampled values
    with torch.no_grad():
        for bi, (_, y) in enumerate(loader):
            y_flat = y.float().view(-1)
            k = max(1, int(y_flat.numel()*subsample_ratio)) if 0<subsample_ratio<1 else y_flat.numel()
            idx = torch.randint(0, y_flat.numel(), (k,), device=y_flat.device)
            vals.append(y_flat[idx].cpu().numpy())
            if max_batches and bi+1>=max_batches: break
    vals = np.concatenate(vals); vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, q)) # Calculate and return the specified percentile value

# Single-pass forward parallel processing with simultaneous statistics for P95/P99/17ms
def eval_extreme_triple(model, loader, thr95: float, thr99: float, thr17: float = 17.0):
    stats = {tag: dict(n=0, se=0., ae=0., TP=0, FP=0, FN=0, TN=0)
             for tag in ("95", "99", "17")}
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            with autocast(enabled=False):
                yhat = model(X)                                                 # Model Prediction
            yhat32, y32 = yhat.float(), y.float()                               # Convert predicted values and actual values to float type
            err = yhat32 - y32                                                  # Calculate errors
            for tag, th in (("95", thr95), ("99", thr99), ("17", thr17)):
                s = stats[tag]                                                  # Retrieve statistics for the current threshold
                tail = (y32 >= th); pred_tail = (yhat32 >= th)                  # Determine whether the actual value and predicted value exceed the threshold
                s["n"]  += tail.sum().item()                                    # Cumulative number of actual values exceeding the threshold
                s["se"] += (err[tail]**2).sum().item()                          # Cumulative squared error exceeding the threshold
                s["ae"] += err[tail].abs().sum().item()                         # Cumulative absolute error exceeding the threshold
                s["TP"] += ( pred_tail &  tail).sum().item()
                s["FP"] += ( pred_tail & ~tail).sum().item()
                s["FN"] += (~pred_tail &  tail).sum().item()
                s["TN"] += (~pred_tail & ~tail).sum().item()

    def fin(s):
        n=max(s["n"],1); rmse=math.sqrt(s["se"]/n); mae=s["ae"]/n              # Calculate N,RMSE,MAE
        TP,FP,FN=s["TP"],s["FP"],s["FN"]
        recall=TP/max(TP+FN,1); precision=TP/max(TP+FP,1)                      # Claculate Recall, Precision
        csi=TP/max(TP+FP+FN,1); far=FP/max(TP+FP,1)                            # Calculate CSI and FAR
        return dict(n_tail=int(n), rmse_tail=rmse, mae_tail=mae,
                    TP=int(TP), FP=int(FP), FN=int(FN), TN=int(s["TN"]),
                    recall=recall, precision=precision, csi=csi, far=far)

    return fin(stats["95"]), fin(stats["99"]), fin(stats["17"])

def main():
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | CUDA OK {torch.cuda.is_available()}\n")

    out_dir = pathlib.Path("results_convlstm")
    out_dir.mkdir(exist_ok=True)

    # Estimated P95/P99 Thresholds (CPU Multi-Process)
    cpu_cfg = LoaderCfg(batch=16, device="cpu", use_fp16=False, num_workers=8)
    _, val_cpu, _ = build_loaders(cpu_cfg)
    q95 = compute_quantile(val_cpu, 0.95, 0.01, 30) # Calculate P95 threshold
    q99 = compute_quantile(val_cpu, 0.99, 0.03, 60) # Calculate P99 threshold
    print(f"P95={q95:.3f}, P99={q99:.3f}\n")

    # Build a GPU loader + ConvLSTM model
    gpu_cfg = LoaderCfg(batch=8, device="cuda" if torch.cuda.is_available() else "cpu",
                        use_fp16=False, num_workers=0)
    _, val_dl, test_dl = build_loaders(gpu_cfg)
    device = torch.device(gpu_cfg.device)

    model = ConvLSTM(len(FEATURE_VARS), 64, 1, gpu_cfg.win_out).to(device)  # Building the ConvLSTM Model(Same Parameters)
    ckpt = "best_convlstm.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))                                     # Load model weight
    model.eval()
    print(f"Load {ckpt}\n")

    # Extreme Evaluation: P95/P99/17m/s
    def fmt(tag,d):
        return (f"{tag}: n={d['n_tail']:>7}, RMSE={d['rmse_tail']:.3f}, MAE={d['mae_tail']:.3f}, "
                f"Rec={d['recall']:.3f}, Pre={d['precision']:.3f}, CSI={d['csi']:.3f}, FAR={d['far']:.3f}")

    v95,v99,v17 = eval_extreme_triple(model,val_dl,q95,q99)       # Evaluate P95/P99/17ms on the validation set
    t95,t99,t17 = eval_extreme_triple(model,test_dl,q95,q99)      # Evaluate P95/P99/17ms on the testing set
    # Print results
    print("==== Extreme segment (ConvLSTM) ====")
    print(fmt("Val  @P95", v95)); print(fmt("Val  @P99", v99)); print(fmt("Val  @17ms", v17))
    print(fmt("Test @P95", t95)); print(fmt("Test @P99", t99)); print(fmt("Test @17ms", t17))

    # Saving results
    with open(out_dir / "extreme_metrics_summary.csv", "w", newline="") as f:
        w=csv.writer(f); w.writerow(["split","thr","n","RMSE","MAE","Rec","Pre","CSI","FAR"])
        for split,d in (("val",v95),("val",v99),("val",v17),
                        ("test",t95),("test",t99),("test",t17)):
            thr = "P95" if d is v95 or d is t95 else "P99" if d is v99 or d is t99 else "17ms"
            w.writerow([split,thr,d["n_tail"],d["rmse_tail"],d["mae_tail"],
                        d["recall"],d["precision"],d["csi"],d["far"]])

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
