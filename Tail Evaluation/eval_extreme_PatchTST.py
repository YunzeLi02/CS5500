from __future__ import annotations
import math, time, json, csv, pathlib, warnings
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from dataset import build_loaders, LoaderCfg, FEATURE_VARS
from layers.PatchTST_backbone import PatchTST_backbone as Model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True
torch.backends.cudnn.deterministic    = False


#---- Function ----
# Transfer y into [B, T, H, W]
def _y_to_grid_raw(y: torch.Tensor) -> torch.Tensor:
    while y.dim() > 5 and y.size(-1) == 1:
        y = y.squeeze(-1)
    if y.dim() == 5 and y.size(2) == 1:
        y = y.squeeze(2)  # [B,T,H,W]
    if y.dim() != 4:
        raise RuntimeError(f"y expect [B,T,H,W], actually received {tuple(y.shape)}")
    return y


# Spatial template P remains “normalized to 1” (dimensionless) after m/s construction; ensures ŷ_grid units match raw y. Ensure that the units of ŷ_grid match those of raw y
def _template_from_inputs(
    x: torch.Tensor,
    idx_ws10: int | None,
    mean_ws10: float, std_ws10: float,
    idx_u10: int | None = None, mean_u10: float | None = None, std_u10: float | None = None,
    idx_v10: int | None = None, mean_v10: float | None = None, std_v10: float | None = None,
    last_k: int = 1, eps: float = 1e-3
) -> torch.Tensor:
    z = x
    while z.dim() > 5 and z.shape[-1] == 1:
        z = z.squeeze(-1)
    if z.dim() != 5:
        raise RuntimeError(f"Unexpected grid tensor shape {tuple(z.shape)}")

    # [B,L,C,H,W] or [B,L,H,W,C]
    if z.shape[2] == len(FEATURE_VARS):      # [B,L,C,H,W]
        def get_feat(ci): return z[:, -last_k:, ci, :, :]      # [B,k,H,W]
    elif z.shape[-1] == len(FEATURE_VARS):   # [B,L,H,W,C]
        def get_feat(ci): return z[:, -last_k:, :, :, ci]      # [B,k,H,W]
    else:
        raise RuntimeError("Please verify that FEATURE_VARS matches the shape of x.")

    if idx_ws10 is not None and idx_ws10 >= 0:
        ws = get_feat(idx_ws10) * std_ws10 + mean_ws10        # Transfer 'z' to 'm/s'
        P  = ws.mean(dim=1)                                   # [B,H,W] (m/s)
    else:
        if (idx_u10 is None) or (idx_v10 is None):
            raise RuntimeError("No ws10")
        u = get_feat(idx_u10) * std_u10 + mean_u10
        v = get_feat(idx_v10) * std_v10 + mean_v10
        P = (u.pow(2) + v.pow(2)).sqrt().mean(dim=1)

    P = torch.clamp(P, min=eps)
    P = P / (P.mean(dim=(-1, -2), keepdim=True) + 1e-6)
    return P  # [B,H,W]，




@torch.no_grad()
# Calculate quantiles
def compute_quantile_raw(loader, q=0.95, subsample_ratio=0.02, max_batches=50):
    import numpy as np, torch
    vals = []   # Store sampled values
    for bi, (_, y) in enumerate(loader):
        y_raw = _y_to_grid_raw(y.to(torch.float32))  # [B,T,H,W] (raw)
        y_flat = y_raw.reshape(-1)
        k = (max(1, int(y_flat.numel() * subsample_ratio))
             if 0 < subsample_ratio < 1 else y_flat.numel())
        idx = torch.randint(0, y_flat.numel(), (k,), device=y_flat.device)
        vals.append(y_flat[idx].cpu().numpy())
        if max_batches and (bi + 1) >= max_batches:
            break
    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, q)) # Calculate and return the specified percentile value

# Extreme Evaluation: P95/P99/17m/s (Grid)
@torch.no_grad()
def eval_extreme_triple_grid_raw(
    model, loader, device,
    thr95_raw: float, thr99_raw: float, thr17_mps: float,
    idx_ws10: int | None, mean_ws10: float, std_ws10: float,
    idx_u10: int | None = None, mean_u10: float | None = None, std_u10: float | None = None,
    idx_v10: int | None = None, mean_v10: float | None = None, std_v10: float | None = None,
    last_k: int = 1,
    *,
    model_out_is_z: bool = True,
    label_is_z: bool = False
):
    stats = {k: dict(n=0, se=0.0, ae=0.0, TP=0, FP=0, FN=0, TN=0)
             for k in ("95", "99", "17")}
    model.eval()

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        #Label to(m/s)Format
        y_grid = _y_to_grid_raw(y.to(torch.float32))     # [B,T,H,W]
        if label_is_z:
            y_grid = y_grid * std_ws10 + mean_ws10       # z -> m/s

        #Model output to m/s
        with autocast(enabled=False):
            yhat_series = model(X).float()               # [B,T]
        if model_out_is_z:
            yhat_series = yhat_series * std_ws10 + mean_ws10

        P = _template_from_inputs(
            X, idx_ws10, mean_ws10, std_ws10,
            idx_u10, mean_u10, std_u10,
            idx_v10, mean_v10, std_v10,
            last_k=last_k
        )                                                # [B,H,W], mean=1

        #Approximate Reconstruction Grid Prediction
        yhat_grid = yhat_series[:, :, None, None] * P[:, None, :, :]  # [B,T,H,W]

        # 3 Thresholds
        err = yhat_grid - y_grid                                                        # Calculate errors
        for tag, th in (("95", thr95_raw), ("99", thr99_raw), ("17", thr17_mps)):
            s = stats[tag]                                                              # Retrieve statistics for the current threshold
            tail = (y_grid >= th)                                                       # Determine whether the actual value exceed the threshold
            pred = (yhat_grid >= th)                                                    # Determine whether the predicted value exceed the threshold
            s["n"]  += tail.sum().item()                                                # Cumulative number of actual values exceeding the threshold
            s["se"] += (err[tail]**2).sum().item()                                      # Cumulative squared error exceeding the threshold
            s["ae"] += err[tail].abs().sum().item()                                     # Cumulative absolute error exceeding the threshold
            TP = ( pred &  tail).sum().item()
            FP = ( pred & ~tail).sum().item()
            FN = (~pred &  tail).sum().item()
            TN = (~pred & ~tail).sum().item()
            s["TP"] += TP; s["FP"] += FP; s["FN"] += FN; s["TN"] += TN

    def fin(s):
        n = max(s["n"], 1)                                                            # Calculate N,RMSE,MAE
        rmse = math.sqrt(s["se"]/n); mae = s["ae"]/n
        TP, FP, FN = s["TP"], s["FP"], s["FN"]
        recall    = TP / max(TP+FN, 1)                                                # Calculate Recall
        precision = TP / max(TP+FP, 1)                                                # Calculate Precision
        csi       = TP / max(TP+FP+FN, 1)                                             # Calculate CSI
        far       = FP / max(TP+FP, 1)                                                # Calculate FAR
        return dict(n_tail=int(n), rmse_tail=rmse, mae_tail=mae,
                    TP=int(TP), FP=int(FP), FN=int(FN), TN=int(s["TN"] ),
                    recall=recall, precision=precision, csi=csi, far=far)

    return fin(stats["95"]), fin(stats["99"]), fin(stats["17"])

def main():
    print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}\n")
    out_dir = pathlib.Path("results_patchtst")
    out_dir.mkdir(exist_ok=True)

    #Load scaler (z <-> m/s)

    sc = np.load("scaler_train.npz")
    mean_all = sc["mean"]; std_all = sc["std"]

    name2idx  = {n:i for i,n in enumerate(FEATURE_VARS)}
    idx_ws10  = name2idx.get("ws10", -1)
    idx_u10   = name2idx.get("u10",  None)
    idx_v10   = name2idx.get("v10",  None)

    mean_ws10 = float(mean_all[name2idx["ws10"]]); std_ws10 = float(std_all[name2idx["ws10"]])
    mean_u10  = float(mean_all[idx_u10])  if idx_u10 is not None else None
    std_u10   = float(std_all[idx_u10])   if idx_u10  is not None else None
    mean_v10  = float(mean_all[idx_v10])  if idx_v10 is not None else None
    std_v10   = float(std_all[idx_v10])   if idx_v10  is not None else None

    # Estimated P95/P99 Thresholds (CPU Multi-Process)
    thr_t0 = time.perf_counter()
    cpu_cfg = LoaderCfg(batch=16, device="cpu", use_fp16=False, num_workers=8)
    _, val_cpu, _ = build_loaders(cpu_cfg)

    q95_raw = compute_quantile_raw(val_cpu, q=0.95, subsample_ratio=0.01, max_batches=30)  # Calculate P95 threshold
    q99_raw = compute_quantile_raw(val_cpu, q=0.99, subsample_ratio=0.03, max_batches=60)  # Calculate P99 threshold
    thr_time = time.perf_counter() - thr_t0

    print(f"P95 = {q95_raw:.3f}, P99 = {q99_raw:.3f}  (Threshold estimation {thr_time:.1f}s)\n")


    # Build a GPU loader + PatchTST model
    gpu_cfg = LoaderCfg(batch=8,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        use_fp16=False, num_workers=4)
    _, val_dl, test_dl = build_loaders(gpu_cfg)
    device = torch.device(gpu_cfg.device)

    model = Model(                                                            # Building the Model(Same Parameters)
        c_in=12, context_window=12, target_window=3,
        patch_len=4, stride=1, padding_patch=None,
        n_layers=3, d_model=256, n_heads=8, d_ff=512,
        dropout=0.1, attn_dropout=0.1, pre_norm=True,
        head_type='fusion', head_dropout=0.1,
        target_idx=8,  # ws10
        revin=False, affine=True, subtract_last=False,
        pe='sincos'
    ).to(device)

    ckpt = "best_patchtst_series"
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval()
    print(f"Load {ckpt}\n")

    # Extreme Evaluation: P95/P99/17m/s
    def _fmt(tag, d):
        return (f"{tag}: n={d['n_tail']:>7}, RMSE={d['rmse_tail']:.3f}, "
                f"MAE={d['mae_tail']:.3f}, Recall={d['recall']:.3f}, "
                f"Precision={d['precision']:.3f}, CSI={d['csi']:.3f}, FAR={d['far']:.3f}")
    #Time Record
    t1 = time.perf_counter()
    v95, v99, v17 = eval_extreme_triple_grid_raw(                                  # Evaluate P95/P99/17ms on the validation set
        model, val_dl, device,
        thr95_raw=q95_raw, thr99_raw=q99_raw, thr17_mps=17.0,
        idx_ws10=(idx_ws10 if idx_ws10 >= 0 else None),
        mean_ws10=mean_ws10, std_ws10=std_ws10,
        idx_u10=idx_u10, mean_u10=mean_u10, std_u10=std_u10,
        idx_v10=idx_v10, mean_v10=mean_v10, std_v10=std_v10,
        last_k=1,
        model_out_is_z=True,
        label_is_z=False
    )

    t2 = time.perf_counter()
    t95, t99, t17 = eval_extreme_triple_grid_raw(                                   # Evaluate P95/P99/17ms on the testing set
        model, test_dl, device,
        thr95_raw=q95_raw, thr99_raw=q99_raw, thr17_mps=17.0,
        idx_ws10=(idx_ws10 if idx_ws10 >= 0 else None),
        mean_ws10=mean_ws10, std_ws10=std_ws10,
        idx_u10=idx_u10, mean_u10=mean_u10, std_u10=std_u10,
        idx_v10=idx_v10, mean_v10=mean_v10, std_v10=std_v10,
        last_k=1,
        model_out_is_z = True,
        label_is_z = False
    )
    t3 = time.perf_counter()
    # Print results
    print("==== Extreme segment (PatchTST, spatial-lift, raw-space) ====")
    print("Val ", _fmt("@P95",  v95))
    print("Val ", _fmt("@P99",  v99))
    print("Val ", _fmt("@17ms", v17))
    print("Test", _fmt("@P95",  t95))
    print("Test", _fmt("@P99",  t99))
    print("Test", _fmt("@17ms", t17))
    print(f"\nTime：val forward {t2 - t1:.1f}s, test forward {t3 - t2:.1f}s, Whole Time: {t3 - thr_t0:.1f}s")

    # Saving results
    out_dir.mkdir(exist_ok=True)
    results = {
        "thresholds": {"P95_raw": q95_raw, "P99_raw": q99_raw, "time_sec": thr_time},
        "extreme": {
            "val":  {"P95": v95,  "P99": v99,  "17ms": v17,  "time_sec": t2 - t1},
            "test": {"P95": t95,  "P99": t99,  "17ms": t17,  "time_sec": t3 - t2},
        },
        "total_time_sec": t3 - thr_t0
    }
    with open(out_dir / "extreme_metrics_patchtst.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "extreme_metrics_patchtst_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "P", "n_tail", "RMSE", "MAE", "Recall", "Precision", "CSI", "FAR"])
        for split, d, tag in (("val", v95, "P95"), ("val", v99, "P99"), ("val", v17, "17ms"),
                              ("test", t95, "P95"), ("test", t99, "P99"), ("test", t17, "17ms")):
            w.writerow([split, tag, d["n_tail"], d["rmse_tail"], d["mae_tail"],
                        d["recall"], d["precision"], d["csi"], d["far"]])


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
