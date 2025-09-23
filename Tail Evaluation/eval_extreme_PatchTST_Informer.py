from __future__ import annotations
import math, time, json, csv, pathlib
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from dataset import build_loaders, LoaderCfg, FEATURE_VARS
from informer import Informer
from layers.PatchTST_backbone import PatchTST_backbone as PatchTST_backbone
import torch.nn as nn

#---- Function ----
def to_BTHW(t: torch.Tensor) -> torch.Tensor:
    # [B,T,1,H,W] -> [B,T,H,W]；[B,T,H,W]
    if t.dim() == 5 and t.size(2) == 1: return t.squeeze(2)
    if t.dim() == 4: return t
    raise RuntimeError(f"Unexpected grid tensor shape {tuple(t.shape)}")

@torch.no_grad()
# Calculate quantiles
def compute_quantile_raw(loader, q=0.95, subsample_ratio=0.02, max_batches=50) -> float:
    import numpy as np
    vals = []   # Store sampled values
    for bi, (_, y) in enumerate(loader):
        y = to_BTHW(y.to(torch.float32))
        flat = y.reshape(-1)
        k = (max(1, int(flat.numel()*subsample_ratio))
             if 0<subsample_ratio<1 else flat.numel())
        idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
        vals.append(flat[idx].cpu().numpy())
        if max_batches and (bi+1) >= max_batches: break
    vals = np.concatenate(vals); vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, q)) # Calculate and return the specified percentile value

#----Light Conditioner ----
class PatchTST_Featurizer(PatchTST_backbone):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, L, C]
        if self.revin: z = self.revin_layer(z, 'norm')
        z = z.permute(0,2,1).contiguous()           # [B,C,L]: adjust the dimension order of z
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(-1, self.patch_len, self.stride)   # [B,C,P,patch_len]: perform a sliding window operation on z
        z = z.permute(0,1,3,2).contiguous()         # [B,C,patch_len,P]: adjust the dimension order of z
        return self.backbone(z)                     # [B,C,d_model,P]: processing z through the backbone layer

class Series2GridConditioner(nn.Module):
    def __init__(self, c_in:int, d_model:int, r:int, H:int, W:int,
                 context_window:int, patch_len:int=1, stride:int=1,
                 n_layers:int=2, n_heads:int=8, d_ff:int=256,
                 revin:bool=True, pe:str='sincos'):
        super().__init__()
        self.H, self.W, self.r = H, W, r           # Initialize height, width, and number of channels
        self.feat = PatchTST_Featurizer(
            c_in=c_in, context_window=context_window, target_window=1,
            patch_len=patch_len, stride=stride, padding_patch=None,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=0.1, attn_dropout=0.1, pre_norm=True,
            head_type='flatten', head_dropout=0.0,
            revin=revin, affine=True, subtract_last=False, pe=pe
        )
        self.merge = nn.Sequential(               # Initialize Feature Extractor
            nn.Conv1d(in_channels=c_in*d_model, out_channels=128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(in_channels=128, out_channels=r, kernel_size=1),
        )                                         # Initialize the merging layer
        self.spatial_basis = nn.Parameter(torch.randn(r, H, W) * 0.01)                  # Initialize the spatial basis

    def forward(self, X_grid: torch.Tensor) -> torch.Tensor:
        # X_grid: [B,L,C,H,W]
        B,L,C,H,W = X_grid.shape
        x_series  = X_grid.mean(dim=(-1,-2))                             # [B,L,C]
        tok       = self.feat(x_series)                                  # [B,C,D,P=L]
        vec       = tok.permute(0,3,1,2).reshape(B, L, -1)               # [B,L,C*D]
        coeff     = self.merge(vec.transpose(1,2)).transpose(1,2)        # [B,L,r]
        grid_r    = torch.einsum('blr,rhw->blrhw', coeff, self.spatial_basis) # [B,L,r,H,W]
        return grid_r

# Extreme Evaluation: P95/P99/17m/s (Grid)
@torch.no_grad()
def eval_extreme_triple_grid_hybrid(model, cond, loader, device,
                                    thr95_raw: float, thr99_raw: float, thr17_mps: float):
    stats = {k: dict(n=0, se=0.0, ae=0.0, TP=0, FP=0, FN=0, TN=0)
             for k in ("95","99","17")}
    model.eval(); cond.eval()

    for X, y in loader:
        X = X.to(device); y = to_BTHW(y.to(device).to(torch.float32))  # [B,T,H,W]: transfer data to the device and convert the format
        # Generate the r-channel condition map and stitch it together.
        X_aug  = torch.cat([X, cond(X)], dim=2)                 # [B,L,C+r,H,W]: concatenate the raw data and condition map
        with autocast(enabled=False):
            pred = model(X_aug)                                         # [B,T,1,H,W] or [B,T,H,W]: model Prediction
        yhat = to_BTHW(pred)                                            # [B,T,H,W]

        err = yhat - y                                                  # Calculate errors
        for tag, th in (("95", thr95_raw), ("99", thr99_raw), ("17", thr17_mps)):
            s = stats[tag]                                              # Retrieve statistics for the current threshold
            tail = (y >= th)                                            # Determine whether the actual value exceed the threshold
            pred_tail = (yhat >= th)                                    # Determine whether the predicted value exceed the threshold
            s["n"]  += tail.sum().item()                                # Cumulative number of actual values exceeding the threshold
            s["se"] += (err[tail]**2).sum().item()                      # Cumulative squared error exceeding the threshold
            s["ae"] += err[tail].abs().sum().item()                     # Cumulative absolute error exceeding the threshold
            TP = ( pred_tail &  tail).sum().item()
            FP = ( pred_tail & ~tail).sum().item()
            FN = (~pred_tail &  tail).sum().item()
            TN = (~pred_tail & ~tail).sum().item()
            s["TP"] += TP; s["FP"] += FP; s["FN"] += FN; s["TN"] += TN

    def fin(s):
        n = max(s["n"], 1)
        rmse = math.sqrt(s["se"]/n); mae = s["ae"]/n                    # Calculate N,RMSE,MAE
        TP,FP,FN = s["TP"], s["FP"], s["FN"]
        recall    = TP / max(TP+FN, 1)                                  # Calculate Recall
        precision = TP / max(TP+FP, 1)                                  # Calculate Precision
        csi       = TP / max(TP+FP+FN, 1)                               # Calculate CSI
        far       = FP / max(TP+FP, 1)                                  # Calculate FAR
        return dict(n_tail=int(n), rmse_tail=rmse, mae_tail=mae,
                    TP=int(TP), FP=int(FP), FN=int(FN), TN=int(s["TN"]),
                    recall=recall, precision=precision, csi=csi, far=far)
    return fin(stats["95"]), fin(stats["99"]), fin(stats["17"])

def main():
    print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}\n")
    out_dir = pathlib.Path("results_patch_informer"); out_dir.mkdir(exist_ok=True)

    # Estimated P95/P99 Thresholds (CPU Multi-Process)
    thr_t0 = time.perf_counter()
    cpu_cfg = LoaderCfg(batch=16, device="cpu", use_fp16=False, num_workers=8)
    _, val_cpu, _ = build_loaders(cpu_cfg)
    q95 = compute_quantile_raw(val_cpu, q=0.95, subsample_ratio=0.01, max_batches=30)  # Calculate P95 threshold
    q99 = compute_quantile_raw(val_cpu, q=0.99, subsample_ratio=0.03, max_batches=60)  # Calculate P99 threshold
    thr_time = time.perf_counter() - thr_t0
    print(f"P95 = {q95:.3f} m/s, P99 = {q99:.3f} m/s  (阈值估计 {thr_time:.1f}s)\n")

    # Build a GPU loader + hybrid model
    gpu_cfg = LoaderCfg(batch=8,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        use_fp16=False, num_workers=4)
    _, val_dl, test_dl = build_loaders(gpu_cfg)
    device = torch.device(gpu_cfg.device)

    H, W   = 145, 253
    C_raw  = len(FEATURE_VARS)
    r      = 4
    model  = Informer(C=C_raw + r, H=H, W=W,                                         # Building the Model(Same Parameters)
                      T_out=gpu_cfg.win_out, d_model=256, n_heads=8,
                      e_layers=3, d_layers=2).to(device)
    cond   = Series2GridConditioner(c_in=C_raw, d_model=256, r=r, H=H, W=W,
                                    context_window=gpu_cfg.win_in,
                                    patch_len=1, stride=1,
                                    n_layers=2, n_heads=8, d_ff=256,
                                    revin=True, pe='sincos').to(device)

    ckpt = torch.load("best_patch_informer.pt", map_location=device)             # Load model weight
    model.load_state_dict(ckpt["model"]); cond.load_state_dict(ckpt["cond"])
    model.eval(); cond.eval()
    print(f"Load best_patch_informer.pt\n")

    # Extreme Evaluation: P95/P99/17m/s (Grid)
    def _fmt(tag, d):
        return (f"{tag}: n={d['n_tail']:>7}, RMSE={d['rmse_tail']:.3f}, "
                f"MAE={d['mae_tail']:.3f}, Recall={d['recall']:.3f}, "
                f"Precision={d['precision']:.3f}, CSI={d['csi']:.3f}, FAR={d['far']:.3f}")

    t1 = time.perf_counter()
    v95, v99, v17 = eval_extreme_triple_grid_hybrid(model, cond, val_dl, device, q95, q99, 17.0)    # Evaluate P95/P99/17ms on the validation set
    t2 = time.perf_counter()
    t95, t99, t17 = eval_extreme_triple_grid_hybrid(model, cond, test_dl, device, q95, q99, 17.0)   # Evaluate P95/P99/17ms on the validation set
    t3 = time.perf_counter()
    # Print results
    print("==== Extreme segment (PatchTST+Informer, grid) ====")
    print("Val ", _fmt("@P95",  v95))
    print("Val ", _fmt("@P99",  v99))
    print("Val ", _fmt("@17ms", v17))
    print("Test", _fmt("@P95",  t95))
    print("Test", _fmt("@P99",  t99))
    print("Test", _fmt("@17ms", t17))
    print(f"\nTime：val forward {t2 - t1:.1f}s, test forward {t3 - t2:.1f}s, Whole Time: {t3 - thr_t0:.1f}s")

    # Saving results
    results = {
        "thresholds_raw": {"P95": q95, "P99": q99, "time_sec": thr_time},
        "extreme": {
            "val":  {"P95": v95,  "P99": v99,  "17ms": v17,  "time_sec": t2 - t1},
            "test": {"P95": t95,  "P99": t99,  "17ms": t17,  "time_sec": t3 - t2},
        },
        "total_time_sec": (t3 - thr_t0)
    }
    with open(out_dir / "extreme_metrics_patch_informer.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "extreme_metrics_patch_informer_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "P", "n_tail", "RMSE", "MAE", "Recall", "Precision", "CSI", "FAR"])
        for split, d, tag in (("val", v95, "P95"), ("val", v99, "P99"), ("val", v17, "17ms"),
                              ("test", t95, "P95"), ("test", t99, "P99"), ("test", t17, "17ms")):
            w.writerow([split, tag, d["n_tail"], d["rmse_tail"], d["mae_tail"],
                        d["recall"], d["precision"], d["csi"], d["far"]])

if __name__ == "__main__":
    import multiprocessing as mp, csv, json
    mp.freeze_support()
    thr_t0 = time.perf_counter()
    main()
