import math, time, torch, torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from layers.PatchTST_backbone import PatchTST_backbone
from dataset import build_loaders,LoaderCfg
import numpy as np

scale = np.load("scaler_train.npz")
mean_ws10, std_ws10 = scale["mean"][8], scale["std"][8]

P95_TH = (8.23  - mean_ws10) / std_ws10                      # Calculate the P95 threshold
P99_TH = (11.48 - mean_ws10) / std_ws10                      # Calculate the P99 threshold

def rmse(x, y): return (F.mse_loss(x, y).sqrt())            # Calculate RMSE

def squeeze_to_series_and_grid(y):
    # Transfer y into a suitable format
    if y.dim() == 5 and y.size(2) == 1:
        y = y.squeeze(2)                  # [B, T, H, W]
    y_grid = y                            # [B, T, H, W]
    y_series = y.mean(dim=(-1, -2))       # [B, T]
    return y_series, y_grid

@torch.no_grad()
def eval_val(model, loader, device):
    model.eval()
    se_series = 0.0; n_series = 0
    se_grid   = 0.0; n_grid   = 0

    for x, y in loader:
        x = x.to(device); y = y.to(device)
        y_series, y_grid = squeeze_to_series_and_grid(y)  # [B,T], [B,T,H,W]

        pred = model(x)
        if pred.dim() == 4:
            # Model Output Grid
            pred_series = pred.mean(dim=(-1, -2))         # [B,T]
            se_series += F.mse_loss(pred_series, y_series, reduction='sum').item()
            n_series  += y_series.numel()

            se_grid   += F.mse_loss(pred, y_grid, reduction='sum').item()
            n_grid    += y_grid.numel()
        else:
            # Model Output Series
            se_series += F.mse_loss(pred, y_series, reduction='sum').item()
            n_series  += y_series.numel()

            B, T      = pred.shape
            H, W      = y_grid.shape[-2:]
            pred_grid = pred.unsqueeze(-1).unsqueeze(-1).expand(B, T, H, W)
            se_grid   += F.mse_loss(pred_grid, y_grid, reduction='sum').item()
            n_grid    += y_grid.numel()

    rmse_series = math.sqrt(se_series / max(n_series, 1))     # Calculate the RMSE of the Series
    rmse_grid   = math.sqrt(se_grid   / max(n_grid,   1))     # Calculate the RMSE of the Grid
    return rmse_series, rmse_grid


def squeeze_to_series(y):  # y: [B,T,1,H,W] / [B,T,H,W] / [B,T]
    if y.dim() == 5 and y.size(2) == 1:
        y = y.squeeze(2)                 # [B,T,H,W]
    if y.dim() == 4:
        y_series = y.mean(dim=(-1, -2))  # [B,T] Global Mean
        return y_series, y
    elif y.dim() == 2:
        return y, None
    else:
        raise RuntimeError(f"Unexpected y shape: {tuple(y.shape)}")
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #---- Data ----
    train_cfg = LoaderCfg(device="cpu", batch=8, num_workers=4, use_fp16=False)   # Configure Training Data Loader

    # Then pass cfg as the sole argument to build_loaders.
    train_dl, val_dl, test_dl = build_loaders(train_cfg)                          # Build training, validation, and test data loaders

    #---- Model ----
    model = PatchTST_backbone(                                                    # Initialize Informer model parameters
        c_in=12, context_window=12, target_window=3,
        patch_len=4, stride=1, padding_patch=None,
        n_layers=3, d_model=256, n_heads=8, d_ff=512,
        dropout=0.1, attn_dropout=0.1, pre_norm=True,
        head_type='spatial',  # ← grid using
        head_dropout=0.1,
        target_idx=8,  # series only
        revin=True, affine=True, subtract_last=False,
        pe='sincos',
        out_mode="grid",  # ← grid output
        grid_H=145, grid_W=253,
        spatial_rank=96
    ).to(device)

   #---- Optim / Scheduler ----
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)               # Build AdamW Optimizer
    epochs = 20
    steps_per_epoch = len(train_dl)
    warm = 300
    total = steps_per_epoch * epochs
    def lr_lambda(step):
        if step < warm: return step / max(1, warm)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, (total - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val, patience, max_pat = 1e9, 0, 3
    clip = 0.8


    #---- Train ----
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tr_se_series, tr_n_series = 0.0, 0  # Training set series MSE
        tr_se_grid, tr_n_grid = 0.0, 0  # Training set grid MSE

        pbar = tqdm(train_dl, desc=f"[Epoch {ep:02d}]")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            # 2 Label Format
            y_series, y_grid = squeeze_to_series(y)                         # [B,T], [B,T,H,W] or None

            # forward
            pred = model(x)                                                 # Forward propagation

            # Obtain pred_series（for calculate train MSE）
            if pred.dim() == 4:                                             # [B,T,H,W]
                pred_grid = pred
                pred_series = pred_grid.mean(dim=(-1, -2))                  # [B,T]
                if y_grid is not None:
                    loss = F.mse_loss(pred_grid, y_grid)                    # grid format loss
                else:
                    loss = F.mse_loss(pred_series, y_series)                # series format loss
            else:  # model output [B,T]
                pred_series = pred
                loss = F.mse_loss(pred_series, y_series)

            # backward
            loss.backward()                                                 # Back propagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)        # Gradient Clipping
            opt.step()                                                      # Update Parameters
            if sched is not None:
                sched.step()                                               # Update Learning Rate
            opt.zero_grad(set_to_none=True)                                # Zeroing Gradient

            # train MSE Collect only
            # series format
            tr_se_series += F.mse_loss(pred_series.detach(), y_series.detach(), reduction='sum').item()
            tr_n_series += y_series.numel()

            # grid  format
            if y_grid is not None:
                if pred.dim() == 2:
                    pred_grid_stat = pred_series.detach()[..., None, None].expand_as(y_grid)
                else:
                    pred_grid_stat = pred_grid.detach()
                tr_se_grid += F.mse_loss(pred_grid_stat, y_grid.detach(), reduction='sum').item()
                tr_n_grid += y_grid.numel()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Train MSE(both series/grid)
        tr_mse_series = tr_se_series / max(tr_n_series, 1)
        tr_mse_grid = (tr_se_grid / tr_n_grid) if tr_n_grid > 0 else float('nan')

        #---- Validtation ----
        val_rmse_series_z, val_rmse_grid_z = eval_val(model, val_dl, device)

        # Output
        print(
            f"Epoch {ep:02d} | train MSE (series) {tr_mse_series:.3f} "
            f"| train MSE (grid) {tr_mse_grid:.3f} "
            f"| val RMSE (series) {val_rmse_series_z:.3f} "
            f"| val RMSE (grid) {val_rmse_grid_z:.3f} "
            f"| time {(time.time() - t0) / 60:.1f}m"
        )
        # Early termination primarily uses the series/grid format
        if val_rmse_series_z < best_val:
            best_val, patience = val_rmse_series_z, 0
            torch.save(model.state_dict(), "best_patchtst_series.pt")
        else:
            patience += 1
            if patience >= max_pat:
                break

    #---- Test ----
    model.load_state_dict(torch.load("best_patchtst_series.pt", map_location=device))         # Load optimal model parameters
    model.eval()

    se_series, n_series = 0.0, 0
    se_grid, n_grid = 0.0, 0

    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)

            # Processing y_series ([B,T]) and y_grid ([B,T,H,W])
            if y.dim() == 5 and y.size(2) == 1:
                y = y.squeeze(2)  # [B,T,H,W]
            if y.dim() == 4:
                y_grid = y  # [B,T,H,W]
                y_series = y.mean(dim=(-1, -2))  # [B,T], Global Mean
            elif y.dim() == 2:
                y_series = y  # [B,T]
                y_grid = None
            else:
                raise RuntimeError(f"Unexpected y shape: {tuple(y.shape)}")

            # Test forward
            pred = model(x)
            if pred.dim() == 4:
                pred_grid = pred  # [B,T,H,W]
                pred_series = pred.mean(dim=(-1, -2))  # [B,T]
            elif pred.dim() == 2:
                pred_series = pred  # [B,T]
                pred_grid = None
            else:
                raise RuntimeError(f"Unexpected pred shape: {tuple(pred.shape)}")

            # series format  RMSE
            se_series += F.mse_loss(pred_series, y_series, reduction='sum').item()
            n_series += y_series.numel()

            # grid format RMSE
            if y_grid is not None:
                if pred_grid is None:
                    pred_grid = pred_series[..., None, None].expand_as(y_grid)  # [B,T,H,W]
                se_grid += F.mse_loss(pred_grid, y_grid, reduction='sum').item()
                n_grid += y_grid.numel()

    test_rmse_series_z = math.sqrt(se_series / n_series)                           # Cumulative Verification RMSE for Series
    test_rmse_grid_z = math.sqrt(se_grid / n_grid) if n_grid > 0 else float('nan')  # Cumulative Verification RMSE


    print(f"\nBest val RMSE {best_val:.3f} | "
          f"Test RMSE (series) {test_rmse_series_z:.3f} | "
          f"Test RMSE (grid) {test_rmse_grid_z:.3f}")

if __name__ == "__main__":
    main()
