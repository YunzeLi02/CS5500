
from dataset import build_loaders, LoaderCfg, FEATURE_VARS
from informer import Informer
import torch.nn.functional as F
import math, time
from tqdm import tqdm
import torch
from torch import nn, Tensor
from layers.PatchTST_backbone import PatchTST_backbone

class PatchTST_Featurizer(PatchTST_backbone):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, L, C]
        if self.revin:
            z = self.revin_layer(z, 'norm')

        # Transform into [B, C, L] and then perform patching (time is the last dimension)
        z = z.permute(0, 2, 1).contiguous()   # [B, C, L]

        if self.padding_patch == 'end':
            # nn.ReplicationPad1d needs [B, C, L]
            z = self.padding_patch_layer(z)

        # Cut patches along the time dimension: [B, C, patch_num, patch_len]
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Reorder to the encoder-required shape:[B, C, patch_len, patch_num]
        z = z.permute(0, 1, 3, 2).contiguous()

        # Pass through the TST encoder to get tokens: [B, C, d_model, patch_num]
        tok = self.backbone(z)
        return tok

class Series2GridConditioner(nn.Module):
    def __init__(self, c_in:int, d_model:int, r:int, H:int, W:int,
                 context_window:int,
                 patch_len:int=1, stride:int=1,
                 n_layers:int=2, n_heads:int=8, d_ff:int=256,
                 revin:bool=False, pe:str='sincos'):
        super().__init__()
        self.H, self.W, self.r = H, W, r

        # pass context_window = L to PatchTST
        self.feat = PatchTST_Featurizer(
            c_in=c_in, context_window=context_window, target_window=1,
            patch_len=patch_len, stride=stride, padding_patch=None,
            n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=0.1, attn_dropout=0.1, pre_norm=True,
            head_type='flatten', head_dropout=0.0,
            revin=revin, affine=True, subtract_last=False, pe=pe
        )  # Initialize the PatchTST feature extractor
        # Aggregate [C, D] into a per-time-step vector, then project linearly to r
        self.merge = nn.Sequential(
            nn.Conv1d(in_channels=c_in*d_model, out_channels=128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(in_channels=128, out_channels=r, kernel_size=1),
        )
        # r spatial bases (low rank): [r, H, W]
        self.spatial_basis = nn.Parameter(torch.randn(r, H, W) * 0.01)

    def forward(self, X_grid: Tensor) -> Tensor:
        # X_grid: [B,L,C,H,W] （Same as Informer）
        B, L, C, H, W = X_grid.shape
        x_series = X_grid.mean(dim=(-1, -2))              # [B,L,C]
        tokens   = self.feat(x_series)                    # [B,C,D,L]
        tok_flat = tokens.permute(0,3,1,2).reshape(B, L, -1)  # [B,L,C*D]
        # Map to r coefficients (treat time as the length for the 1D conv)
        coeff = self.merge(tok_flat.transpose(1,2)).transpose(1,2)  # [B,L,r]
        # Expand into r spatial maps and treat r as 'new channels'
        # coeff: [B,L,r] ; basis: [r,H,W] → grid_r: [B,L,r,H,W]
        grid_r = torch.einsum('blr,rhw->blrhw', coeff, self.spatial_basis)
        # Return to the training loop to concatenate with the original input
        return grid_r

def squeeze_to_series(y: torch.Tensor):
    if y.dim() == 5 and y.size(2) == 1:  # [B,T,1,H,W]
        y = y.squeeze(2)                 # [B,T,H,W]
    if y.dim() == 4:
        return y.mean(dim=(-1,-2)), y    # ([B,T], [B,T,H,W])
    if y.dim() == 2:
        return y, None
    raise RuntimeError(f"bad y shape {tuple(y.shape)}")

def to_BTHW(t):
    # Normalize both pred and y to shape [B, T, H, W]
    if t.dim() == 5 and t.size(2) == 1:     # [B,T,1,H,W] -> [B,T,H,W]
        return t.squeeze(2)
    if t.dim() == 4:                        # Already [B,T,H,W]
        return t
    raise RuntimeError(f"Unexpected shape for grid tensor: {tuple(t.shape)}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'



    #---- Data ----
    train_cfg = LoaderCfg(device="cpu", batch=8, num_workers=4, use_fp16=False)
    train_dl, val_dl, test_dl = build_loaders(train_cfg)

    H, W = 145, 253
    C_raw = len(FEATURE_VARS)    #Feature Varable
    r = 4                        # r value

    # Use existing Informer directly, just change C to C_raw + r
    model = Informer(
        C=C_raw + r, H=H, W=W,
        T_out=train_cfg.win_out,
        d_model=256, n_heads=8,
        e_layers=3, d_layers=2
    ).to(device)

    # Conditioner: Expand the PatchTST features into r images and stitch them back to the original input.
    cond = Series2GridConditioner(
        c_in=C_raw, d_model=256, r=r, H=H, W=W,
        context_window=train_cfg.win_in,
        patch_len=1, stride=1,
        n_layers=2, n_heads=8, d_ff=256,
        revin=True, pe='sincos'
    ).to(device)

    params = list(model.parameters()) + list(cond.parameters())
    opt    = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-3)        # Build AdamW Optimizer
    epochs, clip = 20, 0.8                                                # Training epoch, gradient clipping



    #---- Train ----
    best_val, patience, max_pat = 1e9, 0,10
    for ep in range(1, epochs + 1):
        model.train();
        cond.train()
        t0 = time.time()
        tr_se_series, tr_n_series = 0.0, 0
        tr_se_grid, tr_n_grid = 0.0, 0

        pbar = tqdm(train_dl, desc=f"[Epoch {ep:02d}]")
        for X, y in pbar:
            X = X.to(device)  # [B,L,C,H,W]
            y = y.to(device)  # [B,T,(1),H,W]
            y_series, y_grid = squeeze_to_series(y)  # [B,T], [B,T,(1),H,W] or None

            # generate conditional channels + concatenate
            X_cond = cond(X)  # [B,L,r,H,W]
            X_aug = torch.cat([X, X_cond], dim=2)  # [B,L,C+r,H,W]

            pred = model(X_aug)
            pred_grid = to_BTHW(pred)  # Ensure transfer to [B,T,H,W]

            #Normalize labels to [B, T, H, W]
            y_grid = to_BTHW(y_grid) if y_grid is not None else None

            # [B,T],for Train MSE
            ps = pred_grid.mean(dim=(-1, -2))

            # Training loss: Prioritize grid-based calibration; align with grid data when available. Otherwise, downgrade to sequence-based calibration.
            if y_grid is not None:
                loss = F.mse_loss(pred_grid, y_grid)
            else:
                loss = F.mse_loss(ps, y_series)
            # backward
            loss.backward()                                                     # Back propagation
            torch.nn.utils.clip_grad_norm_(params, clip)                        # Gradient Clipping
            opt.step()                                                          # Update Parameters
            opt.zero_grad(set_to_none=True)                                     # Zeroing Gradient

            # train MSE Collect only
            # series format
            tr_se_series += F.mse_loss(ps.detach(), y_series.detach(), reduction='sum').item()
            tr_n_series += y_series.numel()
            # grid  format
            if y_grid is not None:
                tr_se_grid += F.mse_loss(pred_grid.detach(), y_grid.detach(), reduction='sum').item()
                tr_n_grid += y_grid.numel()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Train MSE(both series/grid)
        tr_mse_series = tr_se_series / max(tr_n_series, 1)
        tr_mse_grid = (tr_se_grid / tr_n_grid)


        #---- Validation----
        model.eval();
        cond.eval()
        se_s, n_s = 0.0, 0
        se_g, n_g = 0.0, 0
        with torch.no_grad():
            for X, y in val_dl:  # Loop val_dl
                X = X.to(device);
                y = y.to(device)
                ys, yg = squeeze_to_series(y)  # ys:[B,T], yg:[B,T,H,W]/[B,T,1,H,W]
                yg = to_BTHW(yg)

                X_aug = torch.cat([X, cond(X)], dim=2)                 # Concatenation Condition Channel
                pred = model(X_aug)                                           # Predict the results
                pred = to_BTHW(pred)                                          # Ensure the predicted result has the shape [B,T,H,W]

                ps = pred.mean(dim=(-1, -2))  # [B,T]
                se_s += F.mse_loss(ps, ys, reduction='sum').item();          # Collect series Validation Loss
                n_s += ys.numel()
                se_g += F.mse_loss(pred, yg, reduction='sum').item();        # Collect grid Validation Loss
                n_g += yg.numel()

        val_rmse_series = math.sqrt(se_s / max(n_s, 1))                      # Calculate the series Validation RMSE
        val_rmse_grid = math.sqrt(se_g / max(n_g, 1))                        # Calculate the grid Validation RMSE

        print(f"Epoch {ep:02d} | train MSE (series) {tr_mse_series:.3f} "
              f"| train MSE (grid) {tr_mse_grid:.3f} "
              f"| val RMSE (series) {val_rmse_series:.3f} "
              f"| val RMSE (grid) {val_rmse_grid:.3f} "
              f"| time {(time.time()-t0)/60:.1f}m")

        # Early termination primarily uses the series/grid format
        if val_rmse_grid < best_val:
            best_val, patience = val_rmse_grid, 0
            torch.save(dict(model=model.state_dict(), cond=cond.state_dict()),
                       "best_patchtst_informer.pt")
        else:
            patience += 1
            if patience >= max_pat: break

    #---- Test（series & grid）----
    ckpt = torch.load("best_patchtst_informer.pt", map_location=device)    # Load optimal model parameters
    model.load_state_dict(ckpt["model"]);
    cond.load_state_dict(ckpt["cond"])
    model.eval();
    cond.eval()
    se_s, n_s = 0.0, 0
    se_g, n_g = 0.0, 0
    with torch.no_grad():
        for X, y in test_dl:  # test_dl Loop
            X = X.to(device);
            y = y.to(device)
            ys, yg = squeeze_to_series(y);
            yg = to_BTHW(yg)

            X_aug = torch.cat([X, cond(X)], dim=2)         # Concatenation Condition Channel
            pred = model(X_aug);                                  # Predict the results
            pred = to_BTHW(pred)                                  # Ensure the predicted result has the shape [B,T,H,W]

            ps = pred.mean(dim=(-1, -2))
            se_s += F.mse_loss(ps, ys, reduction='sum').item();     # Collect series Testing Loss
            n_s += ys.numel()
            se_g += F.mse_loss(pred, yg, reduction='sum').item();   # Collect grid Testing Loss
            n_g += yg.numel()
    print(f"\nBest val RMSE (grid) {best_val:.3f} | "
          f"Test RMSE (series) {math.sqrt(se_s/n_s):.3f} | "
          f"Test RMSE (grid) {math.sqrt(se_g/n_g):.3f}")

if __name__ == "__main__":
    main()
