from __future__ import annotations
import math, platform, torch, torch.nn.functional as F, time
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dataset import build_loaders, LoaderCfg, FEATURE_VARS
from informer import Informer
import numpy as np

# ---------- 环境 & 全局加速开关 ----------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True


scale = np.load("scaler_train.npz")
mean_ws10, std_ws10 = scale["mean"][8], scale["std"][8]

P95_z = (8.23  - mean_ws10) / std_ws10   # 8.23 = 你之前算出的 P95 (m/s)
P99_z = (11.48 - mean_ws10) / std_ws10
ALPHA = 3.0        # 权重系数  y ≥ P95
BETA  = 6.0

def main():
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"CUDA OK: {torch.cuda.is_available()}\n")

    # ---------- DataLoader：CPU 输出，多进程 ----------
    train_cfg = LoaderCfg(device="cpu", batch=8, num_workers=4, use_fp16=False)
    train_dl, val_dl, test_dl = build_loaders(train_cfg)

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Informer(
        C=len(FEATURE_VARS), H=145, W=253,
        T_out=train_cfg.win_out,
        d_model=320, n_heads=10,
        e_layers=2, d_layers=3
    ).to(device)                            # channels_last 去掉以免 5-D 冲突

    # ---------- Optim / Scheduler / AMP ----------
    opt    = torch.optim.AdamW(model.parameters(),   # ==== UPDATE ====
                               lr=2.5e-4,              # ↓ 安全起步
                               weight_decay=5e-3)    # 正则
    scaler = GradScaler(enabled=True,
                    init_scale=2.**8,         # ② 小初值
                    growth_interval=1000,
                    backoff_factor=0.5)

    total_steps = len(train_dl) * 20                 # 20 epoch 预估
    warm_steps  = 300                           # ==== UPDATE ====
    def lr_lambda(step):
        if step < warm_steps:
            return step / warm_steps
        return 0.5 * (1 + math.cos(
            math.pi * (step - warm_steps) / (total_steps - warm_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---------- Train ----------
    best_rmse, patience, max_pat = 1e9, 0, 10
    EPOCHS, ACCUM = 20, 2
    CLIP=0.8
    skip_overflows=0

    for epoch in range(1, EPOCHS + 1):
        model.train();
        tr_loss_w = 0.0  # 记录加权 w-MSE
        tr_loss_p = 0.0  # 记录未加权 plain MSE
        t0 = time.time()

        for bi, (X, y) in enumerate(
                tqdm(train_dl, desc=f"[Epoch {epoch:02d}]", leave=False)):

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---------- 前向 & 加权 MSE ----------
            with autocast():
                pred = model(X)  # same shape as y

                # ---- (1) plain MSE：与以前可直接对比 ----
                plain_mse = F.mse_loss(pred, y)

                # ---- (2) 加权 MSE：用于优化尾部 ----
                w = torch.ones_like(y, dtype=torch.float32)
                w += ALPHA * (y >= P95_z)
                w += BETA * (y >= P99_z)
                w_mse = ((pred - y) ** 2 * w).mean() / ACCUM  # 注意除 ACCUM


            # ---------- backward ----------
            scaler.scale(w_mse).backward()

            # ---------- 累积 step ----------
            if (bi + 1) % ACCUM == 0:
                # -------- 检测是否有 inf ----------
                found_inf = scaler.unscale_(opt)  # bool tensor
                if found_inf:  # 保险丝 ①
                    skip_overflows += 1
                    print(f"⚠ overflow  ep{epoch}  b{bi}, skip {skip_overflows}")
                    opt.zero_grad(set_to_none=True)
                    scaler.update()  # scale /= 2
                    if skip_overflows >= 3:  # 保险丝 ②
                        for pg in opt.param_groups:
                            pg["lr"] *= 0.5  # 自动再降 LR
                        ALPHA, BETA = max(ALPHA - 1, 1), max(BETA - 2, 2)
                        skip_overflows = 0
                        print(f"🔻 LR-> {pg['lr']:.2e}  α/β-> {ALPHA}/{BETA}")
                    continue  # 完全跳过本批

                skip_overflows = 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

                scaler.step(opt)  # 更新权重
                scaler.update()  # 更新 scaler
                opt.zero_grad(set_to_none=True)  # 清梯度
                scheduler.step()  # 学习率调度

            tr_loss_w += w_mse.item() * ACCUM * X.size(0)  # 记录加权
            tr_loss_p += plain_mse.item() * X.size(0)  # 记录未加权


        tr_loss_w /= len(train_dl.dataset)
        tr_loss_p /= len(train_dl.dataset)

        if epoch % 2 == 0:  # 每 2 轮抽样一次即可
            model.eval()
            with torch.no_grad(), autocast():
                Xv, yv = next(iter(val_dl))
                Xv = Xv.to(device);
                yv = yv.to(device)
                pv = model(Xv)

            tail95 = yv >= P95_z
            recall95 = (pv[tail95] >= yv[tail95]).float().mean().item()
            tail99 = yv >= P99_z
            recall99 = (pv[tail99] >= yv[tail99]).float().mean().item()
            print(f" | P95 rec {recall95:.3f} P99 rec {recall99:.3f}")

        print(f"Epoch {epoch:02d} done in {(time.time() - t0) / 60:.1f} min | "
              f"plain MSE {tr_loss_p:.3f} | w-MSE {tr_loss_w:.2f}", end=" | ")

        # ---------- Validate ----------
        model.eval(); se = 0.0; n = 0
        with torch.no_grad(), autocast():
            for X, y in val_dl:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(X)
                se += F.mse_loss(pred, y, reduction='sum').item()
                n  += y.numel()
        val_rmse = math.sqrt(se / n)
        print(f"val RMSE {val_rmse:.3f}")

        # ---------- Early-stop ----------
        if val_rmse < best_rmse:
            best_rmse, patience = val_rmse, 0
            torch.save(model.state_dict(), "best_informer_2.pt")
        else:
            patience += 1
            if patience >= max_pat: break

    # ---------- Test ----------
    model.load_state_dict(torch.load("best_informer_2.pt"))
    model.eval(); se = 0.0; n = 0
    with torch.no_grad(), autocast():
        for X, y in test_dl:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(X)
            se += F.mse_loss(pred, y, reduction='sum').item()
            n  += y.numel()
    print(f"\nBest val RMSE {best_rmse:.3f} | Test RMSE {math.sqrt(se/n):.3f}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
