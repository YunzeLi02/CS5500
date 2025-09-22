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
                               weight_decay=2e-3)    # 正则
    scaler = None #GradScaler(enabled=True,
                    #init_scale=2.**8,         # ② 小初值
                    #growth_interval=1000,
                    #backoff_factor=0.5)

    total_steps = len(train_dl) * 20                 # 20 epoch 预估
    warm_steps  = 300                           # ==== UPDATE ====
    def lr_lambda(step):
        if step < warm_steps:
            return step / warm_steps
        return 0.5 * (1 + math.cos(
            math.pi * (step - warm_steps) / (total_steps - warm_steps)))

    #scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---------- Train ----------
    best_rmse, patience, max_pat = 1e9, 0, 10
    EPOCHS, ACCUM = 20, 2
    CLIP=0.6
    LAMBDA=0.3
    SCHEDULE = [
        (1, 6, 1.5, 3.0),  # 热身：保 RMSE
        (7, 12, 2.5, 5.0),  # 常规：开始关注尾部
        (13, 20, 3.5, 7.0),  # 加强：收尾提升 P99/17ms
    ]
    MILESTONES = {7, 13}

    LAMBDA_SCHEDULE = [
        (1, 6, 0.20),
        (7, 12, 0.30),
        (13, 20, 0.40),
    ]



    def get_stage_alpha_beta(epoch):
        for lo, hi, a, b in SCHEDULE:
            if lo <= epoch <= hi:
                return a, b
        return SCHEDULE[-1][2], SCHEDULE[-1][3]

    def get_stage_lambda(epoch: int):
        for lo, hi, lam in LAMBDA_SCHEDULE:
            if lo <= epoch <= hi:
                return lam
        return LAMBDA_SCHEDULE[-1][2]


    for epoch in range(1, EPOCHS + 1):
        model.train();
        opt.zero_grad(set_to_none=True)
        tr_loss_w = 0.0  # 记录加权 w-MSE
        tr_loss_p = 0.0  # 记录未加权 plain MSE
        t0 = time.time()

        ALPHA, BETA = get_stage_alpha_beta(epoch)
        LAMBDA = get_stage_lambda(epoch)
        # （可选）打印当前超参，便于回溯
        print(f"lr={opt.param_groups[0]['lr']:.2e}, α/β={ALPHA}/{BETA}, λ={LAMBDA:.2f}")

        if epoch in MILESTONES:
            for pg in opt.param_groups:
                pg["lr"] *= 0.5
            print(f"\n🔻 epoch {epoch}: LR→{opt.param_groups[0]['lr']:.2e}, "
                  f"α/β→{ALPHA}/{BETA}, λ→{LAMBDA:.2f}\n")

        for bi, (X, y) in enumerate(
                tqdm(train_dl, desc=f"[Epoch {epoch:02d}]", leave=False)):

            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---------- 前向 & 加权 MSE ----------
            pred = model(X)  # same shape as y

            # ---- (1) plain MSE：与以前可直接对比 ----
            plain_mse = F.mse_loss(pred, y)  # ==== UPDATE ====（用于反传）
            plain_mse_log = plain_mse.detach()

            # ---- (2) 加权 MSE：用于优化尾部 ----
            w = torch.ones_like(y, dtype=torch.float32)
            w += ALPHA * (y >= P95_z).float()  # ==== UPDATE ====
            w += BETA * (y >= P99_z).float()  # ==== UPDATE ====
            # 关键：批内归一化，保持损失量级稳定（不会整体抬高 MSE）
            w = w / (w.mean().detach() + 1e-8)  # ==== UPDATE ====
            w_mse = ((pred - y) ** 2 * w).mean()

            loss_back = ((1.0 - LAMBDA) * plain_mse + LAMBDA * w_mse) / ACCUM


            # ---------- backward ----------
            #scaler.scale(w_mse).backward()
            loss_back.backward()

            tr_loss_p += plain_mse_log.item() * X.size(0)# 记录未加权
            tr_loss_w += w_mse.item() * X.size(0) # 记录加权


            # ---------- 累积 step ----------
            if (bi + 1) % ACCUM != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            if not torch.isfinite(loss_back):  # ==== UPDATE ====
                print(f"⚠ nan/inf loss ep{epoch} b{bi}, skip")  # ==== UPDATE ====
                opt.zero_grad(set_to_none=True)
                continue

            opt.step()  # ← 纯 FP32 直接 step
            opt.zero_grad(set_to_none=True)
            #scheduler.step()



        tr_loss_w /= len(train_dl.dataset)
        tr_loss_p /= len(train_dl.dataset)

        num_batches = len(train_dl)
        if (num_batches % ACCUM) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)

        if epoch % 2 == 0:  # 每 2 轮抽样一次即可
            model.eval()
            with torch.no_grad():
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
        with torch.no_grad():
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
    with torch.no_grad():

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
