from __future__ import annotations
import math, platform, torch, torch.nn.functional as F, time
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dataset import build_loaders, LoaderCfg, FEATURE_VARS
from informer import Informer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True

def main():
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"CUDA OK: {torch.cuda.is_available()}\n")

    #---- DataLoader ----
    train_cfg = LoaderCfg(device="cpu", batch=8, num_workers=4, use_fp16=False)
    train_dl, val_dl, test_dl = build_loaders(train_cfg)

    #---- Model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = Informer(
        C=len(FEATURE_VARS), H=145, W=253,
        T_out=train_cfg.win_out,
        d_model=256, n_heads=8,
        e_layers=3, d_layers=2
    ).to(device)

    #---- Optim / Scheduler / AMP ----
    opt    = torch.optim.AdamW(model.parameters(),
                               lr=2.5e-4,
                               weight_decay=5e-3)
    scaler = GradScaler(enabled=True)

    total_steps = len(train_dl) * 20
    warm_steps  = 300
    def lr_lambda(step):
        if step < warm_steps:
            return step / warm_steps
        return 0.5 * (1 + math.cos(
            math.pi * (step - warm_steps) / (total_steps - warm_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    #---- Train ----
    best_rmse, patience, max_pat = 1e9, 0, 10
    EPOCHS, ACCUM = 20, 1
    for epoch in range(1, EPOCHS + 1):
        model.train(); tr_loss = 0.0; t0 = time.time()
        for bi, (X, y) in enumerate(tqdm(train_dl, desc=f"[Epoch {epoch:02d}]", leave=False)):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # forward
            with autocast():
                pred = model(X)
                loss = F.mse_loss(pred, y) / ACCUM
            # backward
            scaler.scale(loss).backward()

            if (bi + 1) % ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
            tr_loss += loss.item() * ACCUM * X.size(0)

        tr_loss /= len(train_dl.dataset)
        print(f"Epoch {epoch:02d} done in {(time.time()-t0)/60:.1f} min | "
              f"train MSE {tr_loss:.3f}", end=" | ")

        #---- Validation ----
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

        #---- Early-stop -----
        if val_rmse < best_rmse:
            best_rmse, patience = val_rmse, 0
            torch.save(model.state_dict(), "best_informer.pt")
        else:
            patience += 1
            if patience >= max_pat: break

    #---- Test ----
    model.load_state_dict(torch.load("best_informer.pt"))
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
