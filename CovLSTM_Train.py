from __future__ import annotations
import os, math, sys, platform, torch, torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from dataclasses import dataclass
from dataset import build_loaders, LoaderCfg, FEATURE_VARS
import dask


def main():
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | "
          f"CUDA OK: {torch.cuda.is_available()}\n")                      # PyTorch and CUDA check

    #---- Data ----
    dask.config.set(scheduler="threads")
    data_cfg = LoaderCfg(batch=8, use_fp16=True, num_workers=2)
    train_dl, val_dl, test_dl = build_loaders(data_cfg)

    @dataclass
    class TrainCfg:
        hidden: int = 64          # Hidden layer size
        lr: float = 1e-4          # Learning rate
        epochs: int = 20          # Training epoch
        patience: int = 3         # Early Stop Patience Value
        weight_decay: float = 0.0 # Weight decay

    train_cfg = TrainCfg() #Build training configuration
    #---- Model ----
    from CovLSTM import ConvLSTM                      # Model input
    C      = len(FEATURE_VARS)                        # Number of Feature Variables
    T_out  = data_cfg.win_out                         #Output step size
    device = torch.device(data_cfg.device)

    model = ConvLSTM(in_channels=C,
                     hidden_channels=train_cfg.hidden,
                     out_channels=1,
                     out_len=T_out).to(device)

    if os.name != "nt" and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
    else:
        print("torch.compile disable。")

    #---- Optim / AMP ----
    opt    = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)           # Build AdamW Optimizer
    scaler = GradScaler(enabled=(device.type=="cuda" and data_cfg.use_fp16))  # Create a GradScaler for mixed-precision training

    #---- Train ----
    best, patience, max_pat = 1e9, 0, 3
    for epoch in range(1, train_cfg.epochs+1):
        model.train(); tr_loss = 0.0
        for X, y in tqdm(train_dl, desc=f"[Epoch {epoch:02d}]"):
            #forward
            opt.zero_grad(set_to_none=True)                    # Reset gradients to zero
            with autocast(enabled=scaler.is_enabled()):
                pred = model(X)                                # Forward propagation
                loss = F.mse_loss(pred, y)                     # Calculate Mean Squared Error Loss
            #backward
            scaler.scale(loss).backward()                     # Back propagation
            scaler.step(opt); scaler.update()                 # Update the optimizer and scaler
            tr_loss += loss.item() * X.size(0)
        tr_loss /= len(train_dl.dataset)                      # Accumulated Training Loss

        #---- Validation ----
        model.eval(); rmse_sum = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                with autocast(enabled=scaler.is_enabled()):
                    pred = model(X)                                  # Forward propagation
                rmse_sum += math.sqrt(F.mse_loss(pred, y).item()) * X.size(0)   # Cumulative Verification RMSE
        val_rmse = rmse_sum / len(val_dl.dataset)                    # Calculate the average validation RMSE
        print(f"Epoch {epoch:02d} | train MSE {tr_loss:.3f} | val RMSE {val_rmse:.3f}") # Print Validation RMSER


        #---- Early-stop -----
        if val_rmse < best:
            best, patience = val_rmse, 0
            torch.save(model.state_dict(), "best_convlstm.pt")
        else:
            patience += 1
            if patience >= max_pat:
                print("Early stop.")
                break

    #---- Test ----
    model.load_state_dict(torch.load("best_convlstm.pt")) # Load optimal model parameters
    model.eval(); rmse_sum = 0.0
    with torch.no_grad():
        for X, y in test_dl:
            with autocast(enabled=scaler.is_enabled()):
                pred = model(X)   # Forward propagation
            rmse_sum += math.sqrt(F.mse_loss(pred, y).item()) * X.size(0)  # Cumulative Verification RMSE
    print(f"Test RMSE {rmse_sum/len(test_dl.dataset):.3f}") #Print Test RMSE

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
