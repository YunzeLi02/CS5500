from __future__ import annotations
import numpy as np, xarray as xr, torch, time, pathlib, os
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Tuple

# Variable Configuration
FEATURE_VARS = [
    "u10", "v10", "u100", "v100",
    "t2m", "sp", "slor", "sdor",
    "ws10", "ws100", "shear", "blh"
]
TARGET_VAR = "ws10"

# Derive variables
def _derive(ds: xr.Dataset) -> xr.Dataset:
    need = []
    if "ws10"  not in ds: need.append("ws10")
    if "ws100" not in ds: need.append("ws100")
    if "shear" not in ds: need.append("shear")
    if not need:
        return ds
# Copy the dataset to avoid modifying the original data
    ds = ds.copy()
    if "ws10" in need:
        ds["ws10"]  = np.hypot(ds["u10"],  ds["v10"])  # Calculate 'ws10'
    if "ws100" in need:
        ds["ws100"] = np.hypot(ds["u100"], ds["v100"]) # Calculate 'ws100'
    if "shear" in need:
        ds["shear"] = ds["ws100"] - ds["ws10"] # Calculate 'shear'
    return ds

# Simple Timer Decorator
def _timed(tag: str):
    def deco(fn):
        def wrap(*a, **k):    #Record start time
            t0 = time.perf_counter()
            out = fn(*a, **k)
            print(f"{tag:<18} {time.perf_counter()-t0:6.2f}s") # Print function execution time
            return out
        return wrap
    return deco

# Main Dataset
class ERA5Lazy(Dataset):

    @_timed("open_zarr")
    def _open_ds(self, path: os.PathLike):
        return xr.open_zarr(path, consolidated=True)  # Open Zarr

    def __init__(
        self,
        *,
        zarr_path      : str | os.PathLike,
        split          : str,
        win_in         : int,
        win_out        : int,
        train_end      : str,
        val_end        : str,
        device         : str = "cpu",
        scaler_cache   : str | None = None,
        use_fp16       : bool = False,
    ):
        assert split in {"train", "val", "test"}
        self.device   = torch.device(device)
        self.use_fp16 = use_fp16
        self.win_in, self.win_out = win_in, win_out

        # Read the dataset
        ds = self._open_ds(zarr_path)       # Open dataset
        ds = ds[FEATURE_VARS + [TARGET_VAR]]                   # Select Feature Var. and Target Var.
        ds = ds.sel(time=slice("2022", "2024"))                # Select Time domain
        ds = _derive(ds)   #Calculate derive Variables

        # Generate sample start index
        offset = win_in + win_out - 1
        times  = ds.time.values[offset:]

        m_train = times <  np.datetime64(train_end)
        m_val   = (times >= np.datetime64(train_end)) & (times <= np.datetime64(val_end))
        m_test  = times >  np.datetime64(val_end)
        mask    = {"train": m_train, "val": m_val, "test": m_test}[split]

        total   = len(ds.time) - win_in - win_out + 1
        starts  = np.where(mask)[0] - offset
        self.starts = starts[(starts >= 0) & (starts < total)]

        # Mean/Variance scaler
        sc_path = pathlib.Path(scaler_cache) if scaler_cache else None
        if sc_path and sc_path.exists():
            arr   = np.load(sc_path)
            mean, std = arr["mean"], arr["std"]
            print(f"load scaler        {sc_path}  (cached)")
        else:
            idx_train = np.where(m_train)[0] + offset                      # Calculate the scaler using only the training period time step
            mean, std = self._fit_scaler(ds, idx_train)
            if sc_path:
                np.savez(sc_path, mean=mean, std=std)
                print(f"save scaler        {sc_path}")

        self.mean = mean.astype("float32")                                          # Set mean value
        self.std  = np.clip(std, 1e-6, None).astype("float32")         # Set std.

        print("mean/std shape      ", self.mean.shape)
        print(f"{split:<6} samples      {len(self.starts)}")
        self.ds = ds

    # Calculate mean/variance scaler
    @_timed("fit scaler")
    def _fit_scaler(self, ds: xr.Dataset, idx_sel) -> Tuple[np.ndarray, np.ndarray]:
        sel = ds[FEATURE_VARS].isel(time=idx_sel)                               # Select feature var.
        # stack → (C, N)
        arr = (sel.to_array()
                  .stack(z=("time", "latitude", "longitude"))
                  .values)                                                      # Transfer to numpy array
        mean = arr.mean(-1)                                                     # Calculate mean
        std  = arr.std(-1)                                                      # Calculate std.
        return mean, std

    # Torch Dataset API
    def __len__(self): return len(self.starts)

    def _scale(self, x: np.ndarray):
        return (x - self.mean[None,:,None,None]) / self.std[None,:,None,None]

    def __getitem__(self, idx: int):
        i      = self.starts[idx]
        t_in   = slice(i, i+self.win_in)
        t_out  = slice(i+self.win_in, i+self.win_in+self.win_out)

        X = (self.ds[FEATURE_VARS]
                .isel(time=t_in)
                .to_array()
                .transpose("time", "variable", "latitude", "longitude")
                .values)                                # (T_in,C,H,W)

        y = self.ds[TARGET_VAR].isel(time=t_out).values[..., None]

        X = self._scale(X).astype("float16" if self.use_fp16 else "float32")
        y = y.astype("float16" if self.use_fp16 else "float32")

        X = torch.from_numpy(X).to(self.device, non_blocking=True)
        y = torch.from_numpy(y).permute(0,3,1,2).to(self.device, non_blocking=True)
        return X, y

# Construct DataLoader
@dataclass
class LoaderCfg:     # Default Data Pipeline Input Configuration
    zarr_path   : str | os.PathLike = r"D:\FYP\Data\ERA5_CN_2022_2024_full.zarr"
    win_in      : int = 12
    win_out     : int = 3
    train_end   : str = "2023-06-30"
    val_end     : str = "2023-12-31"
    batch       : int = 8
    num_workers : int = 0
    device      : str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16    : bool = False
    scaler_cache: str = "./scaler_train.npz"

def build_loaders(cfg: LoaderCfg):
    common = dict(
        zarr_path    = cfg.zarr_path,
        win_in       = cfg.win_in,
        win_out      = cfg.win_out,
        train_end    = cfg.train_end,
        val_end      = cfg.val_end,
        device       = cfg.device,
        use_fp16     = cfg.use_fp16,
        scaler_cache = cfg.scaler_cache,
    )
    tr = ERA5Lazy(split="train", **common)   # Create the training dataset
    vl = ERA5Lazy(split="val",   **common)   # Create the validation dataset
    ts = ERA5Lazy(split="test",  **common)   # Create the testing dataset

    kwargs = dict(batch_size=cfg.batch,
                  num_workers=cfg.num_workers,
                  pin_memory=False)
    return (
        DataLoader(tr, shuffle=True,  **kwargs),   # Create the training dataloader
        DataLoader(vl, shuffle=False, **kwargs),   # Create the validation dataloader
        DataLoader(ts, shuffle=False, **kwargs),   # Create the testing dataloader
    )

# Quick sanity check
if __name__ == "__main__":
    cfg = LoaderCfg(batch=2, use_fp16=True) # Configuration Parameters
    t0  = time.perf_counter()                  # Record strat time
    train_dl, _, _ = build_loaders(cfg)        # Construct dataloader
    print(f"💡 build_loaders finished in {time.perf_counter()-t0:.2f}s")   # Print Construction time
    xb, yb = next(iter(train_dl))              # Retrieve a batch of data
    print("X:", xb.shape, xb.dtype, "| y:", yb.shape, yb.dtype,
          "| device:", xb.device)              # Print data information
