import xarray as xr
import numpy as np
import time, os, pathlib

NC_IN   = r"D:\FYP\Data\ERA5_CN_2022_2024_full.nc"   #NetCDF File
ZARR_OUT = r"D:\FYP\Data\ERA5_CN_2022_2024_full.zarr"      # ← Target Zarr file address
SCALER   = r"D:\FYP\Data\scaler_train.npz"      # ← mean/std scaler file
CHUNK_T  = 168                            # Time dimension chunk

FEATURE_VARS = [
    "u10","v10","u100","v100",
    "t2m","sp","slor","sdor",
    "ws10","ws100","shear","blh",
]
TRAIN_END = "2023-06-30"

#Transfoer to  Zarr
t0 = time.perf_counter()
print("➜ Converting NetCDF → Zarr …")
ds = xr.open_dataset(NC_IN, chunks={"time": CHUNK_T})
ds.to_zarr(ZARR_OUT, mode="w")
print(f"Zarr written to {ZARR_OUT}  ({time.perf_counter()-t0:.1f}s)")


import pathlib
import numpy as np

if not pathlib.Path(SCALER).exists():
    print("Calculating mean/std for scaler …")

    #Offset = win_in + win_out - 1
    offset = 24 + 6 - 1
    times = ds.time.values[offset:]

    #Training Set Time Mask
    mask_train = times < np.datetime64(TRAIN_END)
    idx = np.where(mask_train)[0] + offset

    # eature extraction: dims = var, time, lat, lon
    feat = ds[FEATURE_VARS].to_array()
    sel = feat.isel(time=idx)

    #Calculate mean and std
    mean = sel.mean(dim=("time", "latitude", "longitude")).compute().values
    std = sel.std(dim=("time", "latitude", "longitude")).compute().values
    std = np.clip(std, 1e-6, None)

    #Save Results
    np.savez(
        SCALER,
        mean=mean.astype("float32"),
        std=std.astype("float32")
    )
    print(f"scaler saved to {SCALER}")
else:
    print("scaler already exists, skip calculation")
