import xarray as xr
import numpy as np
import zarr, os, json, pprint, pathlib

ZARR_PATH = r"D:\FYP\Data\ERA5_CN_2022_2024_full.zarr"

ds = xr.open_zarr(ZARR_PATH, consolidated=True)   # consolidated=.zmetadata
print(ds)                                          #Overview

#valid_time check
if 'valid_time' in ds.coords and 'time' not in ds.coords:
    ds = ds.rename({'valid_time': 'time'})
    print("rename valid_time → time")

#Sequence Gap Check
exp = np.arange('2024-01-01T00', '2025-01-01T00', 3, dtype='datetime64[h]')
missing = np.setdiff1d(exp, ds.time.values)
print("缺步数 :", missing.size)

#Nan and Extreme value check
for v in ds.data_vars:
    da = ds[v]
    print(f"{v:6s}  min={float(da.min()):8.2f}  max={float(da.max()):8.2f}  dtype={da.dtype}")

#ws10 Closure Error
if {'u10','v10','ws10'}.issubset(ds):
    diff = (
        np.abs(ds.ws10 - np.hypot(ds.u10, ds.v10))
        .max()
        .compute()
        .item()
    )
    print(f"Max ws10 Closure Error: {diff:.4f} m/s")

# Underlying Zarr store structure
root = zarr.open_group(ZARR_PATH, mode="r")
print("\n📂 Zarr tree:")
print(root.tree())

# Print the chunk size of each variable & compressor
print("\n Chunk Information:")
for var in ds.data_vars:
    arr = root[var]
    comp = arr.compressor or "None"
    print(f"{var:6s}  chunks={arr.chunks}  compressor={comp}")

