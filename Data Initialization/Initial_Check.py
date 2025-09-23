import xarray as xr, numpy as np, pandas as pd, subprocess, shlex
import netCDF4
FILE = r"D:\FYP\Data\ERA5_CN_2022_2024_full.nc"

# Print Dimensions and Variables
ds = xr.open_dataset(FILE)
print(ds)

if 'valid_time' in ds.coords and 'time' not in ds.coords:
    ds = ds.rename({'valid_time': 'time'})
    print("valid_time to time done")

# ds.time, missing step check
exp = np.arange('2024-01-01T00','2025-01-01T00',3,dtype='datetime64[h]')
missing = np.setdiff1d(exp, ds.time.values)
print("missing step", missing.size)


# 3. NaN/Extreme value check
for v in ds.data_vars:
    print(v, float(ds[v].min()), float(ds[v].max()), ds[v].dtype)

# 4. Verify ws10
if {'u10','v10','ws10'}.issubset(ds):
    diff = np.abs(ds.ws10 - np.hypot(ds.u10, ds.v10)).max().item()
    print(f"ws10 Maximum closed-loop error: {diff:.4f} m/s")

