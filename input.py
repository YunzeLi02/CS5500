import xarray as xr, pathlib, numpy as np

#Folder containing 6 *.nc files

root = pathlib.Path(r"D:\FYP\Data")

#instant_2022.1.nc … instant_2024.2.nc
files = sorted(root.glob("instant_20*.nc"))

print("Waiting for Merging：")
for f in files: print("  ", f.name)

#Open, concatenate, sort, remove duplicates
ds = xr.open_mfdataset(files,
                       combine="by_coords",   # Auto-align coordinates
                       parallel=True,         # Dask parallel processing
                       chunks={"time": 168})  # week chunk

#Coordinate Name Standardization
if "valid_time" in ds.coords and "time" not in ds.coords:
    ds = ds.rename({"valid_time": "time"})

ds = ds.sortby("time")
_, uniq_idx = np.unique(ds["time"], return_index=True)
ds = ds.isel(time=uniq_idx)

#Select Variable + Derive ws10, ws100 & Shear
keep = ['u10','v10','u100','v100','t2m','sp','slor','sdor','blh']
ds   = ds[keep]

#ws10, ws100
ds["ws10"]  = np.hypot(ds["u10"],  ds["v10"])
ds["ws100"] = np.hypot(ds["u100"], ds["v100"])

#Shear
ds["shear"] = ds["ws100"] - ds["ws10"]

#Transfer to float32
ds = ds.astype("float32")

#Writing NetCDF-4 Compression
out = root / "ERA5_CN_2022_2024_full.nc"
comp   = {"zlib": True, "complevel": 4}
encode = {v: comp | {"dtype": "float32"} for v in ds.data_vars}

ds.to_netcdf(out,
             engine="netcdf4",
             format="NETCDF4",
             encoding=encode)

print(f"\nDone → {out}\n   大小 ≈ {out.stat().st_size/1e9:.2f} GB")
