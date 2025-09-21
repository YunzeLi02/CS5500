Extreme Wind-Speed (ws10) Forecasting over China
ConvLSTM · Informer · PatchTST · Hybrid Patch-Informer

This project build an 3-hour, 0.25° ERA5 dataset over China (73–136°E, 18–45°N) for 10 m wind-speed (ws10) forecasting, and provide training & extreme-wind evaluation scripts 
for ConvLSTM, Informer, PatchTST, and a Hybrid Patch-Informer (PatchTST features + low-rank spatial head + Informer).

Code Structure：

repo-root/

├── layers/                         # Reusable modules / custom layers
├── results*/                       # Training outputs (logs, figs, metrics)
├── dataset.py                      # Windowing & data loaders (series/grid)
├── input.py                        # Merge / normalization helpers
├── Initial_Check.py                # Sanity checks for dims/time/missing
├── ToZarr.py                       # Merge NetCDF → Zarr
├── IC_Zarr.py                      # Zarr integrity/index checks
├── CovLSTM.py                      # ConvLSTM model
├── informer.py                     # Informer model
├── PatchTST.py                     # PatchTST model
├── CoVLSTM_Train.py                # Train ConvLSTM
├── informer_train.py               # Train Informer (main)
├── informer_train1.py              # Train Informer (variant)
├── PatchTST_Train.py               # Train PatchTST
├── Patch_Informer_Train.py         # Train Hybrid (PatchTST + low-rank spatial head + Informer)
├── eval_extreme_convlstm.py        # Extreme-wind eval: ConvLSTM
├── eval_extreme_informer.py        # Extreme-wind eval: Informer
├── eval_extreme_PatchTST.py        # Extreme-wind eval: PatchTST
├── eval_extreme_PatchTST_Informer.py # Extreme-wind eval: Hybrid
├── Graph.py                        # Plotting utils
├── function.py                     # Misc utilities
├── scaler_train.npz                # Train-only {mean, std}
├── best_convlstm.pt                # Weights (Git LFS)
├── best_informer.pt
├── best_informer_2.pt
├── best_patchtst_grid.pt
├── best_patchtst_informer.pt
├── best_patchtst_series.pt
└── README.md

<details> <summary><b>Files by purpose (quick scan)</b></summary>

Models

CovLSTM.py, informer.py, PatchTST.py

Training

CoVLSTM_Train.py, informer_train.py, informer_train1.py,
PatchTST_Train.py, Patch_Informer_Train.py

Evaluation

eval_extreme_convlstm.py, eval_extreme_informer.py,
eval_extreme_PatchTST.py, eval_extreme_PatchTST_Informer.py

Data & Preprocessing

dataset.py, input.py, ToZarr.py, IC_Zarr.py, Initial_Check.py,
scaler_train.npz

Utilities & Outputs

Graph.py, function.py, results*/

Weights (LFS)

best_*.pt (tracked with Git LFS)

</details>

Dataset: ERA5 hourly data on single levels (CDS)
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

Dynamic variables:
10m_u_component_of_wind (u10), 10m_v_component_of_wind (v10),
100m_u_component_of_wind (u10), 100m_v_component_of_wind (v10),
2m_temperature (t2m), mean_sea_level_pressure (msl),
total_precipitation (tp), boundary_layer_height (blh)
standard_deviation_of_orography (sdor), slope_of_sub-gridscale_orography (slor)

Area: area = [North=45, West=73, South=18, East=136]

Grid: grid = [0.25, 0.25]

Time: 2022-01 → 2024-12, 3-hour 00:00, 03:00, 06:00.... 21:00

Format: netcdf
