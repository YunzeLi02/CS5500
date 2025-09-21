Extreme Wind-Speed (ws10) Forecasting over China
ConvLSTM · Informer · PatchTST · Hybrid Patch-Informer

This project build an 3-hour, 0.25° ERA5 dataset over China (73–136°E, 18–45°N) for 10 m wind-speed (ws10) forecasting, and provide training & extreme-wind evaluation scripts 
for ConvLSTM, Informer, PatchTST, and a Hybrid Patch-Informer (PatchTST features + low-rank spatial head + Informer).

Code Structure：
├─ layers/                           # Reusable model/layer components
├─ best_convlstm.pt                  # Model weights (use Git LFS)
├─ best_informer.pt
├─ best_informer_2.pt
├─ best_patchtst_grid.pt
├─ best_patchtst_informer.pt
├─ best_patchtst_series.pt
├─ CovLSTM.py                        # ConvLSTM definition
├─ CoVLSTM_Train.py                  # ConvLSTM training script
├─ informer.py                       # Informer definition
├─ informer_train.py                 # Informer training (main)
├─ informer_train1.py                # Informer training (variant)
├─ informer_v2.py                    # Informer variant
├─ PatchTST.py                       # PatchTST definition
├─ PatchTST_Train.py                 # PatchTST training
├─ Patch_Informer_Train.py           # Hybrid: PatchTST + low-rank spatial head + Informer
├─ hybrid_patchtst_informer.pt       # Hybrid model weights (LFS)
├─ dataset.py                        # Dataset windowing & loaders
├─ input.py                          # Data merge / normalization helpers
├─ Initial_Check.py                  # Sanity checks for dims/time/missing
├─ ToZarr.py                         # Merge NetCDF → Zarr
├─ IC_Zarr.py                        # Zarr integrity/index checks
├─ scaler_train.npz                  # Train-only mean/std
├─ eval_extreme_convlstm.py          # Extreme-wind evaluation (ConvLSTM)
├─ eval_extreme_informer.py          # Extreme-wind evaluation (Informer)
├─ eval_extreme_PatchTST.py          # Extreme-wind evaluation (PatchTST)
├─ eval_extreme_PatchTST_Informer.py # Extreme-wind evaluation (Hybrid)
└─ Graph.py                          # Plotting utils

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
