Extreme Wind-Speed (ws10) Forecasting over China
ConvLSTM · Informer · PatchTST · Hybrid Patch-Informer

This project build an 3-hour, 0.25° ERA5 dataset over China (73–136°E, 18–45°N) for 10 m wind-speed (ws10) forecasting, and provide training & extreme-wind evaluation scripts 
for ConvLSTM, Informer, PatchTST, and a Hybrid Patch-Informer (PatchTST features + low-rank spatial head + Informer).

Code Structure：
<pre>
repo-root/
├── layers/                     # custom layers
├── dataset.py                  # windowing & loaders
├── input.py                    # merge / normalization
├── Initial_Check.py            # sanity checks
├── ToZarr.py                   # NetCDF → Zarr
├── IC_Zarr.py                  # Zarr checks
├── CovLSTM.py                  # ConvLSTM model
├── informer.py                 # Informer model
├── PatchTST.py                 # PatchTST model
├── CoVLSTM_Train.py            # train ConvLSTM
├── informer_train.py           # train Informer
├── PatchTST_Train.py           # train PatchTST
├── Patch_Informer_Train.py     # train Hybrid (PatchTST+low-rank head+Informer)
├── eval_extreme_convlstm.py    # eval ConvLSTM
├── eval_extreme_informer.py    # eval Informer
├── eval_extreme_PatchTST.py    # eval PatchTST
├── eval_extreme_PatchTST_Informer.py  # eval Hybrid
├── Graph.py                    # plotting utils
├── function.py                 # misc utils
├── scaler_train.npz            # train-only mean/std
├── best_convlstm.pt            # weights (Git LFS)
├── best_informer.pt
├── best_patchtst_informer.pt
├── best_patchtst_series.pt
├── best_patchtst_series.pt
└── README.md
</pre>


<details> <summary><b>Files by purpose (quick scan)</b></summary>

Models

CovLSTM.py, informer.py, PatchTST.py

Training

CoVLSTM_Train.py, informer_train.py, informer_train1.py,
PatchTST_Train.py, Patch_Informer_Train.py
(The PatchTST model code and portions of the Patch-Informer code are sourced from GitHubs. https://github.com/yuqinie98/PatchTST)
Evaluation

eval_extreme_convlstm.py, eval_extreme_informer.py,
eval_extreme_PatchTST.py, eval_extreme_PatchTST_Informer.py

Data & Preprocessing

dataset.py, input.py, ToZarr.py, IC_Zarr.py, Initial_Check.py,
scaler_train.npz

Utilities & Outputs

Graph.py

Weights (LFS)

best_*.pt (tracked with Git LFS)

Data Set Sample

instant_2022.1.nc

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

The instant_2022.1.nc file serves as a sample dataset containing the aforementioned variables, with a 3-hour time interval and a time dimension spanning the first half of 2022 (January to June).
