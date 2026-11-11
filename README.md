# Crop-calibrated, yield-priced MPC for off-grid solar irrigation

Code and processed results for the article

> *Scheduling energy and drip irrigation on off-grid solar farms in water-scarce regions using crop-calibrated and yield-priced model predictive control* (submitted to *Applied Energy*).

The study sizes an off-grid PV–wind–battery–tank system for a 5 ha drip-irrigated maize field and compares hourly controllers at four water-scarce sites: Bushland (USA), Jodhpur (India), Souss (Morocco) and Azraq (Jordan). The main controller is a joint 48 h model predictive control (MPC) linear program. It prices crop water stress with an FAO-56 crop twin that is calibrated against AquaCrop-OSPy. It is compared with rule-based, two-stage (decoupled), uncalibrated-MPC, hybrid PPO–MPC and model-free PPO controllers on held-out years (2020–2024), with yields evaluated in AquaCrop.

All input data are public and are downloaded by the scripts in `src/` (see `data/README.md`).

## Repository layout

```
src/            Python modules, data-download scripts, notebook builders, post-processing, figures, tables, tests
notebooks/      The three Google Colab notebooks used for the published runs
outputs/        Processed results used for the tables and figures of the article (small CSV/JSON files)
data/           Empty; filled by the download scripts (raw data are not redistributed)
```

### Core modules (`src/`)

| File | Content |
|---|---|
| `sites.json` | Site coordinates, sowing dates and settings |
| `et0.py` | FAO-56 Penman–Monteith reference evapotranspiration |
| `bushland_parse.py` | Parser for the USDA-ARS Bushland maize lysimeter datasets |
| `energy.py` | PV (NOAA solar position, isotropic POA, NOCT) and small-wind models, farm load, hourly energy balance, sizing grid search, NPC/LCOE/LCOW, diesel reference, sourced costs and cost scenarios |
| `soil_limits.py` | Soil-water limits from SoilGrids 2.0 |
| `forecast.py` | Day-ahead gradient-boosting forecasts (PV, wind, load, ET0, rain) with blocked cross-validation |
| `env.py` | Hourly plant (energy, tank, daily FAO-56 soil water, yield-loss value), rule, two-stage and joint MPC controllers (SciPy HiGHS LP), MPC tuning, crop-twin calibration helpers |
| `drl.py` | Gymnasium environments for the hybrid PPO–MPC and the model-free PPO agents |
| `aquacrop_yield.py` | AquaCrop-OSPy evaluation of an irrigation schedule |

### Workflow

| Step | Script / notebook | Output (`outputs/`) | Used for |
|---|---|---|---|
| 1. Download data | `src/01_download.py`, `src/01b_soilgrids_nearest.py`, `src/01c_download_bushland.py` | `data/raw/` | all |
| 2. AquaCrop calibration and validation at Bushland | `notebooks/01_bushland_crop_water_validation_v2.ipynb` (built by `src/build_nb01.py`) | `nb01_v2_drive/` | AquaCrop validation table and figure |
| 3. Resources, sizing and techno-economics | `notebooks/02_energy_resource_sizing.ipynb` (built by `src/build_nb02.py`); run locally with `src/run_nb02_resumable.py` | `nb02_local_test/` (final), `nb02_colab/` (design file used by step 4) | sizing, NPC and sensitivity tables and figure |
| 4. Forecasts, crop-twin calibration, MPC tuning, PPO training, test-year evaluation with AquaCrop | `notebooks/03_controllers_drl_mpc_v3.ipynb` (built by `src/build_nb03v3.py`) | `nb03v3_colab/` | forecast, crop-twin, controller and ablation results |
| 5. Price and outage-penalty scenarios, paired bootstrap | `src/posthoc_nb03v3.py` | `nb03v3_colab/posthoc/` | cost comparison tables and figures |
| 6. Late-season irrigation check | `src/late_season_check.py`, `src/late_season_drive.py` | `late_season_check/` | crop-model horizon check |
| 7. 30 m head sizing check | `src/check_head30.py` | `nb02_local_test/check_head30*.csv` | sensitivity |
| 8. Figures and tables | `src/make_figures.py`, `src/make_tables.py`, `src/make_fig_system.py` | `manuscript/figures/`, `manuscript/parts/tables/`; `make_fig_system.py` writes `system_figure_option2_blockdiagram.png` (Fig. 2) to the repository root | article |

`make_figures.py` writes `fig1`–`fig6`. In the article these are Fig. 1 and Figs. 3–7, because Fig. 2 (system configuration) comes from `make_fig_system.py`.

## Running

Python 3.11+ is required (the published runs used Python 3.13 on Google Colab).

```bash
pip install -r requirements.txt
python src/01_download.py            # NASA POWER hourly/daily 2001–2024 and SoilGrids for the four sites
python src/01b_soilgrids_nearest.py  # nearest valid SoilGrids pixel where the site pixel is masked
python src/01c_download_bushland.py  # USDA-ARS Bushland maize datasets (Ag Data Commons, CC0)
```

**Local steps.** Run the energy sizing with `python src/run_nb02_resumable.py` (results are checkpointed and the run can be resumed). Then run `python src/posthoc_nb03v3.py`, `python src/make_tables.py` and `python src/make_figures.py`. The figure and table scripts read only the files in `outputs/`, so they reproduce the article's numbers without re-running the simulations.

**Colab steps (notebooks 01 and 03).** The notebooks mount Google Drive and use `BASE = /content/drive/MyDrive/solar_wind_irrigation` (change it in the first code cell if needed). Copy these to Drive first:

- `data/raw/` from step 1 and `data/processed/` written by step 3, keeping the same folder names under `BASE`;
- `outputs/nb02_colab/sizing_picks.json` to `BASE/outputs/nb02/sizing_picks.json`;
- for notebook 01, `src/` to `BASE/src`; for notebook 03, `src/forecast.py`, `src/env.py`, `src/drl.py` and `src/aquacrop_yield.py` to `BASE/src_v3`.

Notebook 03 checks the SHA-256 of these four modules before running. It runs on CPU (about 3–4 h), saves checkpoints to Drive, and can be resumed after a disconnect. The published run used seeds 0–2, 60,000 PPO steps for the hybrid agent and 150,000 for the model-free agent.

**Tests.** The `src/test_*.py` scripts are standalone checks run with `python src/test_<name>.py` (for example, `test_et0.py` reproduces FAO-56 Example 18). Most of them need the downloaded data.

## Reproducibility notes

- Package versions of the published runs are recorded in `outputs/*/run_info.json` (Python 3.13.16, pandas 2.2.3, NumPy 2.1.3, SciPy 1.16.3, scikit-learn 1.6.1, stable-baselines3 2.9.2, Gymnasium 1.4.0, aquacrop 3.1.0).
- The MPC linear program has tie-breaking terms, so it is deterministic. The reference operating cost of the perfect-forecast MPC at Azraq in 2020 is 4452.1 USD locally and 4452.3 USD on Colab.
- Cost inputs and their sources are listed in `energy.COSTS`, `energy.COST_SOURCES` and `energy.COST_SCENARIOS`.

## Data sources

- NASA POWER (hourly and daily), https://power.larc.nasa.gov
- SoilGrids 2.0, ISRIC, https://soilgrids.org
- USDA-ARS Conservation and Production Research Laboratory, Bushland, Texas: maize lysimeter, growth-and-yield, soil-water and water-balance datasets, Ag Data Commons (CC0), https://data.nal.usda.gov

## License

Code: MIT (see `LICENSE`). The downloaded datasets remain under their providers' terms.

## Citation

Please cite the article when it is published (reference to be added).
