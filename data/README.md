# Data

Raw data are not stored in this repository. They are public and are downloaded by the scripts in `src/`:

| Folder (created) | Source | Script |
|---|---|---|
| `raw/nasa_power/<site>/` | NASA POWER API: hourly 2001–2024 (community RE, UTC) and daily 2001–2024 (community AG), stored as unchanged gzipped JSON | `src/01_download.py` |
| `raw/soilgrids/` | SoilGrids 2.0 (sand, silt, clay, bulk density, organic carbon, water content at 33 and 1500 kPa, six depths) | `src/01_download.py`, `src/01b_soilgrids_nearest.py` |
| `raw/bushland_maize/` | USDA-ARS Bushland maize datasets (water balance, growth and yield, soil water, calendars; the 2013, 2016 and 2018 seasons are used), Ag Data Commons, CC0 | `src/01c_download_bushland.py` (file list in `src/bushland_listing.json`) |
| `processed/` | Hourly and daily site series written by notebook 02 / `src/run_nb02_resumable.py` | — |

NASA POWER timestamps mark the start of each hour (UTC). The Souss SoilGrids point was moved about 2 km north to the nearest unmasked pixel; this is recorded in `raw/soilgrids/souss_ma_soilgrids_point.json`.
