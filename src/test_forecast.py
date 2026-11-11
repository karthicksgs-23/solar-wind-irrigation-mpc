"""Local test of forecast.py on one site (raw NASA POWER data; no dependency on the nb02 processed files)."""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import energy as en
import forecast as fc
from et0 import et0_fao56

sid = sys.argv[1] if len(sys.argv) > 1 else "azraq_jo"
s = json.loads((ROOT / "src" / "sites.json").read_text(encoding="utf-8"))[sid]
h = en.load_power_hourly(ROOT / "data/raw/nasa_power" / sid, sid)
shift, _ = en.best_time_shift(h, s["lat"], s["lon"])
pv, _ = en.pv_output_per_kwp(h, s["lat"], s["lon"], shift_min=shift)
wd, _, _ = en.wind_output_per_kw(h)
ld = en.farm_load(h.T2M, s["lon"], seed=1)
d, elev = en.load_power_daily(ROOT / "data/raw/nasa_power" / sid / f"{sid}_daily_2001_2024.json.gz")
et0 = pd.Series(et0_fao56(d.index, d.T2M_MAX, d.T2M_MIN, d.RH2M, d.WS2M, d.ALLSKY_SFC_SW_DWN, s["lat"], elev, d.PS), index=d.index)
TRAIN, OTHER = list(range(2001, 2017)), list(range(2017, 2025))
pd.set_option("display.width", 200)
for name, ser in (("pv", pv), ("wind", wd), ("load", ld)):
    t0 = time.time()
    f = fc.forecast_hourly(ser, s["lon"], TRAIN, OTHER)
    print(f"\n{sid} {name}: {time.time() - t0:.0f}s")
    print("  train (blocked CV):"); print(fc.skill_table(f, TRAIN).round(3).to_string(index=False))
    print("  test 2020-2024:"); print(fc.skill_table(f, list(range(2020, 2025))).round(3).to_string(index=False))
for name, ser in (("et0", et0), ("rain", d.PRECTOTCORR)):
    f = fc.forecast_daily(ser, TRAIN, OTHER)
    print(f"\n{sid} {name} test 2020-2024:"); print(fc.skill_table(f, list(range(2020, 2025))).round(3).to_string(index=False))
