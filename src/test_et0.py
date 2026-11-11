"""Check: FAO-56 example 18 (Brussels, 6 July: expected ET0 = 3.9 mm/d) and Bushland NASA POWER seasonal ET0."""
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import bushland_parse as bp
from et0 import et0_fao56

# FAO-56 Example 18: Tmax 21.5, Tmin 12.3, RH 63-84% (ea 1.409), u2 2.078, Rs 22.07, lat 50.80, z 100 m -> 3.9 mm/d
print("FAO-56 ex.18:", et0_fao56(["2026-07-06"], [21.5], [12.3], [100 * 1.409 / ((1.6105 + 2.564) / 2) ], [2.078], [22.07], 50.80, 100)[0].round(2), "(expected 3.9)")

ROOT = Path(__file__).resolve().parents[1]
d = json.load(gzip.open(ROOT / "data/raw/nasa_power/bushland_tx/bushland_tx_daily_2001_2024.json.gz"))["properties"]["parameter"]
w = pd.DataFrame(d); w.index = pd.to_datetime(w.index, format="%Y%m%d")
w["et0"] = et0_fao56(w.index, w.T2M_MAX, w.T2M_MIN, w.RH2M, w.WS2M, w.ALLSKY_SFC_SW_DWN, 35.186, 1170, w.PS)
BASE = ROOT / "data/raw/bushland_maize"
for y in (2013, 2016, 2018):
    s = w.loc[f"{y}-05-15":f"{y}-09-30"]
    wb = bp.daily_water_balance(BASE, y)
    rain = wb[(wb.lysimeter == "NE") & (wb.date >= f"{y}-05-15") & (wb.date <= f"{y}-09-30")].precip.sum()
    print(f"{y} 15 May-30 Sep: ET0 {s.et0.sum():.0f} mm (max day {s.et0.max():.1f}); NASA rain {s.PRECTOTCORR.sum():.0f} mm vs lysimeter rain {rain:.0f} mm")
