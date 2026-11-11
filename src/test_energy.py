"""Local test of energy.py on the downloaded NASA POWER data (all four sites)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import energy as en
from et0 import et0_fao56

SITES = json.loads((ROOT / "src" / "sites.json").read_text(encoding="utf-8"))
SOW = {"bushland_tx": "05-15", "jodhpur_in": "07-01", "souss_ma": "03-15", "azraq_jo": "04-01"}
ELEV = {"bushland_tx": 1170, "jodhpur_in": 224, "souss_ma": 230, "azraq_jo": 520}
for sid, s in SITES.items():
    t0 = time.time()
    h = en.load_power_hourly(ROOT / "data/raw/nasa_power" / sid, sid)
    shift, corr = en.best_time_shift(h, s["lat"], s["lon"])
    pv, poa = en.pv_output_per_kwp(h, s["lat"], s["lon"], shift_min=shift)
    wd, v_hub, alpha = en.wind_output_per_kw(h)
    d = en.load_power_daily(ROOT / "data/raw/nasa_power" / sid / f"{sid}_daily_2001_2024.json.gz")
    et0 = pd.Series(et0_fao56(d.index, d.T2M_MAX, d.T2M_MIN, d.RH2M, d.WS2M, d.ALLSKY_SFC_SW_DWN, s["lat"], ELEV[sid], d.PS), index=d.index)
    dem = en.irrigation_demand(et0, d.PRECTOTCORR, SOW[sid])
    yrs = pv.groupby(pv.index.year).sum()
    print(f"\n{sid}: hours {len(h)}, NaN {int(h.isna().sum().sum())} | best timestamp shift {shift} min {dict((k, round(v, 3)) for k, v in corr.items())}")
    print(f"  PV yield {yrs.mean():.0f} kWh/kWp/yr (range {yrs.min():.0f}-{yrs.max():.0f}), POA {poa.groupby(poa.index.year).sum().mean() / 1000:.0f} kWh/m2/yr")
    print(f"  wind CF {wd.mean():.3f} at 30 m (mean hub speed {v_hub.mean():.2f} m/s, mean shear {alpha.mean():.2f})")
    print(f"  ET0 {et0.groupby(et0.index.year).sum().mean():.0f} mm/yr; seasonal gross irrigation (5 ha) {dem.groupby(dem.index.year).sum().mean():.0f} m3/yr, peak {dem.max():.0f} m3/d")
    pk = dem.max(); pump_kw = pk / 8.0 * en.RHO_W * en.G * 60 / 0.55 / 3.6e6
    tr = slice("2001-01-01", "2016-12-31")
    t1 = time.time()
    hh, summ = en.simulate(pv.loc[tr], wd.loc[tr], dem.loc[tr], 60.0, pv_kwp=2 * pump_kw, wind_kwr=pump_kw, batt_kwh=pump_kw * 2,
                           tank_m3=pk * 1.5, pump_kw=pump_kw)
    print(f"  test system (pump {pump_kw:.1f} kW, PV {2 * pump_kw:.1f} kWp, wind {pump_kw:.1f} kW, batt {2 * pump_kw:.1f} kWh, tank {1.5 * pk:.0f} m3): "
          f"WLPSP {summ['wlpsp']:.3f}, {summ['kwh_per_m3']:.3f} kWh/m3, curtail {summ['curtail_kwh'] / summ['gen_kwh']:.2f} | sim {time.time() - t1:.1f}s, total {time.time() - t0:.1f}s")
    print(f"  NPC {en.npc(2 * pump_kw, pump_kw, 2 * pump_kw, 1.5 * pk, pump_kw):,.0f} USD")
