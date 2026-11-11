"""Local test: farm load at the four sites and energy.size_system with the farm load at one site (reduced grid)."""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import energy as en
from et0 import et0_fao56

SITES = json.loads((ROOT / "src" / "sites.json").read_text(encoding="utf-8"))
for sid, s in SITES.items():
    h = en.load_power_hourly(ROOT / "data/raw/nasa_power" / sid, sid)
    ld = en.farm_load(h.T2M, s["lon"], seed=1)
    m = ld.groupby(ld.index.month).mean() * 24
    night = ld[((ld.index.hour + round(s["lon"] / 15)) % 24).isin(list(range(19, 24)) + list(range(0, 7)))].sum() / ld.sum()
    print(f"{sid}: farm load {ld.groupby(ld.index.year).sum().mean():.0f} kWh/yr, peak {ld.max():.2f} kW, "
          f"kWh/day Jan {m.loc[1]:.1f} Jul {m.loc[7]:.1f}, night share {night:.2f}")

s = SITES["azraq_jo"]
h = en.load_power_hourly(ROOT / "data/raw/nasa_power/azraq_jo", "azraq_jo")
shift, _ = en.best_time_shift(h, s["lat"], s["lon"])
pv, _ = en.pv_output_per_kwp(h, s["lat"], s["lon"], shift_min=shift)
wd, _, _ = en.wind_output_per_kw(h)
ld = en.farm_load(h.T2M, s["lon"], seed=1)
d, elev = en.load_power_daily(ROOT / "data/raw/nasa_power/azraq_jo/azraq_jo_daily_2001_2024.json.gz")
et0 = pd.Series(et0_fao56(d.index, d.T2M_MAX, d.T2M_MIN, d.RH2M, d.WS2M, d.ALLSKY_SFC_SW_DWN, s["lat"], elev, d.PS), index=d.index)
dem = en.irrigation_demand(et0, d.PRECTOTCORR, "04-01")
tr, te = slice("2001-01-01", "2016-12-31"), slice("2020-01-01", "2024-12-31")
peak = dem.loc[tr].max(); pump_kw = peak / 8.0 * en.RHO_W * en.G * 60 / 0.55 / 3.6e6
t0 = time.time()
allc, pick = en.size_system(pv, wd, dem, 60.0, tr, te, pump_kw, peak, load_kw=ld,
                            grid={"pv": [1, 1.5, 2, 3], "wind": [0, 0.5, 1], "batt": [0, 1, 2, 4], "tank": [0.25, 0.5, 1]})
print(f"\nazraq with farm load: {len(allc)} configs in {time.time() - t0:.0f}s; feasible "
      f"{int(((allc.wlpsp_train <= 0.05) & (allc.elpsp_train <= 0.02)).sum())}")
for k, v in pick.items():
    print(f"  {k:10s} PV {v['pv_kwp']:5.1f} wind {v['wind_kwr']:5.1f} batt {v['batt_kwh']:5.1f} tank {v['tank_m3']:4.0f} | "
          f"NPC {v['npc_usd'] / 1e3:5.0f}k LCOE {v['lcoe_usd_kwh']:.2f} curtail {v['curtail_frac']:.2f} | "
          f"WLPSP {v['wlpsp_train']:.3f}/{v['wlpsp_test']:.3f} ELPSP {v['elpsp_train']:.3f}/{v['elpsp_test']:.3f} | "
          f"diesel {v['diesel']['npc'] / 1e3:.0f}k")
