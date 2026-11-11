"""Local test of env.py: one site, one test year, rule / two-stage / MPC (forecast) / MPC (perfect)."""
import gzip
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
import env
import forecast as fc
from et0 import et0_fao56

sid = sys.argv[1] if len(sys.argv) > 1 else "azraq_jo"
year = int(sys.argv[2]) if len(sys.argv) > 2 else 2020
s = json.loads((ROOT / "src" / "sites.json").read_text(encoding="utf-8"))[sid]
SOW = {"bushland_tx": "05-15", "jodhpur_in": "07-01", "souss_ma": "03-15", "azraq_jo": "04-01"}
h = en.load_power_hourly(ROOT / "data/raw/nasa_power" / sid, sid)
shift, _ = en.best_time_shift(h, s["lat"], s["lon"])
pv, _ = en.pv_output_per_kwp(h, s["lat"], s["lon"], shift_min=shift)
wd, _, _ = en.wind_output_per_kw(h)
ld = en.farm_load(h.T2M, s["lon"], seed=1)
d, elev = en.load_power_daily(ROOT / "data/raw/nasa_power" / sid / f"{sid}_daily_2001_2024.json.gz")
et0 = pd.Series(et0_fao56(d.index, d.T2M_MAX, d.T2M_MIN, d.RH2M, d.WS2M, d.ALLSKY_SFC_SW_DWN, s["lat"], elev, d.PS), index=d.index)

sg = json.load(gzip.open(ROOT / "data/raw/soilgrids" / f"{sid}_soilgrids.json.gz"))
L = {l["name"]: l for l in sg["properties"]["layers"]}
w = np.array([5, 10, 15, 30, 40])                                    # 0-100 cm thickness weights
th = lambda n: float(np.dot([x["values"]["mean"] for x in L[n]["depths"][:5]], w) / w.sum() / 1000)
soil = dict(theta_fc=th("wv0033"), theta_wp=th("wv1500"))
print("soil", {k: round(v, 3) for k, v in soil.items()}, "TAW per m", round(1000 * (soil["theta_fc"] - soil["theta_wp"])))

dem = en.irrigation_demand(et0, d.PRECTOTCORR, SOW[sid])
peak = dem.loc["2001":"2016"].max(); pump_kw = peak / 8.0 * en.RHO_W * en.G * 60 / 0.55 / 3.6e6
design = dict(pv_kwp=1.5 * pump_kw, wind_kwr=0.0, batt_kwh=1.0 * pump_kw, tank_m3=0.125 * peak, pump_kw=pump_kw)
print("design", {k: round(v, 1) for k, v in design.items()})
hourly = pd.DataFrame({"pv_kw_per_kwp": pv, "wind_kw_per_kw": wd, "farm_load_kw": ld})
daily = pd.DataFrame({"et0_mm": et0, "rain_mm": d.PRECTOTCORR})
site = env.Site(sid, s["lon"], hourly, daily, design, soil, SOW[sid])

# forecasts -> lookup dicts {(local day, lead): array}
TR, OT = list(range(2001, 2017)), list(range(2017, 2025))
look = {"hourly": {}, "daily": {}}
for var, ser in (("pv", pv), ("wind", wd), ("load", ld)):
    f = fc.forecast_hourly(ser, s["lon"], TR, OT)
    f = f[f.day.dt.year.isin([year - 1, year, year + 1])]
    look["hourly"][var] = {(dd, ll): g.sort_values("hr").pred.values for (dd, ll), g in f.groupby(["day", "lead"]) if len(g) == 24}
for var, ser in (("et0", et0), ("rain", d.PRECTOTCORR)):
    f = fc.forecast_daily(ser, TR, OT)
    look["daily"][var] = {(r.day, r.lead): r.pred for r in f.itertuples() if r.day.year in (year - 1, year, year + 1)}

rows = []
for ctrl, fcl in ((env.RuleController(), None), (env.TwoStageController(), look), (env.MPCController(), look),
                  (env.MPCController(perfect=True), None)):
    t0 = time.time()
    H, S, sm = env.run_year(site, year, ctrl, fcl)
    sm.update(controller=ctrl.name, seconds=round(time.time() - t0, 1)); rows.append(sm)
pd.set_option("display.width", 220)
out = pd.DataFrame(rows).set_index("controller")
print(out[["seconds", "elpsp", "shed_kwh", "pump_kwh", "curtail_kwh", "batt_throughput_kwh", "irrigation_mm", "eta_etc", "yield_ratio",
           "stress_days"]].round(3).to_string())
