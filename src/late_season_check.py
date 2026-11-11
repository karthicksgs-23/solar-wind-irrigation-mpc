"""Quantify irrigation applied after the calibrated AquaCrop maturity (125 calendar days after sowing) but inside the
crop twin's 140-day FAO-56 season. Re-runs the deterministic controllers of notebook 03 v3 locally on the test years
2020-2024 with the Colab calibration (crop_calibration_fit.json) and tuning (mpc_tuned.json); PPO controllers need
Stable-Baselines3 and are not re-run (their daily schedules are on Drive: outputs/nb03_v3/schedules/).
Check: seasonal net irrigation per controller is compared with the Colab evaluation (Table 8).
Outputs: outputs/late_season_check/{daily_irrigation.csv, late_season_summary.csv, check_vs_colab.csv}."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_nb03v3 as B
import env as E

MATURITY_D = 125                                    # AquaCrop MaturityCD (calendar days), aquacrop_yield.CAL
COLAB = ROOT / "outputs" / "nb03v3_colab"
OUTD = ROOT / "outputs" / "late_season_check"; OUTD.mkdir(parents=True, exist_ok=True)
BASE = str(OUTD / "work"); os.makedirs(BASE + "/outputs/nb02", exist_ok=True)
import shutil
shutil.copy(ROOT / "outputs/nb02_colab/sizing_picks.json", BASE + "/outputs/nb02/sizing_picks.json")   # design used on Colab

code = [c["source"] for c in B.cells if c["cell_type"] == "code"]
g = dict(__name__="nb")
exec("import os, sys, json, time, glob, copy\nimport numpy as np, pandas as pd\n"
     f"BASE = r'{BASE}'; SRC = r'{ROOT / 'src'}'; PROC = r'{ROOT / 'outputs/nb02_local_test/processed'}'\n"
     "OUT = f'{BASE}/outputs/nb03_v3'; MOD = f'{OUT}/models'\n"
     "for d in (OUT, MOD, f'{OUT}/schedules'): os.makedirs(d, exist_ok=True)\n"
     "import forecast as fc, env as E, drl\n", g)
exec(code[1], g)
t0 = time.time(); exec(code[2].split("# reproducibility check")[0], g); print(f"sites + forecasts {time.time() - t0:.0f}s")

FIT = json.load(open(COLAB / "crop_calibration_fit.json")); TUNED = json.load(open(COLAB / "mpc_tuned.json"))
SITE, LOOK, SOW, TEST, PRICE, TRAIN = g["SITE"], g["LOOK"], g["SOW"], g["TEST"], g["PRICE"], g["TRAIN"]
import copy
SITE_UNCAL = {}
for sid in SITE:
    SITE_UNCAL[sid] = copy.copy(SITE[sid]); SITE_UNCAL[sid].p, SITE_UNCAL[sid].ky = 0.55, 1.25
    SITE_UNCAL[sid].set_yield_value(PRICE, FIT[sid]["ymax_train_mean"], TRAIN)
    SITE[sid].p, SITE[sid].ky = FIT[sid]["p"], FIT[sid]["ky"]
    SITE[sid].set_yield_value(PRICE, FIT[sid]["ymax_train_mean"], TRAIN)


def controllers(sid):
    t = TUNED[sid]
    yield E.RuleController(), None, SITE[sid]
    yield E.TwoStageController(reserve=t["reserve"]), LOOK[sid], SITE[sid]
    yield E.MPCController(name="mpc_default"), LOOK[sid], SITE[sid]
    yield E.MPCController(**t), LOOK[sid], SITE[sid]
    yield E.MPCController(name="mpc_uncal", **t), LOOK[sid], SITE_UNCAL[sid]
    yield E.MPCController(perfect=True, **t), None, SITE[sid]


daily, summ = [], []
for sid in SITE:
    t0 = time.time()
    for ctrl, look, site in controllers(sid):
        for y in TEST:
            _, S, sm = E.run_year(site, y, ctrl, look)
            sow = pd.Timestamp(f"{y}-{SOW[sid]}")
            s = S[["irr_mm"]].copy(); s["dap"] = (pd.to_datetime(s.index) - sow).days
            s = s.assign(site=sid, controller=ctrl.name, year=y); daily.append(s.reset_index())
            late = float(s.loc[s.dap >= MATURITY_D, "irr_mm"].sum())
            summ.append(dict(site=sid, controller=ctrl.name, year=y, irrigation_mm=float(s.irr_mm.sum()), late_mm=late,
                             twin_season_days=int(s.dap.max()) + 1, colab_check_irr=sm["irrigation_mm"]))
    print(f"{sid}: {time.time() - t0:.0f}s", flush=True)

D = pd.concat(daily, ignore_index=True); D.to_csv(OUTD / "daily_irrigation.csv", index=False)
T = pd.DataFrame(summ); T["late_share_pct"] = 100 * T.late_mm / T.irrigation_mm.where(T.irrigation_mm > 0)
T.to_csv(OUTD / "late_season_years.csv", index=False)
M = T.groupby(["site", "controller"])[["irrigation_mm", "late_mm"]].mean()
M["late_share_pct"] = 100 * M.late_mm / M.irrigation_mm; M.round(2).to_csv(OUTD / "late_season_summary.csv")

# check against the Colab evaluation (net irrigation, test years)
ev = pd.read_csv(COLAB / "evaluation_years.csv")
ev = ev[(ev.split == "test")].groupby(["site", "controller"]).irrigation_mm.mean().rename("colab_irrigation_mm")
C = M.join(ev, how="left"); C["diff_mm"] = C.irrigation_mm - C.colab_irrigation_mm
C.round(2).to_csv(OUTD / "check_vs_colab.csv")
pd.set_option("display.width", 200)
print(C.round(1).to_string())
print("DONE", time.strftime("%Y-%m-%d %H:%M:%S"))
