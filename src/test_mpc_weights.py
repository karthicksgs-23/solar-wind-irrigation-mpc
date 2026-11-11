"""Sensitivity of the MPC to its soil terminal weight w_dr (and battery w_soc) on one site-year, scored with env.op_cost.
Usage: python test_mpc_weights.py [site] [year]"""
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
exec(open(ROOT / "src" / "test_env.py", encoding="utf-8").read().split("rows = []")[0])   # builds site, look

rows = []
for w_dr in (1.0, 3.0, 6.0, 10.0):
    for w_soc in (0.1, 1.0):
        t0 = time.time()
        _, _, sm = env.run_year(site, year, env.MPCController(w_dr=w_dr, w_soc=w_soc), look)
        sm.update(w_dr=w_dr, w_soc=w_soc, s=round(time.time() - t0, 1)); rows.append(sm)
_, _, sm = env.run_year(site, year, env.RuleController(), None); sm.update(w_dr="rule", w_soc=""); rows.append(sm)
pd.set_option("display.width", 220)
print(pd.DataFrame(rows)[["w_dr", "w_soc", "op_cost", "shed_kwh", "irrigation_mm", "yield_ratio", "stress_days"]].round(3).to_string(index=False))
