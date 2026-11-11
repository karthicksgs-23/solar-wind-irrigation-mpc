"""Run the non-SB3 cells of notebook 03 v3 locally (Azraq only, short year lists) against local copies of the Drive
inputs, with a stand-in for AquaCrop (not installable here), to catch errors and obtain the reproducibility reference.
Also checks: hybrid action 0 == tuned MPC; the AquaCrop season correction fires once and the episode reward equals
-(energy cost + AquaCrop loss) / REWARD_SCALE."""
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import build_nb03v3 as B
import drl
import env as E

fake = types.ModuleType("aquacrop_yield")         # stand-in: yield rises with seasonal irrigation, saturating
def season_yield(daily, soil, sow_date, irr_mm, **kw):
    return dict(yield_t_ha=14.0 * min(1.0, (float(irr_mm.sum()) + 150.0) / 900.0) ** 0.8, irrigation_mm=float(irr_mm.sum()), eta_mm=0.0)
fake.season_yield = season_yield

code = [c["source"] for c in B.cells if c["cell_type"] == "code"]
BASE = str(ROOT / "outputs" / "nb03v3_localtest"); os.makedirs(BASE + "/outputs/nb02", exist_ok=True)
import shutil
shutil.rmtree(BASE + "/outputs/nb03_v3", ignore_errors=True)
shutil.copy(ROOT / "outputs/nb02_colab/sizing_picks.json", BASE + "/outputs/nb02/sizing_picks.json")
from unittest.mock import MagicMock                  # no matplotlib locally: figures are only drawn in Colab
plt = MagicMock(); plt.subplots.side_effect = lambda n, m, **kw: (MagicMock(), [MagicMock() for _ in range(n * m)])
pd.DataFrame.plot = property(lambda self: MagicMock())
g = dict(__name__="nb", aq=fake, plt=plt)
exec("import os, sys, json, time, glob, copy\nimport numpy as np, pandas as pd\n"
     f"BASE = r'{BASE}'; SRC = r'{ROOT / 'src'}'; PROC = r'{ROOT / 'outputs/nb02_local_test/processed'}'\n"
     "OUT = f'{BASE}/outputs/nb03_v3'; MOD = f'{OUT}/models'\n"
     "for d in (OUT, MOD, f'{OUT}/schedules'): os.makedirs(d, exist_ok=True)\n"
     "import forecast as fc, env as E, drl\n", g)
exec(code[1], g)
g["LON"] = {"azraq_jo": g["LON"]["azraq_jo"]}; g["TUNE_YEARS"] = [2016]; g["CAL_YEARS"] = [2015, 2016]
t0 = time.time(); exec(code[2], g); print("sites+forecasts", f"{time.time() - t0:.0f}s")
t0 = time.time(); exec(code[3], g); print("calibration", f"{time.time() - t0:.0f}s", g["FIT"])
t0 = time.time(); exec(code[4], g); print("tuning", f"{time.time() - t0:.0f}s")

# 0) resume: re-running the stage cells loads the per-site parts and gives identical results
first = dict(FIT=g["FIT"], TUNED=g["TUNED"], SK=g["SK"].copy())
exec(code[1], g); g["LON"] = {"azraq_jo": g["LON"]["azraq_jo"]}; g["TUNE_YEARS"] = [2016]; g["CAL_YEARS"] = [2015, 2016]
t0 = time.time()
for i in (2, 3, 4): exec(code[i], g)
print("resume pass", f"{time.time() - t0:.0f}s")
assert g["FIT"] == first["FIT"] and g["TUNED"] == first["TUNED"]
pd.testing.assert_frame_equal(g["SK"].reset_index(drop=True), first["SK"].reset_index(drop=True))
log_txt = open(g["OUT"] + "/progress.log").read()
assert "resuming" in log_txt and "tuning azraq_jo: loaded" in log_txt and "calibration runs azraq_jo: loaded" in log_txt
# changed settings must refuse to resume
g2 = dict(g); src_bad = code[1].replace("PRICE = 200.0", "PRICE = 250.0"); assert src_bad != code[1]
try:
    exec(src_bad, g2); raise SystemExit("config check did not fire")
except AssertionError as ex:
    print("config check OK:", str(ex)[-60:])

site, look, tuned = g["SITE"]["azraq_jo"], g["LOOK"]["azraq_jo"], g["TUNED"]["azraq_jo"]
# 1) hybrid with a = 0 reproduces the tuned MPC
_, _, s_mpc = E.run_year(site, 2020, E.MPCController(**tuned), look)
class Zero:
    def predict(self, obs, deterministic=True):
        return np.zeros(3), None
_, _, s_h0 = E.run_year(site, 2020, drl.PolicyController(Zero(), "hybrid", "h0", mpc_kwargs=tuned), look)
print("tuned MPC op_cost", round(s_mpc["op_cost"], 3), "| hybrid a=0", round(s_h0["op_cost"], 3))
assert abs(s_mpc["op_cost"] - s_h0["op_cost"]) < 1e-6
# 2) AquaCrop season correction in both training envs
fn = lambda y, irr: fake.season_yield(None, None, None, irr)["yield_t_ha"]
for Env, kw in ((drl.HybridEnv, dict(mpc_kwargs=tuned)), (drl.DRLOnlyEnv, {})):
    e = Env(site, [2015], look, aq=dict(fn=fn, ymax=g["YMAX"]["azraq_jo"]), **kw)
    e.reset(); R = 0.0; n_end = 0; rng = np.random.default_rng(0); done = False
    while not done:
        _, r, done, _, m = e.step(rng.uniform(-1, 1, 3)); R += r; n_end += "aq_loss_usd" in m
        if "aq_loss_usd" in m: aq_loss = m["aq_loss_usd"]
    expect = -(e.plant.energy_tot + aq_loss) / drl.REWARD_SCALE
    print(Env.__name__, "season-end corrections:", n_end, "| episode reward", round(R, 4), "expected", round(expect, 4))
    assert n_end == 1 and abs(R - expect) < 1e-6

# 3) evaluation without agents (one test year) and the AquaCrop/total-cost and comparison cells
src = code[6].replace("    for kind in ('drl', 'hybrid'):\n        for seed in SEEDS:\n            yield drl.PolicyController(AGENTS[(kind, sid, seed)], kind, f'{kind}_s{seed}', mpc_kwargs=t), LOOK[sid], SITE[sid]\n", "")
assert src != code[6]
g["VAL"] = []; g["TEST"] = [2020]
t0 = time.time(); exec(src, g); print("evaluation", f"{time.time() - t0:.0f}s")
exec(code[7], g); exec(code[8], g); exec(code[9], g)
pd.set_option("display.width", 200)
print(g["TE"][["controller", "total_cost", "energy_cost", "aq_loss_usd", "twin_loss_usd", "shed_kwh", "irrigation_mm", "yield_ratio"]].round(1).to_string(index=False))
print("comparisons:", len(g["CMP"]), "rows")
