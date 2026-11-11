"""Local test of drl.py without gymnasium: (1) the hybrid with action 0 must equal the plain MPC; (2) random-action
episodes of both DRL environments run and return finite rewards; timings for the Colab budget."""
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import drl
import env

sys.argv = [sys.argv[0], "azraq_jo", "2020"]
exec(open(ROOT / "src" / "test_env.py", encoding="utf-8").read().split("rows = []")[0])   # builds site, look


class Zero:
    def predict(self, obs, deterministic=True):
        return np.zeros(3), None


_, _, s_mpc = env.run_year(site, 2020, env.MPCController(), look)
_, _, s_hyb0 = env.run_year(site, 2020, drl.PolicyController(Zero(), "hybrid", "hybrid0"), look)
for k in ("shed_kwh", "irrigation_mm", "yield_ratio", "op_cost"):
    print(f"{k:14s} MPC {s_mpc[k]:.6f} | hybrid(a=0) {s_hyb0[k]:.6f} | equal {np.isclose(s_mpc[k], s_hyb0[k])}")

for cls in (drl.HybridEnv, drl.DRLOnlyEnv):
    e = cls(site, [2016, 2017], look, seed=0)
    obs, info = e.reset()
    rng = np.random.default_rng(0); R = 0.0; n = 0; t0 = time.time()
    done = False
    while not done:
        obs, r, done, _, m = e.step(rng.uniform(-1, 1, 3)); R += r; n += 1
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
    print(f"{cls.__name__}: year {info['year']}, {n} steps, return {R:.1f}, {time.time() - t0:.1f}s ({1000 * (time.time() - t0) / n:.1f} ms/step), "
          f"obs range [{obs.min():.2f}, {obs.max():.2f}]")
