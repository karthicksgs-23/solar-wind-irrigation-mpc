"""Run notebook 02 locally like test_nb02_local.py, but checkpoint every en.size_system call (cell 9 sizing and the
section-6 head re-sizings) to outputs/nb02_local_test/checkpoints/, so a crashed run resumes instead of starting over.
Launch detached: python -u src/run_nb02_resumable.py > outputs/nb02_sourced_costs.log"""
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / "notebooks" / "02_energy_resource_sizing.ipynb").read_text(encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
OUT = ROOT / "outputs" / "nb02_local_test"; OUT.mkdir(parents=True, exist_ok=True)
PROC = OUT / "processed"; PROC.mkdir(exist_ok=True)
CKPT = OUT / "checkpoints"; CKPT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
g = {"__name__": "nb02", "display": print}
exec(f"import os, sys, json, gzip, time, platform, shutil\nimport numpy as np, pandas as pd\n"
     f"BASE=r'{ROOT}'; RAWP=r'{ROOT / 'data' / 'raw' / 'nasa_power'}'; PROC=r'{PROC}'; OUT=r'{OUT}'; SRC=r'{ROOT / 'src'}'\n", g)


def checkpointed(fn):
    def wrapper(pv, wind, demand, head_m, train, test, pump_kw, peak_m3_day, **kw):
        key = repr((float(pv.sum()), float(wind.sum()), float(demand.sum()), head_m, train, test, round(pump_kw, 6),
                    round(float(peak_m3_day), 6), float(kw["load_kw"].sum()) if kw.get("load_kw") is not None else None,
                    sorted((k, repr(v)) for k, v in kw.items() if k != "load_kw")))
        f = CKPT / (hashlib.sha256(key.encode()).hexdigest()[:16] + ".pkl")
        if f.exists():
            print(f"   [checkpoint hit {f.name}, head {head_m} m]")
            return pickle.loads(f.read_bytes())
        t0 = time.time()
        res = fn(pv, wind, demand, head_m, train, test, pump_kw, peak_m3_day, **kw)
        f.write_bytes(pickle.dumps(res))
        print(f"   [sized head {head_m} m in {time.time() - t0:.0f}s -> {f.name}]")
        return res
    return wrapper


for i, src in enumerate(code):
    first = src.splitlines()[0] if src else ""
    if first.startswith(("!pip", "%%writefile")) or "plt.subplots" in src or "import pvlib" in src or "files.download" in src \
            or "HOURLY = " in src:
        print(f"-- skip cell {i}: {first[:60]}")
        continue
    print(f"-- run cell {i}: {first[:60]}  ({time.strftime('%H:%M:%S')})")
    exec(compile(src, f"nb02_cell{i}", "exec"), g)
    if "en" in g and not getattr(g["en"].size_system, "_ckpt", False):  # cell 4 reloads energy; patch after it
        g["en"].size_system = checkpointed(g["en"].size_system); g["en"].size_system._ckpt = True
print(g["S"].round(3).to_string()); print(g["D"].round(3).to_string()); print(g["SENS"].round(1).to_string())
print("DONE", time.strftime("%Y-%m-%d %H:%M:%S"))
