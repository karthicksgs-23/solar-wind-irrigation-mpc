"""Run notebook 02 sections 1, 3, 5 and 6 locally (no Colab, no download, no pvlib, no plots) with the cell code taken
from the built notebook. Outputs go to outputs/nb02_local_test/."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / "notebooks" / "02_energy_resource_sizing.ipynb").read_text(encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
OUT = ROOT / "outputs" / "nb02_local_test"; OUT.mkdir(parents=True, exist_ok=True)
PROC = OUT / "processed"; PROC.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
g = {"__name__": "nb02", "display": print}
setup = (f"import os, sys, json, gzip, time, platform, shutil\nimport numpy as np, pandas as pd\n"
         f"BASE=r'{ROOT}'; RAWP=r'{ROOT / 'data' / 'raw' / 'nasa_power'}'; PROC=r'{PROC}'; OUT=r'{OUT}'; SRC=r'{ROOT / 'src'}'\n")
exec(setup, g)
for i, src in enumerate(code):
    first = src.splitlines()[0] if src else ""
    if first.startswith(("!pip", "%%writefile")) or "plt.subplots" in src or "import pvlib" in src or "files.download" in src \
            or "HOURLY = " in src:
        print(f"-- skip cell {i}: {first[:60]}")
        continue
    print(f"-- run cell {i}: {first[:60]}")
    exec(compile(src, f"nb02_cell{i}", "exec"), g)
print(g["S"].round(3).to_string()); print(g["D"].round(3).to_string()); print(g["SENS"].round(1).to_string())
