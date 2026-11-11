"""Check whether the head_30m sensitivity picks (wind at Bushland/Jodhpur) are an artefact of the reduced grid, whose sizes
scale with pump power (battery <= 2x pump_kw) while the farm load does not. Re-sizes head 30 m with a wider battery/PV grid."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[1]
nb = json.loads((ROOT / "notebooks" / "02_energy_resource_sizing.ipynb").read_text(encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
OUT = ROOT / "outputs" / "nb02_local_test"; PROC = OUT / "processed"
sys.path.insert(0, str(ROOT / "src"))
g = {"__name__": "nb02", "display": print}
exec(f"import os, sys, json, gzip, time, platform, shutil\nimport numpy as np, pandas as pd\n"
     f"BASE=r'{ROOT}'; RAWP=r'{ROOT / 'data' / 'raw' / 'nasa_power'}'; PROC=r'{PROC}'; OUT=r'{OUT}'; SRC=r'{ROOT / 'src'}'\n", g)
for src in code:
    if src.startswith("GRID = en.GRID_DEFAULT"):
        break
    first = src.splitlines()[0] if src else ""
    if first.startswith(("!pip", "%%writefile")) or "plt.subplots" in src or "import pvlib" in src or "HOURLY = " in src:
        continue
    exec(compile(src, "nb02", "exec"), g)

en, R, TRAIN, TEST, PUMP_HOURS = g["en"], g["R"], g["TRAIN"], g["TEST"], g["PUMP_HOURS"]
SENS_GRID = {'pv': [1, 1.5, 2, 3], 'wind': [0, 0.5, 1], 'batt': [0, 1, 2], 'tank': [0.03, 0.0625, 0.125, 0.25]}
WIDE_GRID = {'pv': [1, 1.5, 2, 3, 4], 'wind': [0, 0.5, 1], 'batt': [0, 1, 2, 3, 4], 'tank': [0.03, 0.0625, 0.125, 0.25]}
# optional: python check_head30.py <site> <max battery multiple>, e.g. jodhpur_in 6
SITES_RUN = ("bushland_tx", "jodhpur_in")
if len(sys.argv) > 2:
    SITES_RUN = (sys.argv[1],); WIDE_GRID['batt'] = list(range(int(sys.argv[2]) + 1))
TAG = f"_{sys.argv[1]}_b{sys.argv[2]}" if len(sys.argv) > 2 else ""
rows = []
for sid in SITES_RUN:
    r = R[sid]; head = 30.0
    peak = r['dem'].loc[TRAIN].max(); pump_kw = peak / PUMP_HOURS * en.RHO_W * en.G * head / 0.55 / 3.6e6
    print(f"{sid}: pump_kw {pump_kw:.1f}, farm load mean {r['load'].mean():.2f} kW, peak {r['load'].max():.2f} kW")
    for name, grid in (("wide", WIDE_GRID),):  # the SENS_GRID result is already in sensitivity.csv
        allc, pk = en.size_system(r['pv'], r['wind'], r['dem'], head, TRAIN, TEST, pump_kw, peak, load_kw=r['load'], grid=grid,
                                  c=en.site_costs(sid))
        for kind in ("least_cost", "pv_only", "hybrid"):
            b = pk.get(kind)
            if b:
                rows.append(dict(site=sid, grid=name, design=kind, pv_kwp=b['pv_kwp'], wind_kw=b['wind_kwr'], batt_kwh=b['batt_kwh'],
                                 tank_m3=b['tank_m3'], npc_usd=b['npc_usd'], diesel_npc_usd=b['diesel']['npc'],
                                 ratio=b['npc_usd'] / b['diesel']['npc'], elpsp_train=b['elpsp_train'], wlpsp_train=b['wlpsp_train']))
            else:
                rows.append(dict(site=sid, grid=name, design=kind))
        n_pv_feas = int(((allc.wind_kwr == 0) & (allc.wlpsp_train <= 0.05) & (allc.elpsp_train <= 0.02)).sum())
        print(f"  {name}: {len(allc)} configs, feasible PV-only configs {n_pv_feas}")
D = g["pd"].DataFrame(rows)
print(D.round(3).to_string())
D.to_csv(OUT / f"check_head30{TAG}.csv", index=False)
