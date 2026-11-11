"""Late-season irrigation (>= 125 days after sowing, after the calibrated AquaCrop maturity) for every controller,
from the Colab daily schedules (Drive outputs/nb03_v3/schedules, unzipped to outputs/nb03v3_colab/schedules_drive).
Learning controllers: mean over seeds 0-2, plus the seed range. Cross-check: deterministic controllers against the
local re-run (outputs/late_season_check/late_season_years.csv). Writes outputs/late_season_check/all_controllers_*.csv."""
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "nb03v3_colab" / "schedules_drive"
OUTD = ROOT / "outputs" / "late_season_check"
SOW = {"bushland_tx": "05-15", "jodhpur_in": "07-01", "souss_ma": "03-15", "azraq_jo": "04-01"}
TEST, MATURITY_D = range(2020, 2025), 125
PAT = re.compile(r"^(bushland_tx|jodhpur_in|souss_ma|azraq_jo)_(.+)_(\d{4})\.csv$")

rows = []
for f in SRC.rglob("*.csv"):
    m = PAT.match(f.name)
    sid, ctrl, y = m.group(1), m.group(2), int(m.group(3))
    if y not in TEST:
        continue
    s = pd.read_csv(f, parse_dates=["day"])
    dap = (s.day - pd.Timestamp(f"{y}-{SOW[sid]}")).dt.days
    rows.append(dict(site=sid, controller=ctrl, family=re.sub(r"_s\d$", "", ctrl), year=y, season_days=int(dap.max()) + 1,
                     irrigation_mm=float(s.irr_mm.sum()), late_mm=float(s.irr_mm[dap >= MATURITY_D].sum())))
Y = pd.DataFrame(rows)
assert len(Y) == 4 * 12 * 5, len(Y)
Y.to_csv(OUTD / "all_controllers_years.csv", index=False)

per_run = Y.groupby(["site", "family", "controller"])[["irrigation_mm", "late_mm"]].mean()
fam = per_run.groupby(["site", "family"]).agg(irrigation_mm=("irrigation_mm", "mean"), late_mm=("late_mm", "mean"),
                                              late_min=("late_mm", "min"), late_max=("late_mm", "max"))
fam["late_share_pct"] = 100 * fam.late_mm / fam.irrigation_mm
fam["before_mm"] = fam.irrigation_mm - fam.late_mm
fam.round(2).to_csv(OUTD / "all_controllers_summary.csv")

# cross-check against the local re-run (deterministic controllers)
loc = pd.read_csv(OUTD / "late_season_years.csv").groupby(["site", "controller"])[["irrigation_mm", "late_mm"]].mean()
chk = per_run.reset_index(level="family", drop=True).join(loc, rsuffix="_local", how="inner")
print("max |diff| vs local re-run (mm): irrigation", round((chk.irrigation_mm - chk.irrigation_mm_local).abs().max(), 2),
      "| late", round((chk.late_mm - chk.late_mm_local).abs().max(), 2), "| n =", len(chk))
print("season days:", sorted(Y.season_days.unique()))
pd.set_option("display.width", 200)
print(fam.round(1).to_string())
j = fam.xs("mpc", level="family")
for other in ("rule", "two_stage", "mpc_uncal", "hybrid", "drl"):
    o = fam.xs(other, level="family")
    print(f"\njoint MPC - {other}:"); print((j[["irrigation_mm", "late_mm", "before_mm"]] - o[["irrigation_mm", "late_mm", "before_mm"]]).round(1).to_string())
spread = fam.groupby("site").late_mm.agg(["min", "max"]); spread["range"] = spread["max"] - spread["min"]
print("\nlate_mm across all controller families per site:"); print(spread.round(1).to_string())
print("overall late range", round(fam.late_mm.min(), 1), "-", round(fam.late_mm.max(), 1),
      "| share", round(fam.late_share_pct.min(), 1), "-", round(fam.late_share_pct.max(), 1))
