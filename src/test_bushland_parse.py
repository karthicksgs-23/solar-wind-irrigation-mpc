"""Local test of bushland_parse on the downloaded files (everything the Colab notebook needs except AquaCrop)."""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
import bushland_parse as bp

pd.set_option("display.width", 220)
BASE = Path(__file__).resolve().parents[1] / "data" / "raw" / "bushland_maize"
for y in (2013, 2016, 2018):
    wb = bp.daily_water_balance(BASE, y)
    for side, lys in (("East", ["NE", "SE"]), ("West", ["NW", "SW"])):
        pdt = bp.planting_date(bp.calendar(BASE, y, side))
        ys = bp.lysimeter_yield(BASE, y, side)
        ps = bp.profile_storage(BASE, y, side)
        fc, pwp = ps["FC"].dropna().iloc[0], ps["PWP"].dropna().iloc[0]
        for L in lys:
            s = wb[(wb.lysimeter == L) & (wb.date >= pdt) & (wb.date <= pd.Timestamp(f"{y}-10-31"))]
            yl = ys[ys.lysimeter == L]
            st = [c for c in ps.columns if f"{L} Lys" in str(c) and "1.5 m" in str(c)]
            first = pd.to_numeric(ps[st[0]], errors="coerce").dropna() if st else pd.Series(dtype=float)
            print(f"{y} {L} ({'drip' if side == 'East' else 'sprinkler'}): planted {pdt.date()} | ET {s.et.sum():.0f} mm, "
                  f"rain {s.precip.sum():.0f}, irrig {s.irrig.sum():.0f} | yield {yl['mean'].iloc[0] if len(yl) else float('nan'):.2f} Mg/ha "
                  f"(n={int(yl['count'].iloc[0]) if len(yl) else 0}) | FC {fc} PWP {pwp} | first storage {first.iloc[0] if len(first) else 'NA':.0f} mm "
                  f"on {ps.loc[first.index[0], 'date'].date() if len(first) else 'NA'}")
