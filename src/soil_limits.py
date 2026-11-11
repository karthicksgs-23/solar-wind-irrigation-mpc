"""Root-zone water limits (0-100 cm, thickness-weighted) from the SoilGrids files in data/raw/soilgrids.
Usage: python soil_limits.py  -> prints a JSON dict site -> {theta_fc, theta_wp} (water content at 33 and 1500 kPa)."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def soil_from_soilgrids(sid):
    sg = json.load(gzip.open(ROOT / "data/raw/soilgrids" / f"{sid}_soilgrids.json.gz"))
    L = {l["name"]: l for l in sg["properties"]["layers"]}
    w = np.array([5, 10, 15, 30, 40])                                # 0-5, 5-15, 15-30, 30-60, 60-100 cm
    th = lambda n: float(np.dot([x["values"]["mean"] for x in L[n]["depths"][:5]], w) / w.sum() / 1000)
    return dict(theta_fc=round(th("wv0033"), 4), theta_wp=round(th("wv1500"), 4))


if __name__ == "__main__":
    sites = json.loads((ROOT / "src" / "sites.json").read_text(encoding="utf-8"))
    print(json.dumps({sid: soil_from_soilgrids(sid) for sid in sites}))
