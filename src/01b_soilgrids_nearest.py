"""Step 1b: for sites whose SoilGrids point returned no data (masked pixel, e.g. a town), query a ring of nearby points
and keep the nearest one with data. The chosen offset is recorded in data/raw/soilgrids/<site>_soilgrids_point.json.
Usage: python 01b_soilgrids_nearest.py
"""
import gzip
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SG = ROOT / "data" / "raw" / "soilgrids"
SITES = json.loads(Path(__file__).with_name("sites.json").read_text(encoding="utf-8"))
PROPS = ["sand", "silt", "clay", "bdod", "soc", "wv0033", "wv1500"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]


def query(lat, lon):
    q = [("lon", round(lon, 4)), ("lat", round(lat, 4))] + [("property", p) for p in PROPS] + \
        [("depth", d) for d in DEPTHS] + [("value", "mean")]
    with urllib.request.urlopen("https://rest.isric.org/soilgrids/v2.0/properties/query?" + urllib.parse.urlencode(q), timeout=120) as r:
        return r.read()


def has_data(payload):
    d = json.loads(payload)
    return all(L["depths"][0]["values"]["mean"] is not None for L in d["properties"]["layers"])


for sid, s in SITES.items():
    f = SG / f"{sid}_soilgrids.json.gz"
    if f.exists() and has_data(gzip.open(f).read()):
        continue
    found = None
    for ring_km in (2, 4, 6, 8, 10):
        for k in range(8):
            a = math.radians(45 * k)
            dlat = ring_km * math.cos(a) / 111.0
            dlon = ring_km * math.sin(a) / (111.0 * math.cos(math.radians(s["lat"])))
            lat, lon = s["lat"] + dlat, s["lon"] + dlon
            try:
                payload = query(lat, lon)
            except Exception as e:
                print(sid, "query failed", e); time.sleep(5); continue
            time.sleep(1)
            if has_data(payload):
                found = (lat, lon, ring_km, 45 * k, payload); break
        if found:
            break
    if not found:
        print(sid, "no SoilGrids data within 10 km"); continue
    lat, lon, km, bearing, payload = found
    with gzip.open(f, "wb") as g:
        g.write(payload)
    (SG / f"{sid}_soilgrids_point.json").write_text(json.dumps(
        {"site_lat": s["lat"], "site_lon": s["lon"], "soil_lat": round(lat, 4), "soil_lon": round(lon, 4),
         "offset_km": km, "bearing_deg": bearing, "reason": "site pixel masked in SoilGrids"}, indent=1), encoding="utf-8")
    print(f"{sid}: data at {km} km, bearing {bearing} deg ({lat:.4f}, {lon:.4f})")
