"""Step 1: download public weather (NASA POWER) and soil (SoilGrids) data for every site in sites.json.

Raw API responses are stored unchanged (gzipped JSON) under data/raw/ for provenance; nothing is processed here.
NASA POWER: hourly (community RE, UTC, 2001-2024, one request per site-year) and daily (community AG, 2001-2024).
SoilGrids v2.0: mean sand, silt, clay, bulk density, organic carbon and water content at 33 and 1500 kPa, six depths.
Re-running skips files that already exist.  Usage: python 01_download.py
"""
import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SITES = json.loads((Path(__file__).with_name("sites.json")).read_text(encoding="utf-8"))
YEARS = range(2001, 2025)

HOURLY = ["ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF", "T2M", "RH2M", "PRECTOTCORR", "WS10M", "WS50M", "PS"]
DAILY = ["ALLSKY_SFC_SW_DWN", "T2M_MAX", "T2M_MIN", "T2M", "RH2M", "WS2M", "PRECTOTCORR", "PS"]
SOIL_PROPS = ["sand", "silt", "clay", "bdod", "soc", "wv0033", "wv1500"]
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]


def fetch(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception as e:  # network errors and API throttling
            wait = 10 * (k + 1)
            print(f"   retry {k + 1} in {wait}s: {e}")
            time.sleep(wait)
    raise RuntimeError(f"failed: {url}")


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.loads(payload)                       # refuse to store anything that is not valid JSON
    with gzip.open(path, "wb") as f:
        f.write(payload)


def power(kind, site, s, start, end, params, community):
    q = dict(parameters=",".join(params), community=community, longitude=s["lon"], latitude=s["lat"],
             start=start, end=end, format="JSON")
    if kind == "hourly":
        q["time-standard"] = "UTC"
    return f"https://power.larc.nasa.gov/api/temporal/{kind}/point?" + urllib.parse.urlencode(q)


for sid, s in SITES.items():
    out = RAW / "nasa_power" / sid
    f = out / f"{sid}_daily_2001_2024.json.gz"
    if not f.exists():
        print(sid, "daily"); save(f, fetch(power("daily", sid, s, "20010101", "20241231", DAILY, "AG")))
    for y in YEARS:
        f = out / f"{sid}_hourly_{y}.json.gz"
        if f.exists():
            continue
        print(sid, "hourly", y)
        save(f, fetch(power("hourly", sid, s, f"{y}0101", f"{y}1231", HOURLY, "RE")))
        time.sleep(1)
    f = RAW / "soilgrids" / f"{sid}_soilgrids.json.gz"
    if not f.exists():
        q = [("lon", s["lon"]), ("lat", s["lat"])] + [("property", p) for p in SOIL_PROPS] + \
            [("depth", d) for d in DEPTHS] + [("value", "mean")]
        print(sid, "soilgrids")
        try:
            save(f, fetch("https://rest.isric.org/soilgrids/v2.0/properties/query?" + urllib.parse.urlencode(q), tries=3))
        except RuntimeError as e:
            print("   SoilGrids unavailable for", sid, "-", e)
print("done")
