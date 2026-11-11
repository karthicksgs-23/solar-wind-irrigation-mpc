"""Step 1c: download the Bushland, Texas maize validation files (CC0, USDA-ARS via Ag Data Commons / figshare).

Reads src/bushland_listing.json (written by list_bushland.py) and downloads only maize files plus the soil-water
README into data/raw/bushland_maize/<dataset>/. File sizes are checked against the listing; existing files are kept.
Usage: python 01c_download_bushland.py
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "bushland_maize"
listing = json.loads(Path(__file__).with_name("bushland_listing.json").read_text(encoding="utf-8"))
FOLDER = {"24855957": "water_balance", "24855909": "growth_yield", "24855873": "calendars", "24855858": "soil_water"}

n, total = 0, 0
for item, folder in FOLDER.items():
    d = OUT / folder
    d.mkdir(parents=True, exist_ok=True)
    for f in listing[item]["files"]:
        if "maize" not in f["name"].lower() and not f["name"].lower().startswith("readme"):
            continue
        p = d / f["name"]
        if p.exists() and p.stat().st_size == f["size"]:
            continue
        for k in range(4):
            try:
                with urllib.request.urlopen(f["download_url"], timeout=300) as r:
                    data = r.read()
                break
            except Exception as e:
                print("   retry", k + 1, f["name"], e); time.sleep(10 * (k + 1))
        else:
            raise RuntimeError(f["name"])
        assert len(data) == f["size"], f"size mismatch for {f['name']}: {len(data)} vs {f['size']}"
        p.write_bytes(data)
        n += 1; total += len(data)
        print(f"{folder}/{f['name']}  {len(data) / 1e6:.2f} MB")
(OUT / "SOURCES.json").write_text(json.dumps({FOLDER[k]: {"title": v["title"], "doi": v["doi"], "license": v["license"]}
                                              for k, v in listing.items() if k in FOLDER}, indent=1), encoding="utf-8")
print(f"downloaded {n} files, {total / 1e6:.1f} MB")
