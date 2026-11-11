"""List the Bushland, Texas maize-for-grain validation datasets and their files (no download).

Ag Data Commons items are read through the public figshare API by item id; the station weather dataset is read
from the data.gov catalogue (CKAN) because its Ag Data Commons id was not found.
Writes bushland_listing.json next to this script.  Usage: python list_bushland.py
"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
ITEMS = {24855957: "water balance (ET, irrigation, rain, drainage)", 24855909: "growth and yield",
         24855873: "agronomic calendars", 24855858: "soil water content (all crops)"}
WEATHER = "standard-quality-controlled-research-weather-data-usda-ars-bushland-texas-f4f0b"


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


listing = {}
for i, what in ITEMS.items():
    m = get(f"https://api.figshare.com/v2/articles/{i}")
    size = sum(f["size"] for f in m["files"]) / 1e6
    print(f"\n{i} [{what}] {m['title'][:90]} | {len(m['files'])} files | {size:.1f} MB | {(m.get('license') or {}).get('name')} | {m.get('doi')}")
    for f in m["files"]:
        print(f"      {f['name']}  {f['size'] / 1e6:.2f} MB")
    listing[str(i)] = {"what": what, "title": m["title"], "doi": m.get("doi"), "license": (m.get("license") or {}).get("name"),
                       "files": [{k: f[k] for k in ("name", "size", "download_url")} for f in m["files"]]}

try:
    w = get(f"https://catalog.data.gov/api/3/action/package_show?id={WEATHER}")["result"]
    print(f"\n[weather] {w['title']} | license {w.get('license_title')}")
    res = [{"name": r.get("name") or r.get("url", "").rsplit("/", 1)[-1], "url": r.get("url"), "format": r.get("format")} for r in w["resources"]]
    for r in res:
        print(f"      {r['name'][:70]} | {r['format']} | {r['url']}")
    listing["weather"] = {"title": w["title"], "license": w.get("license_title"), "resources": res}
except Exception as e:                      # catalogue entry moved; station weather is optional (NASA POWER covers the site)
    print("\n[weather] data.gov catalogue lookup failed:", e)
json.dump(listing, open(__file__.replace("list_bushland.py", "bushland_listing.json"), "w"), indent=1)
