"""Parse the Bushland, Texas maize files into tidy tables (shared by the Colab notebook and local tests).

Functions take the folder data/raw/bushland_maize (layout written by 01c_download_bushland.py) and a year.
Lysimeters: NE, SE, NW, SW. Columns are found by pattern because headers vary slightly between years.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

LYS = ["NE", "SE", "NW", "SW"]


def _sheet(path, pattern):
    xl = pd.ExcelFile(path)
    names = [s for s in xl.sheet_names if re.search(pattern, s, re.I) and not s.lower().startswith("dic")]
    if not names:
        raise KeyError(f"no sheet matching {pattern!r} in {path.name}: {xl.sheet_names}")
    return pd.read_excel(xl, names[0])


def intro_text(base, year):
    """All text of the water-balance introduction sheet (states which lysimeters had drip vs sprinkler)."""
    df = _sheet(Path(base) / "water_balance" / f"{year}_Maize_water_balance.xlsx", r"introduction")
    return " ".join(str(v) for v in df.astype(str).values.ravel() if v != "nan")


def daily_water_balance(base, year):
    """Long table: date, lysimeter, et, dew, precip, irrig, drainage (mm/day). Lysimeters with no data are dropped."""
    df = _sheet(Path(base) / "water_balance" / f"{year}_Maize_water_balance.xlsx", r"daily wbal")
    df = df.replace("#N/A", np.nan)
    doy = pd.to_numeric(df[[c for c in df.columns if re.match(r"day of year", str(c), re.I)][0]], errors="coerce")
    date = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(doy - 1, unit="D")
    out = []
    for L in LYS:
        cols = {k: [c for c in df.columns if re.match(fr"{L}\s+{pat}", str(c), re.I)]
                for k, pat in [("et", r"ET\b"), ("dew", r"dew"), ("precip", r"precip"), ("irrig", r"irrig"), ("drainage", r"drain")]}
        if not cols["et"]:
            continue
        t = pd.DataFrame({"date": date, "lysimeter": L})
        for k, c in cols.items():
            t[k] = pd.to_numeric(df[c[0]], errors="coerce") if c else np.nan
        if t.et.notna().sum() > 30:
            out.append(t)
    return pd.concat(out, ignore_index=True).dropna(subset=["date"])


def profile_storage(base, year, side):
    """Soil-water storage (mm) in the top 1.5 m by lysimeter/field column, with FC, PWP and MAD in mm.
    side: 'East' or 'West'."""
    f = Path(base) / "soil_water" / f"{year}_{side}_Maize_Soil-water.xlsx"
    df = _sheet(f, r"profile wc").replace("#N/A", np.nan)
    doy = pd.to_numeric(df["DOY"], errors="coerce")
    df.insert(0, "date", pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(doy - 1, unit="D"))
    return df


def calendar(base, year, side):
    """Calendar as DATE, DOY, ACTION columns. The header row is located by its 'DATE' cell because some years have
    leading blank or title rows; the action text is the third column whatever its header."""
    f = [p for p in (Path(base) / "calendars").glob(f"{year}_{side}_Maize*Calendar*.xlsx")][0]
    xl = pd.ExcelFile(f)
    raw = pd.read_excel(xl, [s for s in xl.sheet_names if not s.lower().startswith("dic")][0], header=None)
    hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip().str.upper().eq("DATE")]
    body = raw.iloc[(hdr[0] + 1 if len(hdr) else 0):, :3].copy()
    body.columns = ["DATE", "DOY", "ACTION"]
    body["DATE"] = pd.to_datetime(body["DATE"], errors="coerce")
    return body.dropna(subset=["DATE"]).reset_index(drop=True)


def planting_date(cal):
    """First planting date in the calendar text (harvest dates are not used: AquaCrop ends the season at maturity,
    and some yield sheets carry inconsistent dates)."""
    act = cal["ACTION"].astype(str).str.strip()
    plant = cal.DATE[act.str.match(r"(?:started |finished )?plant(?:ed|ing)\b", case=False)]
    return plant.min() if len(plant) else pd.NaT


def initial_storage(prof, lys, sow):
    """Measured storage (mm, top 1.5 m) in the lysimeter closest in time to sowing, and its date."""
    c = [c for c in prof.columns if f"{lys} Lys" in str(c) and "1.5 m" in str(c)][0]
    s = pd.DataFrame({"date": prof["date"], "mm": pd.to_numeric(prof[c], errors="coerce")}).dropna()
    i = (s.date - sow).abs().idxmin()
    return float(s.loc[i, "mm"]), s.loc[i, "date"]


def lysimeter_yield(base, year, side, row_m=0.762):
    """Oven-dry grain yield (Mg/ha) of hand-harvested lysimeter samples: one row per lysimeter (mean, sd, n).
    Samples are identified by an id containing 'LYS' (e.g. NELYS1); yield is taken from the 'Oven dry grain yield in
    kg/ha' column, else from g/m^2, else from grain mass / (row length x 0.762 m row spacing)."""
    h = yield_sheets(base, year, side)["hand"]
    idc = next(c for c in h.columns if h[c].astype(str).str.contains("LYS", case=False).any())
    h = h[h[idc].astype(str).str.contains("LYS", case=False)].copy()
    h["lysimeter"] = h[idc].astype(str).str.upper().str.extract(r"^(NE|SE|NW|SW)")[0]
    col = lambda pat: [c for c in h.columns if re.search(pat, str(c), re.I)]
    kg, gm = col(r"oven.dry (grain|corn) yield in kg/ha"), col(r"oven.dry (grain|corn) yield in g/m")
    if kg:
        y = pd.to_numeric(h[kg[0]], errors="coerce") / 1000
    elif gm:
        y = pd.to_numeric(h[gm[0]], errors="coerce") / 100
    else:                                   # grain mass (g) over sampled area (m^2) -> g/m^2 -> Mg/ha (/100)
        m = pd.to_numeric(h[col(r"total oven.dry shelled (grain|kernel) mass in g")[0]], errors="coerce")
        area = col(r"(sample area|area sampled) in m")
        a = (pd.to_numeric(h[area[0]], errors="coerce") if area
             else pd.to_numeric(h[col(r"length of row")[0]], errors="coerce") * row_m)
        y = m / a / 100
    h["yield_Mg_ha"] = y
    return (h.dropna(subset=["lysimeter", "yield_Mg_ha"]).groupby("lysimeter")["yield_Mg_ha"]
            .agg(["mean", "std", "count"]).reset_index().assign(year=year))


def yield_sheets(base, year, side):
    """Return the hand-yield and combine-harvest sheets as DataFrames (structure printed by the notebook)."""
    f = [p for p in (Path(base) / "growth_yield").glob(f"{year}_{side}_Maize_Growth_and_Yield*.xlsx")][0]
    out = {}
    for key, pat in (("hand", r"hand yield"), ("combine", r"comb\.? harv")):
        try:
            df = _sheet(f, pat).replace("#N/A", np.nan)
        except KeyError:
            continue
        if sum(str(c).startswith("Unnamed") for c in df.columns) > len(df.columns) / 2:   # header is on a lower row
            r = df.index[df.iloc[:, 0].astype(str).str.strip().eq("Date")]
            if len(r):
                df.columns = [str(a) if str(a) != "nan" else str(b)
                              for a, b in zip(df.loc[r[0]].values, df.columns)]
                df = df.loc[r[0] + 1:].reset_index(drop=True)
        out[key] = df
    return out
