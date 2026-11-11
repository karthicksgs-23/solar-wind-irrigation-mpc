"""Build notebooks/01_bushland_crop_water_validation.ipynb (Colab). Embeds the locally tested modules bushland_parse.py
and et0.py verbatim via %%writefile so the notebook and the local tests use identical code.  Usage: python build_nb01.py"""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
OUT = SRC.parent / "notebooks" / "01_bushland_crop_water_validation_v2.ipynb"
# v2 (26 Oct 2026), after the first Colab run: Maturity defaults to -9 (placeholder) so the duration parameter is now
# chosen from the crop object; root depth (Zmax) and the SDI wetted surface are calibrated; Es and Tr are reported
# separately; soil-water storage enters the objective; a diagnostic cell runs first.
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Notebook 01 — Crop–water layer: AquaCrop calibration and validation on Bushland, Texas lysimeter data

**Purpose.** The digital twin uses AquaCrop-OSPy to turn weather and irrigation into soil water, crop ET and yield. Before it is used at the four study sites it is checked against measured data: USDA-ARS Bushland weighing lysimeters, maize, 2013, 2016 and 2018, four lysimeters per year (NE and SE subsurface drip; NW deficit and SW full linear-move sprinkler) = 12 lysimeter-seasons.

**Data (public, downloaded by this notebook).**
- Bushland maize collection, Ag Data Commons (CC0): water balance (DOI 10.15482/USDA.ADC/1526334), growth and yield (…/1526328), agronomic calendars (…/1526331), soil water (…/1526332).
- NASA POWER daily (community AG) at the site: Tmax, Tmin, RH, wind at 2 m, solar radiation, pressure → FAO-56 ET0.
- Measured lysimeter rain and irrigation are used as model inputs (NASA POWER rain differs from the site gauge).

**Design (v2).** Calibrate on 2013 and 2016 (8 seasons), validate on 2018 (4 seasons, not used for calibration). Calibrated: normalised water productivity (WP), basal crop coefficient (Kcb), maximum root depth (Zmax), crop duration (whichever of MaturityCD / Maturity the default maize file uses) and the wetted surface under subsurface drip. Compared: seasonal ET (with soil evaporation and transpiration reported separately), daily ET, soil-water storage in the top 1.5 m, grain yield.

**Why v2.** In the first run all lysimeters were simulated alike (ET about 760–800 mm, yield about 14.5 Mg/ha), so the drip-vs-sprinkler ET difference and the deficit-irrigation yield loss were missed; the Maturity parameter was a placeholder (-9) and had no effect.

**Runtime.** CPU is enough (about 15–30 min). **Output:** `export_nb01_v2.zip` (tables, figures, run info) is saved to Drive and downloaded.
""")

code(r"""
!pip -q install "aquacrop>=3,<4" openpyxl
import os, sys, json, gzip, time, platform, shutil, itertools, urllib.request, urllib.parse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from google.colab import drive, files
drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/solar_wind_irrigation'
RAW = f'{BASE}/data/raw/bushland_maize'; OUT = f'{BASE}/outputs/nb01_v2'; SRC = f'{BASE}/src'
for d in (RAW, OUT, SRC): os.makedirs(d, exist_ok=True)
sys.path.insert(0, SRC)
import aquacrop
print('aquacrop', aquacrop.__version__ if hasattr(aquacrop, '__version__') else '?', '| pandas', pd.__version__, '| numpy', np.__version__)
""")

md("## 1. Shared modules (identical to the locally tested `src/bushland_parse.py` and `src/et0.py`)")
for name in ("bushland_parse.py", "et0.py"):
    code(f"%%writefile {{SRC}}/{name}\n" + (SRC / name).read_text(encoding="utf-8"))
code(r"""
import importlib, bushland_parse as bp, et0 as et0m
importlib.reload(bp); importlib.reload(et0m)
""")

md("## 2. Download the public data (skips files already on Drive)")
code(r"""
ITEMS = {24855957: 'water_balance', 24855909: 'growth_yield', 24855873: 'calendars', 24855858: 'soil_water'}
def fetch(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r: return r.read()
        except Exception as e:
            print('  retry', k + 1, e); time.sleep(10 * (k + 1))
    raise RuntimeError(url)
sources = {}
for item, folder in ITEMS.items():
    m = json.loads(fetch(f'https://api.figshare.com/v2/articles/{item}'))
    sources[folder] = {'title': m['title'], 'doi': m.get('doi'), 'license': (m.get('license') or {}).get('name')}
    os.makedirs(f'{RAW}/{folder}', exist_ok=True)
    for f in m['files']:
        if 'maize' not in f['name'].lower() and not f['name'].lower().startswith('readme'): continue
        p = f'{RAW}/{folder}/{f["name"]}'
        if os.path.exists(p) and os.path.getsize(p) == f['size']: continue
        data = fetch(f['download_url']); assert len(data) == f['size'], f['name']
        open(p, 'wb').write(data); print('downloaded', folder, f['name'])
json.dump(sources, open(f'{RAW}/SOURCES.json', 'w'), indent=1)

LAT, LON, ELEV = 35.186714, -102.094189, 1170.0           # lysimeter site (collection metadata)
q = dict(parameters='ALLSKY_SFC_SW_DWN,T2M_MAX,T2M_MIN,RH2M,WS2M,PRECTOTCORR,PS', community='AG',
         longitude=LON, latitude=LAT, start='20120101', end='20191231', format='JSON')
pw = json.loads(fetch('https://power.larc.nasa.gov/api/temporal/daily/point?' + urllib.parse.urlencode(q)))
W = pd.DataFrame(pw['properties']['parameter']); W.index = pd.to_datetime(W.index, format='%Y%m%d')
assert not (W == -999).any().any(), 'NASA POWER fill values present'
W['ET0'] = et0m.et0_fao56(W.index, W.T2M_MAX, W.T2M_MIN, W.RH2M, W.WS2M, W.ALLSKY_SFC_SW_DWN, LAT, ELEV, W.PS)
W.to_csv(f'{OUT}/nasa_power_daily_bushland_2012_2019.csv')
print(W.loc['2013-05-15':'2013-09-30', 'ET0'].sum().round(0), 'mm ET0, 15 May-30 Sep 2013 (local test gave 1000)')
""")

md("## 3. Measured inputs and targets for the 12 lysimeter-seasons")
code(r"""
SEASONS = []
for y in (2013, 2016, 2018):
    wb = bp.daily_water_balance(RAW, y)
    for side, lys_list, method in (('East', ['NE', 'SE'], 'SDI'), ('West', ['NW', 'SW'], 'sprinkler')):
        sow = bp.planting_date(bp.calendar(RAW, y, side))
        prof = bp.profile_storage(RAW, y, side)
        fc, pwp = float(prof['FC'].dropna().iloc[0]), float(prof['PWP'].dropna().iloc[0])
        yl = bp.lysimeter_yield(RAW, y, side)
        for L in lys_list:
            s0, s0_date = bp.initial_storage(prof, L, sow)
            obs = wb[wb.lysimeter == L].set_index('date').sort_index()
            col = [c for c in prof.columns if f'{L} Lys' in str(c) and '1.5 m' in str(c)][0]
            st = pd.Series(pd.to_numeric(prof[col], errors='coerce').values, index=prof['date']).dropna()
            SEASONS.append(dict(year=y, lys=L, method=method, sow=sow, fc_mm=fc, pwp_mm=pwp, s0_mm=s0, s0_date=s0_date,
                                obs=obs, storage=st, yield_obs=float(yl.loc[yl.lysimeter == L, 'mean'].iloc[0]),
                                yield_sd=float(yl.loc[yl.lysimeter == L, 'std'].iloc[0])))
summary = pd.DataFrame([{k: v for k, v in s.items() if k not in ('obs', 'storage')} for s in SEASONS])
summary['ET_obs_sow_30Sep'] = [s['obs'].loc[s['sow']:f"{s['year']}-09-30", 'et'].sum() for s in SEASONS]
summary['irrig_mm'] = [s['obs'].loc[s['sow']:f"{s['year']}-10-31", 'irrig'].sum() for s in SEASONS]
summary.to_csv(f'{OUT}/seasons_measured.csv', index=False); summary.round(2)
""")

md(r"""
## 4. AquaCrop set-up
- **Soil:** one uniform profile from the measured Bushland limits: θFC = FC/1500 mm, θWP = PWP/1500 mm (top 1.5 m), saturation 0.46 and Ksat 40 mm/d (assumed for Pullman clay loam; sensitivity is listed as an author query). Compartments: 15 × 0.1 m + 6 × 0.15 m = 2.4 m, so storage in the top 1.5 m is the sum of the first 15 compartments.
- **Weather:** Tmin and Tmax from NASA POWER, ET0 from FAO-56, rain = lysimeter-measured.
- **Irrigation:** measured daily depths as a fixed schedule (`irrigation_method=3`). Wetted surface: 100% for sprinkler; for subsurface drip 1% or 20%, chosen by calibration.
- **Start:** sowing date; initial water = measured storage closest to sowing, spread uniformly.
""")
code(r"""
from aquacrop import AquaCropModel, Soil, Crop, InitialWaterContent, IrrigationManagement
DZ = [0.1] * 15 + [0.15] * 6

def make_soil(fc_mm, pwp_mm):
    s = Soil('custom', dz=DZ)
    s.add_layer(thickness=s.zSoil, thWP=pwp_mm / 1500, thFC=fc_mm / 1500, thS=0.46, Ksat=40, penetrability=100)
    return s

def run_season(s, params=None):
    params = dict(params or {}); wet_sdi = params.pop('wet_sdi', 20)
    y, sow = s['year'], s['sow']
    end = pd.Timestamp(f'{y}-12-15')
    w = W.loc[sow:end]
    rain = s['obs']['precip'].reindex(w.index).fillna(0.0).clip(lower=0)
    wdf = pd.DataFrame({'MinTemp': w.T2M_MIN.values, 'MaxTemp': w.T2M_MAX.values, 'Precipitation': rain.values,
                        'ReferenceET': w.ET0.values, 'Date': w.index})
    irr = s['obs']['irrig'].loc[sow:end].fillna(0.0)
    sched = pd.DataFrame({'Date': irr.index[irr > 0], 'Depth': irr[irr > 0].values})
    crop = Crop('Maize', planting_date=sow.strftime('%m/%d'), **params)
    model = AquaCropModel(sim_start_time=sow.strftime('%Y/%m/%d'), sim_end_time=end.strftime('%Y/%m/%d'), weather_df=wdf,
                          soil=make_soil(s['fc_mm'], s['pwp_mm']), crop=crop,
                          initial_water_content=InitialWaterContent(wc_type='Num', method='Layer', depth_layer=[1], value=[s['s0_mm'] / 1500]),
                          irrigation_management=IrrigationManagement(irrigation_method=3, Schedule=sched,
                                                                     WetSurf=wet_sdi if s['method'] == 'SDI' else 100))
    model.run_model(till_termination=True)
    res = model.get_simulation_results()
    flux = model._outputs.water_flux.copy(); stor = model._outputs.water_storage.copy()
    flux['date'] = sow + pd.to_timedelta(flux['time_step_counter'], unit='D')
    stor['date'] = sow + pd.to_timedelta(stor['time_step_counter'], unit='D')
    flux = flux[flux['dap'] > 0].set_index('date')
    th = [c for c in stor.columns if c.startswith('th') and c[2:].isdigit()]
    stor = stor.set_index('date'); stor['S15_mm'] = stor[th[:15]].sum(axis=1) * 100.0
    ycol = [c for c in res.columns if 'dry yield' in c.lower()] or [c for c in res.columns if 'yield' in c.lower() and 'potential' not in c.lower()]
    hcol = [c for c in res.columns if 'harvest date' in c.lower() and 'yyyy' in c.lower()]
    return dict(yield_sim=float(res[ycol[0]].iloc[0]), maturity=pd.to_datetime(res[hcol[0]].iloc[0]) if hcol else pd.NaT,
                et=(flux['Es'] + flux['Tr']), es=flux['Es'], tr=flux['Tr'], storage=stor['S15_mm'])

# default maize parameters actually used by AquaCrop-OSPy (printed for the methods section)
D0 = Crop('Maize', planting_date='05/01')
crop_defaults = {k: v for k, v in vars(D0).items() if not k.startswith('_') and isinstance(v, (int, float, str))}
json.dump(crop_defaults, open(f'{OUT}/maize_default_parameters.json', 'w'), indent=1, default=str)
DUR = 'MaturityCD' if getattr(D0, 'CalendarType', 2) == 1 else 'Maturity'
print('CalendarType', getattr(D0, 'CalendarType', None), '| duration parameter', DUR, '=', getattr(D0, DUR, None),
      '| Zmax', getattr(D0, 'Zmax', None), '| WP', D0.WP, '| Kcb', D0.Kcb, '| HI0', D0.HI0)
assert getattr(D0, DUR, -9) > 0, f'{DUR} is not a positive number; check maize_default_parameters.json'

# smoke test: one season, default maize parameters
t = run_season(SEASONS[0])
print('2013 NE default: yield', round(t['yield_sim'], 2), 'Mg/ha (obs', round(SEASONS[0]['yield_obs'], 2), ') | maturity', t['maturity'],
      '| ET sow-30Sep', round(t['et'].loc[:'2013-09-30'].sum()), 'mm (obs', round(summary.ET_obs_sow_30Sep.iloc[0]), ')')
""")

md("## 5. Metrics")
code(r"""
def metrics(obs, sim):
    o, s = np.asarray(obs, float), np.asarray(sim, float); k = np.isfinite(o) & np.isfinite(s); o, s = o[k], s[k]
    e = s - o
    return dict(n=len(o), rmse=np.sqrt(np.mean(e ** 2)), nrmse_pct=100 * np.sqrt(np.mean(e ** 2)) / np.mean(o), bias=np.mean(e),
                nse=1 - np.sum(e ** 2) / np.sum((o - o.mean()) ** 2) if len(o) > 1 else np.nan,
                d=1 - np.sum(e ** 2) / np.sum((np.abs(s - o.mean()) + np.abs(o - o.mean())) ** 2) if len(o) > 1 else np.nan,
                r2=np.corrcoef(o, s)[0, 1] ** 2 if len(o) > 2 else np.nan)

def evaluate(params, seasons):
    rows, daily, stor = [], [], []
    for s in seasons:
        r = run_season(s, params)
        win = slice(s['sow'], pd.Timestamp(f"{s['year']}-09-30"))
        eo, es_ = s['obs']['et'].loc[win], r['et'].loc[win]
        j = eo.index.intersection(es_.index)
        daily.append(pd.DataFrame({'year': s['year'], 'lys': s['lys'], 'obs': eo.loc[j].values, 'sim': es_.loc[j].values}, index=j))
        so = s['storage'].loc[s['sow']:pd.Timestamp(f"{s['year']}-10-31")]
        js = so.index.intersection(r['storage'].index)
        stor.append(pd.DataFrame({'year': s['year'], 'lys': s['lys'], 'obs': so.loc[js].values, 'sim': r['storage'].loc[js].values}, index=js))
        rows.append(dict(year=s['year'], lys=s['lys'], method=s['method'], ET_obs=eo.loc[j].sum(), ET_sim=es_.loc[j].sum(),
                         Es_sim=r['es'].loc[win].sum(), Tr_sim=r['tr'].loc[win].sum(),
                         yield_obs=s['yield_obs'], yield_sim=r['yield_sim'], maturity_sim=r['maturity']))
    return pd.DataFrame(rows), pd.concat(daily), pd.concat(stor)

def score(tab, stor):  # calibration objective: mean nRMSE (%) of seasonal ET, grain yield and 1.5 m soil-water storage
    return np.mean([metrics(tab.ET_obs, tab.ET_sim)['nrmse_pct'], metrics(tab.yield_obs, tab.yield_sim)['nrmse_pct'],
                    metrics(stor.obs, stor.sim)['nrmse_pct']])
""")

md("""
## 5b. Diagnostic (calibration years only): do drip wetted surface and root depth change ET split and deficit yield?
2013 NE (subsurface drip) and 2013 NW (deficit sprinkler, measured yield 7.9 Mg/ha) under three wetted-surface settings and three root depths, all other parameters default.
""")
code(r"""
diag = []
for s in [x for x in SEASONS if x['year'] == 2013 and x['lys'] in ('NE', 'NW')]:
    for wet in (100, 20, 1):
        if s['method'] != 'SDI' and wet != 100: continue
        for zmax in (D0.Zmax, 1.5, 1.2):
            r = run_season(s, {'wet_sdi': wet, 'Zmax': zmax}); win = slice(s['sow'], pd.Timestamp('2013-09-30'))
            diag.append(dict(lys=s['lys'], wet_sdi=wet if s['method'] == 'SDI' else 100, Zmax=zmax,
                             ET=r['et'].loc[win].sum(), Es=r['es'].loc[win].sum(), Tr=r['tr'].loc[win].sum(), yield_sim=r['yield_sim'],
                             ET_obs=s['obs']['et'].loc[win].sum(), yield_obs=s['yield_obs']))
diag = pd.DataFrame(diag); diag.to_csv(f'{OUT}/diagnostic_wet_zmax_2013.csv', index=False); diag.round(2)
""")

md("## 6. Calibration on 2013 + 2016 (8 seasons); 2018 untouched")
code(r"""
CAL = [s for s in SEASONS if s['year'] in (2013, 2016)]; VAL = [s for s in SEASONS if s['year'] == 2018]
GRID = {'WP': [D0.WP * f for f in (0.8, 0.9, 1.0)], 'Kcb': [D0.Kcb * f for f in (0.9, 0.95, 1.0)],
        'Zmax': [1.2, 1.5, D0.Zmax], DUR: [getattr(D0, DUR) * f for f in (0.95, 1.0, 1.05)], 'wet_sdi': [1, 20]}
GRID[DUR] = [round(v) for v in GRID[DUR]]
runs, t0 = [], time.time()
for i, combo in enumerate(itertools.product(*GRID.values())):
    p = dict(zip(GRID, combo))
    try:
        tab, _, stor = evaluate(p, CAL); runs.append({**p, 'score': score(tab, stor)})
    except Exception as e:
        runs.append({**p, 'score': np.nan, 'error': str(e)[:120]})
    if i % 20 == 0: print(i, f'{time.time() - t0:.0f}s', runs[-1])
runs = pd.DataFrame(runs).sort_values('score'); runs.to_csv(f'{OUT}/calibration_grid.csv', index=False)
assert runs.score.notna().any(), runs.head()
BEST = runs.iloc[0][list(GRID)].to_dict(); DEFAULT = {}
edge = {k: v for k, v in BEST.items() if len(GRID[k]) > 2 and v in (min(GRID[k]), max(GRID[k]))}
print('best:', BEST, '| score', round(runs.iloc[0]['score'], 2), '| on grid edge:', edge or 'none')
""")

md("## 7. Results: default vs calibrated parameters, calibration years vs held-out 2018")
code(r"""
out = []
for label, p in (('default', DEFAULT), ('calibrated', BEST)):
    for split, ss in (('calibration_2013_2016', CAL), ('validation_2018', VAL)):
        tab, daily, stor = evaluate(p, ss)
        tab.assign(params=label, split=split).to_csv(f'{OUT}/seasons_{label}_{split}.csv', index=False)
        daily.to_csv(f'{OUT}/daily_et_{label}_{split}.csv'); stor.to_csv(f'{OUT}/storage_{label}_{split}.csv')
        for q, (o, s_) in {'seasonal_ET_mm': (tab.ET_obs, tab.ET_sim), 'daily_ET_mm': (daily.obs, daily.sim),
                           'storage_1p5m_mm': (stor.obs, stor.sim), 'yield_Mg_ha': (tab.yield_obs, tab.yield_sim)}.items():
            out.append({'params': label, 'split': split, 'quantity': q, **metrics(o, s_)})
M = pd.DataFrame(out); M.to_csv(f'{OUT}/validation_metrics.csv', index=False)
M.round(3)
""")
code(r"""
tabs = pd.concat([pd.read_csv(f'{OUT}/seasons_calibrated_{s}.csv') for s in ('calibration_2013_2016', 'validation_2018')])
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, (o, s_, lab) in zip(ax[:2], (('ET_obs', 'ET_sim', 'Seasonal ET, sowing–30 Sep (mm)'), ('yield_obs', 'yield_sim', 'Grain yield (Mg/ha)'))):
    for split, mk in (('calibration_2013_2016', 'o'), ('validation_2018', 's')):
        t = tabs[tabs.split == split]; a.scatter(t[o], t[s_], marker=mk, label=split)
    lo, hi = min(tabs[o].min(), tabs[s_].min()), max(tabs[o].max(), tabs[s_].max()); a.plot([lo, hi], [lo, hi], 'k--', lw=.8)
    a.set_xlabel('Measured'); a.set_ylabel('Simulated'); a.set_title(lab, fontsize=9); a.legend(fontsize=7)
d18 = pd.read_csv(f'{OUT}/daily_et_calibrated_validation_2018.csv', index_col=0, parse_dates=True)
for L in sorted(d18.lys.unique()):
    t = d18[d18.lys == L]; ax[2].plot(t.index, t.obs.cumsum(), label=f'{L} measured'); ax[2].plot(t.index, t.sim.cumsum(), '--', label=f'{L} simulated')
ax[2].set_title('Cumulative ET, 2018 (held out)', fontsize=9); ax[2].legend(fontsize=6, ncol=2); fig.autofmt_xdate()
plt.tight_layout(); plt.savefig(f'{OUT}/fig_validation.png', dpi=200); plt.show()
""")

md("## 8. Export for checking (`export_nb01_v2.zip`)")
code(r"""
info = {'run_utc': pd.Timestamp.utcnow().isoformat(), 'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
        'aquacrop': __import__('importlib.metadata').metadata.version('aquacrop'), 'notebook_version': 'v2',
        'duration_parameter': DUR, 'best_params': BEST, 'grid': GRID,
        'assumptions': {'thS': 0.46, 'Ksat_mm_d': 40, 'WetSurf_sprinkler_pct': 100,
                        'weather': 'NASA POWER daily (Tmin, Tmax, ET0 FAO-56); rain and irrigation measured'}}
json.dump(info, open(f'{OUT}/run_info.json', 'w'), indent=1, default=str)
shutil.make_archive(f'{BASE}/outputs/export_nb01_v2', 'zip', OUT)
files.download(f'{BASE}/outputs/export_nb01_v2.zip')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
                                    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(OUT, len(cells), "cells")
