"""Build notebooks/02_energy_resource_sizing.ipynb (Colab). Embeds the locally tested src/energy.py, src/et0.py and
src/sites.json verbatim.  Usage: python build_nb02.py"""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
OUT = SRC.parent / "notebooks" / "02_energy_resource_sizing.ipynb"
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Notebook 02 — Energy layer: solar and wind resource, irrigation demand and system sizing at four water-scarce sites

**Purpose.** Build the energy side of the digital twin and size an off-grid PV–wind–battery–pump–tank farm microgrid for each site before any controller is compared. The microgrid serves (i) irrigation pumping, which is seasonal and flexible through the tank, and (ii) year-round farm loads (farmhouse appliances, a produce cold room, a workshop), which are critical and partly at night.

**Sites.** Bushland (Texas, USA), Jodhpur (Rajasthan, India), Taroudant (Souss-Massa, Morocco), Azraq (Jordan).

**Data (public, downloaded here).** NASA POWER hourly (2001–2024: GHI, DNI, DHI, temperature, humidity, rain, wind at 10 and 50 m, pressure) and daily (agro-climatology community) for FAO-56 ET0.

**Models.** PV: NOAA solar position, isotropic-sky plane-of-array irradiance (tilt = latitude, equator-facing), NOCT cell temperature, PVWatts-type output with 14% system losses and 96% inverter efficiency; cross-checked against pvlib. Wind: hourly shear exponent from the 10 m and 50 m winds, power-law extrapolation to a 30 m hub, IEC air-density correction, generic small-turbine curve (3/12/25 m/s), 10% losses. Demand for sizing: FAO-56 single-crop-coefficient maize (5 ha, drip efficiency 0.90), simple daily effective rain. Farm load: appliance-based stochastic profile (RAMP-style; randomised daily use windows, seeded) with temperature-dependent fans and cold-room compressor. System: variable-speed pump (wire-to-water efficiency 0.55, 30% minimum power), tank, Li-ion battery (SoC 10–90%, 0.96/0.96 charge/discharge, 0.5C), rule-based dispatch (farm load first, then pump, then battery; the battery tops up the pump only above a 50% night reserve).

**Sizing.** Grid search over PV, wind, battery and tank sizes; feasible if the water loss-of-supply probability (WLPSP) is at most 5% and the farm-load energy loss-of-supply probability (ELPSP) at most 2% on 2001–2016; least net present cost (NPC) chosen; reliability then re-checked on 2020–2024 (not used for sizing). Diesel reference: a genset supplying the same pumping and farm-load energy at constant efficiency (favours diesel). Costs are sourced base values (IRENA 2024, PNNL distributed wind 2025/2026, NREL ATB 2024b; tank and pump assumed) with site-specific 2024 diesel prices; sources in `energy.COST_SOURCES`, bounds in `energy.COST_SCENARIOS`.

**Runtime.** CPU; about 45–75 min (sizing about 4–6 min per site, head sensitivity about 15 min, hourly data download about 10–15 min on the first run). Keep the browser tab open. **Output:** `export_nb02.zip`.
""")

code(r"""
!pip -q install pvlib pyarrow
import os, sys, json, gzip, time, platform, shutil, urllib.request, urllib.parse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from google.colab import drive, files
drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/solar_wind_irrigation'
RAWP = f'{BASE}/data/raw/nasa_power'; PROC = f'{BASE}/data/processed'; OUT = f'{BASE}/outputs/nb02'; SRC = f'{BASE}/src'
for d in (RAWP, PROC, OUT, SRC): os.makedirs(d, exist_ok=True)
sys.path.insert(0, SRC)
""")

md("## 1. Shared modules and site list (identical to the locally tested `src/energy.py`, `src/et0.py`, `src/sites.json`)")
for name in ("energy.py", "et0.py", "sites.json"):
    code(f"%%writefile {{SRC}}/{name}\n" + (SRC / name).read_text(encoding="utf-8"))
code(r"""
import importlib, energy as en, et0 as et0m
importlib.reload(en); importlib.reload(et0m)
SITES = json.load(open(f'{SRC}/sites.json', encoding='utf-8'))
# ASSUMPTIONS for sizing (author queries): maize sowing date per site, pumping head, farm area
SOW = {'bushland_tx': '05-15', 'jodhpur_in': '07-01', 'souss_ma': '03-15', 'azraq_jo': '04-01'}
HEAD_M = 60.0; AREA_HA = 5.0; PUMP_HOURS = 8.0; LOAD_SEED = 1
TRAIN, TEST = slice('2001-01-01', '2016-12-31'), slice('2020-01-01', '2024-12-31')
""")

md("## 2. Download NASA POWER hourly (2001–2024) and daily data (skips files already on Drive)")
code(r"""
HOURLY = 'ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,RH2M,PRECTOTCORR,WS10M,WS50M,PS'
DAILY = 'ALLSKY_SFC_SW_DWN,T2M_MAX,T2M_MIN,T2M,RH2M,WS2M,PRECTOTCORR,PS'
def fetch(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r: return r.read()
        except Exception as e:
            print('  retry', k + 1, e); time.sleep(10 * (k + 1))
    raise RuntimeError(url)
def save(path, payload):
    json.loads(payload)
    with gzip.open(path, 'wb') as f: f.write(payload)
for sid, s in SITES.items():
    d = f'{RAWP}/{sid}'; os.makedirs(d, exist_ok=True)
    f = f'{d}/{sid}_daily_2001_2024.json.gz'
    if not os.path.exists(f):
        q = dict(parameters=DAILY, community='AG', longitude=s['lon'], latitude=s['lat'], start='20010101', end='20241231', format='JSON')
        save(f, fetch('https://power.larc.nasa.gov/api/temporal/daily/point?' + urllib.parse.urlencode(q)))
    for y in range(2001, 2025):
        f = f'{d}/{sid}_hourly_{y}.json.gz'
        if os.path.exists(f): continue
        q = dict(parameters=HOURLY, community='RE', longitude=s['lon'], latitude=s['lat'], start=f'{y}0101', end=f'{y}1231',
                 format='JSON', **{'time-standard': 'UTC'})
        save(f, fetch('https://power.larc.nasa.gov/api/temporal/hourly/point?' + urllib.parse.urlencode(q))); time.sleep(1)
    print(sid, 'files:', len(os.listdir(d)))
""")

md("## 3. Resource assessment, irrigation demand and processed tables (saved for notebooks 03+)")
code(r"""
R, summary = {}, []
for sid, s in SITES.items():
    h = en.load_power_hourly(f'{RAWP}/{sid}', sid)
    assert len(h) == 210384 and not h.isna().any().any(), (sid, len(h), int(h.isna().sum().sum()))
    shift, corr = en.best_time_shift(h, s['lat'], s['lon'])
    pv, poa = en.pv_output_per_kwp(h, s['lat'], s['lon'], shift_min=shift)
    wd, vhub, alpha = en.wind_output_per_kw(h)
    dly, elev = en.load_power_daily(f'{RAWP}/{sid}/{sid}_daily_2001_2024.json.gz')
    et0 = pd.Series(et0m.et0_fao56(dly.index, dly.T2M_MAX, dly.T2M_MIN, dly.RH2M, dly.WS2M, dly.ALLSKY_SFC_SW_DWN, s['lat'], elev, dly.PS), index=dly.index)
    dem = en.irrigation_demand(et0, dly.PRECTOTCORR, SOW[sid], area_ha=AREA_HA)
    load = en.farm_load(h.T2M, s['lon'], seed=LOAD_SEED)
    R[sid] = dict(pv=pv, wind=wd, dem=dem, load=load, shift=shift, elev=elev)
    hourly = pd.DataFrame({'pv_kw_per_kwp': pv, 'wind_kw_per_kw': wd, 'farm_load_kw': load, 'poa_w_m2': poa, 'v_hub': vhub,
                           'ghi': h.ALLSKY_SFC_SW_DWN, 't2m': h.T2M, 'rh2m': h.RH2M, 'ws50m': h.WS50M, 'rain_mm': h.PRECTOTCORR})
    hourly.to_parquet(f'{PROC}/{sid}_hourly.parquet')
    pd.DataFrame({'et0_mm': et0, 'rain_mm': dly.PRECTOTCORR, 'tmax': dly.T2M_MAX, 'tmin': dly.T2M_MIN, 'demand_m3': dem}).to_parquet(f'{PROC}/{sid}_daily.parquet')
    mpv, mwd, mdem = pv.resample('MS').mean(), wd.resample('MS').mean(), dem.resample('MS').sum()
    summary.append(dict(site=sid, name=s['name'], power_cell_elev_m=elev, timestamp_shift_min=shift,
                        pv_kwh_per_kwp_yr=pv.groupby(pv.index.year).sum().mean(), pv_yield_cv=pv.groupby(pv.index.year).sum().std() / pv.groupby(pv.index.year).sum().mean(),
                        poa_kwh_m2_yr=poa.groupby(poa.index.year).sum().mean() / 1000, wind_cf_30m=wd.mean(), mean_vhub=vhub.mean(), mean_shear=alpha.mean(),
                        et0_mm_yr=et0.groupby(et0.index.year).sum().mean(), demand_m3_season=dem.groupby(dem.index.year).sum().mean(),
                        peak_demand_m3_day=dem.loc[TRAIN].max(), farm_load_kwh_yr=load.groupby(load.index.year).sum().mean(),
                        farm_load_peak_kw=load.max(), r_hourly_pv_wind=np.corrcoef(pv, wd)[0, 1],
                        r_monthly_pv_wind=np.corrcoef(mpv, mwd)[0, 1]))
    print(sid, 'done')
S = pd.DataFrame(summary); S.to_csv(f'{OUT}/resource_summary.csv', index=False); S.round(3)
""")
code(r"""
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharey=False)
for a, (sid, r) in zip(ax, R.items()):
    m = lambda x: x.groupby(x.index.month).mean()
    a.plot(range(1, 13), m(r['pv']) * 24, label='PV, kWh/kWp/day'); a.plot(range(1, 13), m(r['wind']) * 24, label='Wind, kWh/kW/day')
    a.plot(range(1, 13), m(r['load']) * 24 / 10, ls=':', color='k', label='Farm load, kWh/day ÷10')
    a2 = a.twinx(); dm = r['dem'].groupby([r['dem'].index.year, r['dem'].index.month]).sum().groupby(level=1).mean()
    a2.bar(dm.index, dm.values, alpha=.25, color='tab:green', label='Irrigation demand'); a2.set_ylabel('m³/month', fontsize=8)
    a.set_title(SITES[sid]['name'], fontsize=9); a.set_xticks(range(1, 13)); a.legend(fontsize=7, loc='upper left')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_resource_demand.png', dpi=200); plt.show()
""")

md("## 4. Cross-check of the PV irradiance model against pvlib (2020, same inputs)")
code(r"""
import pvlib
chk = []
for sid, s in SITES.items():
    h = en.load_power_hourly(f'{RAWP}/{sid}', sid).loc['2020']
    t = h.index + pd.Timedelta(minutes=R[sid]['shift'])
    sp = pvlib.solarposition.get_solarposition(t, s['lat'], s['lon'])
    tilt, az = abs(s['lat']), 180.0
    tot = pvlib.irradiance.get_total_irradiance(tilt, az, sp['apparent_zenith'].values, sp['azimuth'].values,
                                                h.ALLSKY_SFC_SW_DNI.values, h.ALLSKY_SFC_SW_DWN.values, h.ALLSKY_SFC_SW_DIFF.values,
                                                albedo=0.25, model='isotropic')
    poa_pvlib = pd.Series(np.nan_to_num(tot['poa_global']), index=h.index)
    _, poa_own = en.pv_output_per_kwp(h, s['lat'], s['lon'], shift_min=R[sid]['shift'])
    chk.append(dict(site=sid, poa_own_kwh_m2=poa_own.sum() / 1000, poa_pvlib_kwh_m2=poa_pvlib.sum() / 1000,
                    diff_pct=100 * (poa_own.sum() / poa_pvlib.sum() - 1), r2_hourly=np.corrcoef(poa_own, poa_pvlib)[0, 1] ** 2))
C = pd.DataFrame(chk); C.to_csv(f'{OUT}/pvlib_crosscheck.csv', index=False); C.round(3)
""")

md("## 5. Sizing at each site (head 60 m; feasible if WLPSP ≤ 5% and ELPSP ≤ 2% on 2001–2016; reliability re-checked on 2020–2024)")
code(r"""
GRID = en.GRID_DEFAULT
picks, t0 = {}, time.time()
for sid, r in R.items():
    peak = r['dem'].loc[TRAIN].max()
    pump_kw = peak / PUMP_HOURS * en.RHO_W * en.G * HEAD_M / 0.55 / 3.6e6
    allc, pick = en.size_system(r['pv'], r['wind'], r['dem'], HEAD_M, TRAIN, TEST, pump_kw, peak, load_kw=r['load'], grid=GRID,
                                c=en.site_costs(sid))
    allc.to_csv(f'{OUT}/sizing_all_{sid}.csv', index=False)
    picks[sid] = {'pump_kw': pump_kw, 'peak_m3_day': peak, **pick}
    print(f'{sid}: {len(allc)} configs, {time.time() - t0:.0f}s, least-cost design:',
          {k: round(v, 1) for k, v in pick['least_cost'].items() if k in ('pv_kwp', 'wind_kwr', 'batt_kwh', 'tank_m3', 'npc_usd')})
json.dump(picks, open(f'{OUT}/sizing_picks.json', 'w'), indent=1, default=float)
rows = []
for sid, p in picks.items():
    for kind in ('least_cost', 'hybrid', 'pv_only', 'wind_only'):
        if kind in p:
            b = p[kind]
            rows.append(dict(site=sid, design=kind, pump_kw=p['pump_kw'], pv_kwp=b['pv_kwp'], wind_kw=b['wind_kwr'], batt_kwh=b['batt_kwh'],
                             tank_m3=b['tank_m3'], npc_usd=b['npc_usd'], lcoe_usd_kwh=b['lcoe_usd_kwh'], curtail_frac=b['curtail_frac'],
                             batt_cycles_yr=b['batt_cycles_yr'], wlpsp_train=b['wlpsp_train'], wlpsp_test=b['wlpsp_test'],
                             elpsp_train=b['elpsp_train'], elpsp_test=b['elpsp_test'],
                             diesel_npc_usd=b['diesel']['npc'], co2_avoided_t_yr=b['diesel']['co2_t_per_year']))
D = pd.DataFrame(rows); D.to_csv(f'{OUT}/sizing_summary.csv', index=False); D.round(3)
""")
code(r"""
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for a, sid in zip(ax, R):
    allc = pd.read_csv(f'{OUT}/sizing_all_{sid}.csv')
    for lab, m, c in (('PV only', (allc.wind_kwr == 0), 'tab:orange'), ('Wind only', (allc.pv_kwp == 0), 'tab:blue'),
                      ('Hybrid', (allc.pv_kwp > 0) & (allc.wind_kwr > 0), 'tab:green')):
        a.scatter(allc.loc[m, ['wlpsp_train', 'elpsp_train']].max(axis=1) * 100, allc.loc[m, 'npc_usd'] / 1e3, s=8, alpha=.6, label=lab, color=c)
    a.axvline(5, color='k', lw=.6, ls='--'); a.set_xlabel('max(WLPSP, ELPSP) 2001–2016 (%)'); a.set_ylabel('NPC (k USD)')
    a.set_xlim(-1, 40); a.set_title(SITES[sid]['name'], fontsize=9); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_sizing_tradeoff.png', dpi=200); plt.show()
""")

md("## 6. Sensitivity: sourced cost bounds, discount rate, battery life, diesel price (re-scored, same simulations) and pumping head (re-sized on a reduced grid; battery and PV reach 4x pump power because the farm load does not scale with head)")
code(r"""
sens = []
for sid in R:
    allc = pd.read_csv(f'{OUT}/sizing_all_{sid}.csv'); feas = allc[(allc.wlpsp_train <= 0.05) & (allc.elpsp_train <= 0.02)].copy(); pk = picks[sid]
    for name in en.COST_SCENARIOS:
        c = en.site_costs(sid, name)
        feas['npc_s'] = [en.npc(r.pv_kwp, r.wind_kwr, r.batt_kwh, r.tank_m3, pk['pump_kw'], c=c) for r in feas.itertuples()]
        b = feas.sort_values('npc_s').iloc[0]
        dz = en.diesel_reference(b.pump_kwh_yr + b.load_kwh_yr, pk['pump_kw'], float(R[sid]['load'].max()), c)
        sens.append(dict(site=sid, scenario=name, pv_kwp=b.pv_kwp, wind_kw=b.wind_kwr, batt_kwh=b.batt_kwh, tank_m3=b.tank_m3,
                         npc_usd=b.npc_s, diesel_npc_usd=dz['npc'], renewable_cheaper=b.npc_s < dz['npc']))
for sid, r in R.items():
    for head in (30.0, 90.0, 120.0):
        peak = r['dem'].loc[TRAIN].max(); pump_kw = peak / PUMP_HOURS * en.RHO_W * en.G * head / 0.55 / 3.6e6
        _, pk = en.size_system(r['pv'], r['wind'], r['dem'], head, TRAIN, TEST, pump_kw, peak, load_kw=r['load'],
                               grid={'pv': [1, 1.5, 2, 3, 4], 'wind': [0, 0.5, 1], 'batt': [0, 1, 2, 3, 4], 'tank': [0.03, 0.0625, 0.125, 0.25]},
                               c=en.site_costs(sid))
        b = pk.get('least_cost')
        if b:
            sens.append(dict(site=sid, scenario=f'head_{int(head)}m', pv_kwp=b['pv_kwp'], wind_kw=b['wind_kwr'], batt_kwh=b['batt_kwh'],
                             tank_m3=b['tank_m3'], npc_usd=b['npc_usd'], diesel_npc_usd=b['diesel']['npc'],
                             renewable_cheaper=b['npc_usd'] < b['diesel']['npc']))
SENS = pd.DataFrame(sens); SENS.to_csv(f'{OUT}/sensitivity.csv', index=False); SENS.round(1)
""")

md("## 7. Export for checking (`export_nb02.zip`)")
code(r"""
info = {'run_utc': pd.Timestamp.utcnow().isoformat(), 'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
        'pvlib': pvlib.__version__, 'sow': SOW, 'head_m': HEAD_M, 'area_ha': AREA_HA, 'pump_hours': PUMP_HOURS, 'grid': GRID,
        'farm_load_assumed': en.FARM_LOAD, 'load_seed': LOAD_SEED, 'wlpsp_max': 0.05, 'elpsp_max': 0.02,
        'costs': en.COSTS, 'diesel_usd_per_l': en.DIESEL_USD_PER_L, 'cost_scenarios': en.COST_SCENARIOS,
        'cost_sources': en.COST_SOURCES, 'train': [TRAIN.start, TRAIN.stop], 'test': [TEST.start, TEST.stop]}
json.dump(info, open(f'{OUT}/run_info.json', 'w'), indent=1, default=str)
shutil.make_archive(f'{BASE}/outputs/export_nb02', 'zip', OUT)
files.download(f'{BASE}/outputs/export_nb02.zip')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
                                    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(OUT, len(cells), "cells")
