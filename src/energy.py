"""Energy layer of the digital twin: PV and wind generation from NASA POWER hourly data, FAO-56 irrigation demand for
sizing, an hourly PV-wind-battery-pump-tank simulation with a rule-based dispatch, and simple techno-economics.

All functions are plain numpy/pandas so they can be tested without pvlib; the Colab notebook cross-checks the PV model
against pvlib. Units: kW, kWh, m3, mm, deg C, m/s. Times are UTC unless stated.
"""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

RHO_W, G = 1000.0, 9.81


# ---------------------------------------------------------------- data
def load_power_hourly(folder, site):
    """Concatenate NASA POWER hourly files <site>_hourly_<year>.json.gz into one UTC-indexed DataFrame."""
    frames = []
    for f in sorted(Path(folder).glob(f"{site}_hourly_*.json.gz")):
        p = json.load(gzip.open(f))["properties"]["parameter"]
        d = pd.DataFrame(p)
        d.index = pd.to_datetime(d.index, format="%Y%m%d%H", utc=True)
        frames.append(d)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()]
    return df.replace(-999.0, np.nan)


def load_power_daily(path):
    """Daily POWER file -> (DataFrame, elevation of the POWER grid cell in m, from the response geometry)."""
    j = json.load(gzip.open(path))
    d = pd.DataFrame(j["properties"]["parameter"])
    d.index = pd.to_datetime(d.index, format="%Y%m%d")
    coords = j.get("geometry", {}).get("coordinates", [None, None, None])
    return d.replace(-999.0, np.nan), (coords[2] if len(coords) > 2 else None)


# ---------------------------------------------------------------- solar geometry (NOAA general solar position)
def solar_position(times_utc, lat, lon):
    """Zenith and azimuth (deg, azimuth clockwise from north) for UTC timestamps (NOAA spreadsheet equations)."""
    t = pd.DatetimeIndex(times_utc)
    doy = t.dayofyear.values
    hr = t.hour.values + t.minute.values / 60.0
    gam = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(gam) - 0.032077 * np.sin(gam) - 0.014615 * np.cos(2 * gam) - 0.040849 * np.sin(2 * gam))
    dec = (0.006918 - 0.399912 * np.cos(gam) + 0.070257 * np.sin(gam) - 0.006758 * np.cos(2 * gam) + 0.000907 * np.sin(2 * gam)
           - 0.002697 * np.cos(3 * gam) + 0.00148 * np.sin(3 * gam))
    tst = hr * 60 + eot + 4 * lon                      # true solar time, minutes
    ha = np.radians(tst / 4 - 180)
    phi = np.radians(lat)
    cosz = np.clip(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha), -1, 1)
    zen = np.arccos(cosz)
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * np.sin(phi) - np.tan(dec) * np.cos(phi))) + 180
    return np.degrees(zen), az % 360


def best_time_shift(df, lat, lon, shifts=(-60, -30, 0, 30, 60)):
    """Minutes to add to the POWER timestamp so that GHI best matches cos(zenith) (checks the hour convention)."""
    sub = df.loc[df.index.year == df.index.year.max()]
    out = {}
    for m in shifts:
        z, _ = solar_position(sub.index + pd.Timedelta(minutes=m), lat, lon)
        out[m] = np.corrcoef(np.clip(np.cos(np.radians(z)), 0, None), sub.ALLSKY_SFC_SW_DWN.fillna(0))[0, 1]
    return max(out, key=out.get), out


# ---------------------------------------------------------------- PV (isotropic sky, NOCT cell temperature, PVWatts-type DC)
def pv_output_per_kwp(df, lat, lon, tilt=None, az=None, albedo=0.25, gamma=-0.0037, noct=45.0, losses=0.14,
                      inv_eff=0.96, shift_min=30):
    """AC-side PV output (kW per kWp) each hour. Tilt defaults to |lat|, azimuth to the equator. Hourly irradiance is
    treated as the mean over the hour, evaluated at timestamp + shift_min."""
    tilt = abs(lat) if tilt is None else tilt
    az = (180.0 if lat >= 0 else 0.0) if az is None else az
    zen, saz = solar_position(df.index + pd.Timedelta(minutes=shift_min), lat, lon)
    z, b, sa, pa = map(np.radians, (zen, tilt, saz, az))
    cos_aoi = np.cos(z) * np.cos(b) + np.sin(z) * np.sin(b) * np.cos(sa - pa)
    ghi, dni, dhi = (df[c].fillna(0).clip(lower=0).values for c in ("ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_SW_DNI", "ALLSKY_SFC_SW_DIFF"))
    up = zen < 90
    poa = (dni * np.clip(cos_aoi, 0, None) * up + dhi * (1 + np.cos(b)) / 2 + ghi * albedo * (1 - np.cos(b)) / 2)
    tcell = df["T2M"].values + poa / 800.0 * (noct - 20.0)
    p = poa / 1000.0 * (1 + gamma * (tcell - 25.0)) * (1 - losses) * inv_eff
    return pd.Series(np.clip(p, 0, None), index=df.index, name="pv_kw_per_kwp"), pd.Series(poa, index=df.index)


# ---------------------------------------------------------------- wind (shear from 10/50 m, density correction, generic curve)
def wind_output_per_kw(df, hub=30.0, v_in=3.0, v_r=12.0, v_out=25.0, alpha_bounds=(0.05, 0.40), losses=0.10):
    """Output (kW per kW rated) of a generic small turbine. Hourly shear exponent from the 10 m and 50 m POWER winds,
    hub-height speed by the power law, IEC density correction v*(rho/1.225)^(1/3), cubic curve between cut-in and
    rated speed, availability/electrical losses."""
    v10, v50 = df.WS10M.clip(lower=0.1).values, df.WS50M.clip(lower=0.1).values
    alpha = np.clip(np.log(v50 / v10) / np.log(50 / 10), *alpha_bounds)
    v = v50 * (hub / 50.0) ** alpha
    rho = df.PS.values * 1000 / (287.05 * (df.T2M.values + 273.15))
    v = v * (rho / 1.225) ** (1 / 3)
    p = np.where((v >= v_in) & (v < v_r), (v ** 3 - v_in ** 3) / (v_r ** 3 - v_in ** 3), 0.0)
    p = np.where((v >= v_r) & (v < v_out), 1.0, p) * (1 - losses)
    return pd.Series(p, index=df.index, name="wind_kw_per_kw"), pd.Series(v, index=df.index), pd.Series(alpha, index=df.index)


# ---------------------------------------------------------------- FAO-56 maize demand for sizing (single Kc)
def maize_kc(n_days, stages=(25, 40, 45, 30), kc=(0.30, 1.20, 0.35)):
    ini, dev, mid, late = stages
    k = np.empty(n_days)
    for d in range(n_days):
        if d < ini:
            k[d] = kc[0]
        elif d < ini + dev:
            k[d] = kc[0] + (kc[1] - kc[0]) * (d - ini) / dev
        elif d < ini + dev + mid:
            k[d] = kc[1]
        elif d < sum(stages):
            k[d] = kc[1] + (kc[2] - kc[1]) * (d - ini - dev - mid) / late
        else:
            k[d] = 0.0
    return k


def irrigation_demand(daily_et0, daily_rain, sow_mmdd, area_ha=5.0, eff=0.90, stages=(25, 40, 45, 30)):
    """Daily gross irrigation demand (m3) for maize sown each year on sow_mmdd: (Kc*ET0 - Peff)+ / efficiency.
    Peff = 0.8*(P - 5 mm)+ (simple daily effective-rain rule)."""
    out = pd.Series(0.0, index=daily_et0.index)
    L = sum(stages)
    for y in sorted(set(daily_et0.index.year)):
        s = pd.Timestamp(f"{y}-{sow_mmdd}")
        idx = pd.date_range(s, periods=L, freq="D")
        idx = idx[idx.isin(daily_et0.index)]
        if len(idx) == 0:
            continue
        k = maize_kc(L, stages)[:len(idx)]
        peff = 0.8 * np.clip(daily_rain.loc[idx].values - 5.0, 0, None)
        nir = np.clip(k * daily_et0.loc[idx].values - peff, 0, None)
        out.loc[idx] = nir / eff * 10.0 * area_ha           # mm over area_ha -> m3
    return out


# ---------------------------------------------------------------- year-round farm loads (appliance-based, stochastic, seeded)
FARM_LOAD = {   # ASSUMED appliance list (author query); RAMP-style: rated power, daily use window (local hours), daily minutes
    "lights":   dict(kw=0.06, window=(18, 23), minutes=240),
    "tv":       dict(kw=0.08, window=(19, 23), minutes=150),
    "workshop": dict(kw=2.00, window=(9, 17), minutes=120),
    "fridge_kw": 0.15, "fridge_duty": 0.40,
    "fans_kw": 0.15, "fans_tmax_c": 30.0,                      # fans run 11:00-18:00 local on days with Tmax above this
    "cold_ua_kw_per_k": 0.12, "cold_set_c": 2.0, "cold_cop": 2.0, "cold_base_kw": 0.30,
}


def farm_load(t2m_utc, lon, seed=0, cfg=FARM_LOAD):
    """Hourly year-round farm load (kW) from hourly air temperature (UTC index): household appliances in randomised daily
    windows (seeded), fridge duty cycle, fans on hot days, and a cold room whose compressor power follows
    UA*(T - Tset)/COP + base. Local time = UTC + lon/15 h (solar time, rounded)."""
    rng = np.random.default_rng(seed)
    idx = t2m_utc.index
    loc_hr = ((idx.hour + int(round(lon / 15))) % 24).values
    local_day = (idx + pd.Timedelta(hours=int(round(lon / 15)))).normalize()
    days, d_idx = np.unique(local_day.values, return_inverse=True)
    p = np.full(len(idx), cfg["fridge_kw"] * cfg["fridge_duty"])
    for name in ("lights", "tv", "workshop"):
        a = cfg[name]
        w0, w1 = a["window"]
        hours = max(1, int(round(a["minutes"] / 60)))
        starts = rng.integers(w0, max(w0 + 1, w1 - hours + 1), size=len(days))         # random start hour each day
        on = (loc_hr >= starts[d_idx]) & (loc_hr < starts[d_idx] + hours)
        p += on * a["kw"] * rng.uniform(0.8, 1.2, size=len(days))[d_idx]
    t = t2m_utc.values
    tmax_day = pd.Series(t).groupby(d_idx).transform("max").values
    p += ((tmax_day > cfg["fans_tmax_c"]) & (loc_hr >= 11) & (loc_hr < 18)) * cfg["fans_kw"]
    p += np.clip(t - cfg["cold_set_c"], 0, None) * cfg["cold_ua_kw_per_k"] / cfg["cold_cop"] + cfg["cold_base_kw"]
    return pd.Series(p, index=idx, name="farm_load_kw")


# ---------------------------------------------------------------- hourly system simulation (rule-based baseline dispatch)
def simulate(pv_kw, wind_kw, demand_m3_day, head_m, pv_kwp, wind_kwr, batt_kwh, tank_m3, pump_kw, load_kw=None,
             eta_pump=0.55, soc_min=0.10, soc_max=0.90, eta_ch=0.96, eta_dis=0.96, c_rate=0.5, pump_min_frac=0.3,
             tank0=0.5, pump_batt_reserve=0.5):
    """Rule-based dispatch. Each hour: renewables serve the farm load first (critical; battery covers any shortfall),
    then the pump (if the tank is not full), then charge the battery, then are curtailed. The battery may top up the
    pump to its minimum power only while the tank is below half or the day's demand is outstanding, and only above
    pump_batt_reserve of capacity (night reserve for the farm load). Irrigation is drawn from the tank at the first hour
    of each UTC day and from water pumped later that day. Returns hourly frame and summary dict.
    pv_kw / wind_kw: per-unit output series (hourly, aligned); demand: daily series (m3); load_kw: hourly kW or None."""
    n = len(pv_kw)
    gen = (pv_kw.values * pv_kwp + wind_kw.values * wind_kwr).tolist()
    load = (load_kw.reindex(pv_kw.index).fillna(0.0).values if load_kw is not None else np.zeros(n)).tolist()
    reserve = (soc_min + pump_batt_reserve * (soc_max - soc_min)) * batt_kwh
    unmet_e = 0.0
    kwh_per_m3 = RHO_W * G * head_m / eta_pump / 3.6e6
    days = pv_kw.index.normalize().tz_localize(None)
    uniq, day_idx = np.unique(days.values, return_inverse=True)
    dem_arr = demand_m3_day.reindex(pd.DatetimeIndex(uniq)).fillna(0.0).values.tolist()
    new_day = np.r_[True, day_idx[1:] != day_idx[:-1]].tolist()
    day_idx = day_idx.tolist()
    soc = soc_min * batt_kwh + 0.5 * (soc_max - soc_min) * batt_kwh
    tank = tank0 * tank_m3
    p_ch_max = c_rate * batt_kwh
    rec = np.zeros((n, 8))
    delivered = unmet = 0.0
    owed = 0.0
    for i in range(n):
        if new_day[i]:                                       # new day: settle yesterday, draw today's irrigation
            unmet += owed
            owed = dem_arr[day_idx[i]]
            take = min(owed, tank); tank -= take; owed -= take; delivered += take
        g = gen[i]
        L = load[i]
        served = min(g, L); g -= served; short = L - served   # farm load first
        if short > 0:
            dis = min(short, max(soc - soc_min * batt_kwh, 0.0) * eta_dis, p_ch_max)
            soc -= dis / eta_dis; short -= dis; served += dis
        unmet_e += short
        room = max(tank_m3 - tank, 0.0)
        p_pump = 0.0
        if room > 1e-9 and pump_kw > 0:
            p_need = min(pump_kw, room * kwh_per_m3)
            p_pump = min(g, p_need)
            if p_pump < pump_min_frac * pump_kw:            # VFD minimum: try to top up from battery when water is needed
                want = (tank < 0.5 * tank_m3) or owed > 0
                top = min(pump_min_frac * pump_kw - p_pump, max(soc - reserve, 0) * eta_dis, p_ch_max) if want else 0.0
                if p_pump + top >= pump_min_frac * pump_kw - 1e-9:
                    soc -= top / eta_dis; p_pump += top
                else:
                    p_pump = 0.0
        surplus = g - min(p_pump, g)
        ch = min(surplus, p_ch_max, max(soc_max * batt_kwh - soc, 0) / eta_ch)
        soc += ch * eta_ch
        curtail = surplus - ch
        water = p_pump / kwh_per_m3
        tank += water
        if owed > 0:                                         # water arriving later in the day still serves the demand
            take = min(owed, tank); tank -= take; owed -= take; delivered += take
        rec[i] = (gen[i], p_pump, ch, curtail, soc, tank, served, L)
    if owed > 0:
        unmet += owed
    h = pd.DataFrame(rec, index=pv_kw.index,
                     columns=["gen_kw", "pump_kw", "charge_kw", "curtail_kw", "soc_kwh", "tank_m3", "load_served_kw", "load_kw"])
    total_dem = float(sum(dem_arr)); total_load = float(sum(load))
    summ = dict(demand_m3=total_dem, delivered_m3=delivered, unmet_m3=unmet,
                wlpsp=unmet / total_dem if total_dem > 0 else 0.0, gen_kwh=h.gen_kw.sum(), pump_kwh=h.pump_kw.sum(),
                load_kwh=total_load, load_unmet_kwh=unmet_e, elpsp=unmet_e / total_load if total_load > 0 else 0.0,
                curtail_kwh=h.curtail_kw.sum(), batt_throughput_kwh=h.charge_kw.sum(), kwh_per_m3=kwh_per_m3)
    return h, summ


# ---------------------------------------------------------------- economics
COSTS = {   # base case, USD (2022-2025 dollar years, not re-indexed); sources and status per item in COST_SOURCES
    "pv_per_kwp": 691, "wind_per_kw": 6680, "batt_per_kwh": 350, "tank_per_m3": 100, "pump_per_kw": 900,
    "om_frac": {"pv": 19 / 691, "wind": 39 / 6680, "batt": 0.025, "tank": 0.01, "pump": 0.03},
    "life": {"pv": 25, "wind": 20, "batt": 15, "tank": 25, "pump": 10},
    "discount": 0.075, "project_years": 25,
    "diesel_usd_per_l": 1.00, "genset_kwh_per_l": 3.0, "genset_usd_per_kw": 400, "co2_kg_per_l": 2.68,
}

# 2024 average retail diesel (pump price incl. taxes) converted at 2024 average exchange rates; see COST_SOURCES
DIESEL_USD_PER_L = {"bushland_tx": 0.99, "jodhpur_in": 1.07, "souss_ma": 1.26, "azraq_jo": 1.00}

# capital-cost bounds for the sensitivity analysis (O&M scales with capital through om_frac)
COST_SCENARIOS = {
    "base": {},
    "capex_low": {"pv_per_kwp": 525, "wind_per_kw": 3775, "batt_per_kwh": 192},
    "capex_high": {"pv_per_kwp": 1780, "wind_per_kw": 9630, "batt_per_kwh": 466},
    "discount_4%": {"discount": 0.04},
    "discount_12%": {"discount": 0.12},
    "battery_life_10y": {"life": {**COSTS["life"], "batt": 10}},
    "diesel_-30%": {"diesel_scale": 0.7},
    "diesel_+30%": {"diesel_scale": 1.3},
    "tank_low": {"tank_per_m3": 85},
    "tank_high": {"tank_per_m3": 170},
    "tank_elevated_steel": {"tank_per_m3": 620, "tank_fixed_usd": 5200, "life": {**COSTS["life"], "tank": 20}},
    "pump_low": {"pump_per_kw": 125},
    "pump_high": {"pump_per_kw": 1180},
}

COST_SOURCES = {   # value basis and verification status, used for the manuscript cost table
    "pv_per_kwp": "691: IRENA RPGC in 2024 (2025), global weighted-average total installed cost of utility-scale PV, 2024 USD; "
                  "low 525: same report, India; high 1780: NREL ATB 2024b commercial PV (200 kWdc), 2023",
    "wind_per_kw": "6680: PNNL Distributed Wind Energy Technology Data Update 2025 ed., average installed cost of new small wind "
                   "(<=100 kW) projects in 2024, n=30; low 3775: PNNL 2026 ed., midsize/large turbines 2016-2022 average; "
                   "high 9630: PNNL 2026 ed., small wind 2025, n=5",
    "batt_per_kwh": "350: assumed, inside the sourced range; low 192: IRENA RPGC in 2024, utility-scale BESS 2024; "
                    "high 466: NREL ATB 2024b residential battery pack 283 + inverter 183 USD/kWh (2022 USD, hardware only)",
    "om_frac": "PV 19 USD/kW-yr: NREL ATB 2024b commercial PV (2023); wind 39 USD/kW-yr: NREL ATB 2024b distributed wind; "
               "battery 2.5%/yr of capital: NREL ATB 2024b residential/commercial battery; tank 1%, pump 3%: assumed",
    "life": "PV 25 yr: IRENA RPGC Table A1; wind 20 yr: IEC 61400-2 minimum design life for small turbines; battery 15 yr: "
            "NREL ATB 2024b; tank 25 yr, pump 10 yr: assumed",
    "discount": "7.5% real: IRENA RPGC Table A1 rest-of-world WACC since 2020; sensitivity 4% (Asia PV) and 12% (Africa PV): "
                "IRENA RPGC in 2024, Fig. 1.13",
    "tank_per_m3": "100: inside 85-170 USD/m3 for ground-level steel-frame tank kits of 95-45 m3 (UNICEF Supply Catalogue, "
                   "indicative prices); sensitivity low 85, high 170, and an elevated steel tank 620 USD/m3 + 5,200 USD, "
                   "life 20 yr (Soenen et al. 2021, Energies 14:2483, Table 2)",
    "pump_per_kw": "900: installed submersible pump with drive, inside the range from about 125 USD/kW (pump only, derived from "
                   "MNRE benchmark costs 2019-20: standalone solar pump minus pump solarisation) to about 1,180 USD/kW "
                   "(US NRCS FY2024 practice 533 electric pumping plant, payment rate at 75 % of cost); sensitivity 125 / 1180",
    "diesel_usd_per_l": "USA 0.99: EIA on-highway diesel 2024 average 3.76 USD/gal (FRED GASDESA); India 1.07: 89.9 INR/L, "
                        "PPAC Delhi daily RSP 2024 (mean 88.03) with Delhi taxes removed and Rajasthan VAT 17.30% + 1.75 INR/L "
                        "cess applied, at 83.7 INR/USD; Morocco 1.26: 12.51 MAD/L, mean of 24 fortnightly national average "
                        "pump prices TTC 2024 (Conseil de la Concurrence quarterly gasoil reportings) at 9.95 MAD/USD; "
                        "Jordan 1.00: 705.8 fils/L, mean of the 12 monthly pricing-committee prices 2024 (MEMR) at 1.41 USD/JOD",
    "genset_usd_per_kw": "400: within 260-510 USD/kVA (UNICEF supply catalogue, 20-30 kVA sets) and about 360 USD/kW "
                         "(World Bank mini-grid planning example)",
    "genset_kwh_per_l": "3.0: assumed full-load efficiency at all loads (favours diesel)",
    "co2_kg_per_l": "2.68: IPCC 2006 default diesel emission factor (74.1 t CO2/TJ) at about 36 MJ/L",
}


def site_costs(site_id, scenario="base"):
    """Base costs with the site's diesel price and one sensitivity scenario applied."""
    ch = dict(COST_SCENARIOS[scenario])
    scale = ch.pop("diesel_scale", 1.0)
    return {**COSTS, "diesel_usd_per_l": DIESEL_USD_PER_L[site_id] * scale, **ch}


def crf(i, n):
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


def diesel_reference(kwh_per_year, pump_kw, peak_load_kw=0.0, c=COSTS):
    """NPC and annual CO2 of a diesel genset supplying the same energy (pumping + farm load). Constant fuel efficiency
    (genset_kwh_per_l) is assumed, which favours diesel at part load, so renewable savings are not overstated.
    Genset rated 1.25 x (pump + peak farm load), life 10 yr; pump capital included."""
    i, N = c["discount"], c["project_years"]
    litres = kwh_per_year / c["genset_kwh_per_l"]
    ann = (1 - (1 + i) ** -N) / i
    cap = (pump_kw + peak_load_kw) * 1.25 * c["genset_usd_per_kw"] + pump_kw * c["pump_per_kw"]
    npc_d = cap + sum(cap / (1 + i) ** t for t in range(10, N, 10)) + litres * c["diesel_usd_per_l"] * ann + cap * 0.05 * ann
    return dict(npc=npc_d, litres_per_year=litres, co2_t_per_year=litres * c["co2_kg_per_l"] / 1000)


GRID_DEFAULT = {"pv": [0, 1, 1.5, 2, 3, 4], "wind": [0, 0.5, 1, 2], "batt": [0, 1, 2, 4], "tank": [0.03, 0.0625, 0.125, 0.25, 0.5]}


def size_system(pv, wind, demand, head_m, train, test, pump_kw, peak_m3_day, load_kw=None, grid=None,
                wlpsp_max=0.05, elpsp_max=0.02, c=COSTS):
    """Grid search over PV, wind, battery and tank sizes (multiples of pump power / peak daily demand).
    Feasible = water loss-of-supply probability <= wlpsp_max AND farm-load energy loss-of-supply probability
    <= elpsp_max on the training years. Returns all configurations and the least-NPC design of any mix ('least_cost'),
    and the least-NPC 'hybrid' (PV and wind both > 0), 'pv_only' and 'wind_only' designs, each re-simulated on the test
    years (out-of-sample reliability)."""
    grid = grid or GRID_DEFAULT
    tr_years = len(set(pv.loc[train].index.year))
    ld_tr = load_kw.loc[train] if load_kw is not None else None
    peak_load = float(load_kw.max()) if load_kw is not None else 0.0
    rows = []
    for gp in grid["pv"]:
        for gw in grid["wind"]:
            if gp == 0 and gw == 0:
                continue
            for gb in grid["batt"]:
                for gt in grid["tank"]:
                    cfg = dict(pv_kwp=gp * pump_kw, wind_kwr=gw * pump_kw, batt_kwh=gb * pump_kw, tank_m3=gt * peak_m3_day)
                    _, s = simulate(pv.loc[train], wind.loc[train], demand.loc[train], head_m, pump_kw=pump_kw, load_kw=ld_tr, **cfg)
                    cost = npc(**cfg, pump_kw=pump_kw, c=c)
                    ann = cost * crf(c["discount"], c["project_years"])
                    used = s["pump_kwh"] + s["load_kwh"] - s["load_unmet_kwh"]
                    rows.append({**cfg, "npc_usd": cost, "wlpsp_train": s["wlpsp"], "elpsp_train": s["elpsp"],
                                 "curtail_frac": s["curtail_kwh"] / s["gen_kwh"], "lcoe_usd_kwh": ann / (used / tr_years),
                                 "pump_kwh_yr": s["pump_kwh"] / tr_years, "load_kwh_yr": s["load_kwh"] / tr_years,
                                 "batt_cycles_yr": (s["batt_throughput_kwh"] / cfg["batt_kwh"] / tr_years) if cfg["batt_kwh"] > 0 else 0.0})
    allc = pd.DataFrame(rows)
    feas = allc[(allc.wlpsp_train <= wlpsp_max) & (allc.elpsp_train <= elpsp_max)]
    pick = {}
    for name, sub in (("least_cost", feas), ("hybrid", feas[(feas.pv_kwp > 0) & (feas.wind_kwr > 0)]),
                      ("pv_only", feas[feas.wind_kwr == 0]), ("wind_only", feas[feas.pv_kwp == 0])):
        if len(sub):
            best = sub.sort_values("npc_usd").iloc[0].to_dict()
            _, st = simulate(pv.loc[test], wind.loc[test], demand.loc[test], head_m, pump_kw=pump_kw,
                             load_kw=load_kw.loc[test] if load_kw is not None else None,
                             **{k: best[k] for k in ("pv_kwp", "wind_kwr", "batt_kwh", "tank_m3")})
            best["wlpsp_test"], best["elpsp_test"] = st["wlpsp"], st["elpsp"]
            best["diesel"] = diesel_reference(best["pump_kwh_yr"] + best["load_kwh_yr"], pump_kw, peak_load, c)
            pick[name] = best
    return allc, pick


def npc(pv_kwp, wind_kwr, batt_kwh, tank_m3, pump_kw, c=COSTS):
    """Net present cost over the project life: capital + replacements (salvage ignored) + O&M."""
    i, N = c["discount"], c["project_years"]
    caps = {"pv": pv_kwp * c["pv_per_kwp"], "wind": wind_kwr * c["wind_per_kw"], "batt": batt_kwh * c["batt_per_kwh"],
            "tank": tank_m3 * c["tank_per_m3"] + (c.get("tank_fixed_usd", 0.0) if tank_m3 > 0 else 0.0),
            "pump": pump_kw * c["pump_per_kw"]}
    total = 0.0
    pv_annuity = (1 - (1 + i) ** -N) / i
    for k, cap in caps.items():
        L = c["life"][k]
        total += cap + sum(cap / (1 + i) ** t for t in range(L, N, L)) + cap * c["om_frac"][k] * pv_annuity
    return total
