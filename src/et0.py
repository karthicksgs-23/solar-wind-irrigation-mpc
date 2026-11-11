"""FAO-56 Penman-Monteith daily reference evapotranspiration (Allen et al. 1998, eqs. 7-8, 11-13, 17, 21-25, 28-39, 6).

Inputs per day: Tmax, Tmin (deg C), mean RH (%), u2 wind at 2 m (m/s), Rs (MJ m-2 d-1), pressure (kPa) or elevation (m),
latitude (deg). Missing pressure is estimated from elevation (eq. 7).
"""
import numpy as np
import pandas as pd


def es(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def et0_fao56(date, tmax, tmin, rh_mean, u2, rs, lat_deg, elev_m, p_kpa=None):
    date = pd.to_datetime(pd.Series(date)).reset_index(drop=True)
    tmax, tmin, rh, u2, rs = (np.asarray(x, float) for x in (tmax, tmin, rh_mean, u2, rs))
    tmean = (tmax + tmin) / 2
    p = np.asarray(p_kpa, float) if p_kpa is not None else np.full_like(tmean, 101.3 * ((293 - 0.0065 * elev_m) / 293) ** 5.26)
    gamma = 0.000665 * p
    delta = 4098 * es(tmean) / (tmean + 237.3) ** 2
    e_s = (es(tmax) + es(tmin)) / 2
    e_a = rh / 100 * e_s                                   # eq. 19 (mean RH)
    j = date.dt.dayofyear.values
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365)
    dec = 0.409 * np.sin(2 * np.pi * j / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    rso = (0.75 + 2e-5 * elev_m) * ra
    rns = (1 - 0.23) * rs
    rel = np.clip(rs / rso, 0.3, 1.0)
    rnl = 4.903e-9 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4) / 2 * (0.34 - 0.14 * np.sqrt(e_a)) * (1.35 * rel - 0.35)
    rn = rns - rnl
    et0 = (0.408 * delta * rn + gamma * 900 / (tmean + 273) * u2 * (e_s - e_a)) / (delta + gamma * (1 + 0.34 * u2))
    return np.maximum(et0, 0.0)
