"""Day-ahead forecasts for the controllers: hourly PV (kW/kWp), wind (kW/kW) and farm load (kW) for the next two local
days, and daily ET0 and rain for the next two days.

Issue time: local solar midnight. Lead day 1 = the coming day, lead day 2 = the day after. Features use only data
available at issue time (values up to and including the previous local day). Model: scikit-learn
HistGradientBoostingRegressor per variable and lead day; baselines: persistence (same hour of the last observed day)
and climatology (training-period mean for that day of year and hour).

Leakage control: forecasts for training years are produced by blocked cross-validation (models never see the block
they forecast); forecasts for later years come from a model fitted on all training years.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def _local(index_utc, lon):
    """Local solar-time index (UTC + lon/15 h, rounded to the hour) for daily grouping."""
    return index_utc.tz_localize(None) + pd.Timedelta(hours=int(round(lon / 15)))


def hourly_frame(series_utc, lon):
    """Reshape an hourly UTC series into a local-day x 24-hour matrix (rows = local days)."""
    s = pd.Series(series_utc.values, index=_local(series_utc.index, lon))
    df = pd.DataFrame({"v": s.values, "day": s.index.normalize(), "hr": s.index.hour})
    m = df.pivot_table(index="day", columns="hr", values="v", aggfunc="mean")
    return m.reindex(columns=range(24))


def _features_hourly(mat, lead):
    """Rows = (target day, hour). Features from days <= target - lead: same-hour value 1, 2, 7 days before issue,
    previous-day mean/max, hour, day-of-year harmonics."""
    days = mat.index
    rows = []
    prev = mat.shift(lead)               # day D-lead (last observed day relative to the issue for lead 'lead')
    prev2 = mat.shift(lead + 1)
    prev7 = mat.shift(lead + 6)
    dmean, dmax = prev.mean(axis=1), prev.max(axis=1)
    doy = days.dayofyear.values
    for h in range(24):
        rows.append(pd.DataFrame({"day": days, "hr": h, "y": mat[h].values, "lag_d": prev[h].values, "lag_d2": prev2[h].values,
                                  "lag_d7": prev7[h].values, "dmean": dmean.values, "dmax": dmax.values,
                                  "sin_doy": np.sin(2 * np.pi * doy / 365.25), "cos_doy": np.cos(2 * np.pi * doy / 365.25),
                                  "hr_sin": np.sin(2 * np.pi * h / 24), "hr_cos": np.cos(2 * np.pi * h / 24)}))
    return pd.concat(rows, ignore_index=True)


def _features_daily(series, lead):
    d = pd.DataFrame({"y": series})
    for k in range(lead, lead + 3):
        d[f"lag{k}"] = series.shift(k)
    d["roll7"] = series.shift(lead).rolling(7, min_periods=1).mean()
    doy = series.index.dayofyear.values
    d["sin_doy"], d["cos_doy"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return d


def _fit_predict(X_tr, y_tr, X_te, seed=0, nonneg=True):
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_leaf_nodes=31, random_state=seed)
    m.fit(X_tr, y_tr)
    p = m.predict(X_te)
    return np.clip(p, 0, None) if nonneg else p


def _blocked_predict(feat, train_years, other_years, n_blocks=4, seed=0):
    """Out-of-sample predictions: blocked CV inside the training years; one full fit for the other years."""
    cols = [c for c in feat.columns if c not in ("day", "hr", "y")]
    feat = feat.dropna(subset=cols + ["y"]).copy()
    yrs = feat["day"].dt.year
    pred = pd.Series(np.nan, index=feat.index)
    blocks = np.array_split(np.array(sorted(train_years)), n_blocks)
    for b in blocks:
        tr = yrs.isin(train_years) & ~yrs.isin(b)
        te = yrs.isin(b)
        pred[te] = _fit_predict(feat.loc[tr, cols], feat.loc[tr, "y"], feat.loc[te, cols], seed)
    tr = yrs.isin(train_years); te = yrs.isin(other_years)
    if te.any():
        pred[te] = _fit_predict(feat.loc[tr, cols], feat.loc[tr, "y"], feat.loc[te, cols], seed)
    feat["pred"] = pred
    return feat


def forecast_hourly(series_utc, lon, train_years, other_years, leads=(1, 2), seed=0):
    """Day-ahead hourly forecasts (local day x hour) for each lead day, with persistence and climatology baselines.
    Returns a long DataFrame: day, hr, lead, y, pred, persist, clim."""
    mat = hourly_frame(series_utc, lon)
    clim = mat[mat.index.year.isin(train_years)].groupby(mat.index[mat.index.year.isin(train_years)].dayofyear).mean()
    out = []
    for lead in leads:
        f = _blocked_predict(_features_hourly(mat, lead), train_years, other_years, seed=seed)
        f["lead"] = lead
        f["persist"] = f["lag_d"]
        f["clim"] = [clim.loc[min(d.dayofyear, clim.index.max()), h] for d, h in zip(f["day"], f["hr"])]
        out.append(f[["day", "hr", "lead", "y", "pred", "persist", "clim"]])
    return pd.concat(out, ignore_index=True)


def forecast_daily(series, train_years, other_years, leads=(1, 2), seed=0, nonneg=True):
    """Day-ahead daily forecasts (e.g. ET0, rain) for each lead day with persistence and climatology baselines."""
    clim = series[series.index.year.isin(train_years)].groupby(series.index[series.index.year.isin(train_years)].dayofyear).mean()
    out = []
    for lead in leads:
        d = _features_daily(series, lead)
        d["day"] = d.index
        f = _blocked_predict(d.reset_index(drop=True), train_years, other_years, seed=seed)
        f["lead"] = lead
        f["persist"] = f[f"lag{lead}"]
        f["clim"] = [clim.loc[min(x.dayofyear, clim.index.max())] for x in f["day"]]
        out.append(f[["day", "lead", "y", "pred", "persist", "clim"]])
    return pd.concat(out, ignore_index=True)


def skill_table(fc, years, by=("lead",)):
    """nMAE (MAE / mean observed) of the model, persistence and climatology, and skill = 1 - nMAE_model / nMAE_persist."""
    f = fc[fc["day"].dt.year.isin(years)].dropna(subset=["y", "pred", "persist", "clim"])
    rows = []
    for key, g in f.groupby(list(by)):
        mu = g["y"].abs().mean()
        r = {k: v for k, v in zip(by, key if isinstance(key, tuple) else (key,))}
        for m in ("pred", "persist", "clim"):
            r[f"nmae_{m}"] = (g[m] - g["y"]).abs().mean() / mu if mu > 0 else np.nan
        r["skill_vs_persist"] = 1 - r["nmae_pred"] / r["nmae_persist"]
        r["skill_vs_clim"] = 1 - r["nmae_pred"] / r["nmae_clim"]
        rows.append(r)
    return pd.DataFrame(rows)
