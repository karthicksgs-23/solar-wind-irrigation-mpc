"""Post-hoc analysis of the nb03 v3 Colab results (outputs/nb03v3_colab) without re-running any controller.

1. Corrected reference yield: Ymax(site, year) = max of the AquaCrop yields of the 'full' calibration schedule and of
   every evaluated test schedule. Controller differences are unchanged; absolute yield-loss values become >= 0.
2. Cost split: unserved-load penalty (COST['shed'] = 50 USD/kWh in the run) vs. the remaining operating cost.
3. Ex-post sensitivity: penalty weight {0, 5, 20, 50} USD/kWh x maize price {165.5, 237.4, 318.8} USD/t
   (World Bank Pink Sheet, US maize, 2020-2024 min / mean / max). Controllers were tuned and trained with 50 USD/kWh
   and 200 USD/t, so this re-scores fixed schedules; it does not re-optimise them.
4. Paired year-bootstrap (2,000 resamples, seed 42) for the key contrasts under each scenario; family ranking per site.
Writes outputs/nb03v3_colab/posthoc/*.csv and SUMMARY.md."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
IN = Path(__file__).resolve().parents[1] / "outputs" / "nb03v3_colab"
OUT = IN / "posthoc"; OUT.mkdir(exist_ok=True)
AREA_HA, SHED_W_RUN, PRICE_RUN = 5.0, 50.0, 200.0
PINK = {2020: 165.5, 2021: 259.5, 2022: 318.8, 2023: 252.7, 2024: 190.6}      # USD/t, World Bank CMO annual, Oct 2026 edition
PRICES = {"low": min(PINK.values()), "base": round(float(np.mean(list(PINK.values()))), 1), "high": max(PINK.values())}
VOLL = [0.0, 5.0, 20.0, 50.0]
FAM = ["rule", "two_stage", "mpc_default", "mpc_uncal", "mpc", "hybrid", "drl", "mpc_perfect"]
CONTRASTS = [("hybrid", "mpc"), ("drl", "mpc"), ("mpc", "rule"), ("mpc", "two_stage"), ("mpc", "mpc_uncal"),
             ("mpc", "mpc_default"), ("mpc_perfect", "mpc"), ("two_stage", "rule"), ("hybrid", "drl")]

TE = pd.read_csv(IN / "evaluation_test_with_aquacrop.csv")
TE["family"] = TE.controller.str.replace(r"_s\d+$", "", regex=True)
CAL = pd.read_csv(IN / "crop_calibration_runs.csv")

# 1. corrected Ymax
y_cal = CAL[CAL.schedule == "full"].groupby(["site", "year"]).yield_t_ha.max()
y_ctl = TE.groupby(["site", "year"]).yield_t_ha.max()
ymax = pd.concat([y_cal, y_ctl], axis=1).max(axis=1).rename("ymax_corr")
TE = TE.join(ymax, on=["site", "year"])
TE["ymax_full_schedule"] = [y_cal[(s, y)] for s, y in zip(TE.site, TE.year)]
TE["yield_gap_t_ha"] = TE.ymax_corr - TE.yield_t_ha
assert (TE.yield_gap_t_ha >= -1e-9).all()

# 2. cost split (energy_cost = shed penalty + water + battery cycling terms)
TE["penalty_run"] = SHED_W_RUN * TE.shed_kwh
TE["opex_ex_penalty"] = TE.energy_cost - TE.penalty_run


def total(df, voll, price):
    return df.opex_ex_penalty + voll * df.shed_kwh + price * AREA_HA * df.yield_gap_t_ha


for v in VOLL:
    for k, p in PRICES.items():
        TE[f"tc_v{int(v)}_{k}"] = total(TE, v, p)
TE.to_csv(OUT / "test_rows_rescored.csv", index=False)
SCEN = [c for c in TE.columns if c.startswith("tc_v")]

# year-level family means (agents averaged over seeds)
Y = TE.groupby(["site", "family", "year"])[SCEN + ["shed_kwh", "irrigation_mm", "yield_t_ha", "wp_kg_m3", "opex_ex_penalty",
                                                    "penalty_run", "yield_gap_t_ha"]].mean().reset_index()

# physical and cost tables (test means; agents: mean and sd over seeds)
phys = TE.groupby(["site", "family"])[["shed_kwh", "elpsp", "irrigation_mm", "yield_t_ha", "wp_kg_m3", "yield_gap_t_ha",
                                       "curtail_kwh", "batt_throughput_kwh", "opex_ex_penalty", "penalty_run",
                                       "tc_v50_base", "tc_v0_base"]].mean()
phys = phys.reindex(FAM, level=1).round(3); phys.to_csv(OUT / "table_physical_and_cost.csv")
seed_sd = TE[TE.family.isin(["hybrid", "drl"])].groupby(["site", "family", "controller"])[["tc_v50_base", "tc_v0_base", "irrigation_mm", "yield_t_ha"]].mean()
seed_sd = seed_sd.groupby(["site", "family"]).agg(["mean", "std", "min", "max"]).round(2); seed_sd.to_csv(OUT / "table_seed_spread.csv")

# 3./4. paired bootstrap for every scenario
rng = np.random.default_rng(42)
rows = []
for col in SCEN + ["shed_kwh", "irrigation_mm", "yield_t_ha"]:
    for sid in Y.site.unique():
        for a, b in CONTRASTS:
            A = Y[(Y.site == sid) & (Y.family == a)].set_index("year")[col]
            B = Y[(Y.site == sid) & (Y.family == b)].set_index("year")[col]
            d = (A - B).dropna().values
            boots = rng.choice(d, (2000, len(d))).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            base = B.mean()
            rows.append(dict(metric=col, site=sid, contrast=f"{a} - {b}", mean=d.mean(), lo95=lo, hi95=hi,
                             pct_of_b=100 * d.mean() / base if base else np.nan, sig=bool(lo > 0 or hi < 0)))
CMP = pd.DataFrame(rows); CMP.to_csv(OUT / "paired_bootstrap_all_scenarios.csv", index=False)

# ranking robustness: family ranks per site under each scenario
rank = []
for col in SCEN:
    m = Y.groupby(["site", "family"])[col].mean().unstack()
    for sid, r in m.iterrows():
        order = r.reindex([f for f in FAM if f != "mpc_perfect"]).sort_values()
        rank.append(dict(scenario=col, site=sid, order=" < ".join(order.index), best=order.index[0],
                         mpc_rank=list(order.index).index("mpc") + 1, hybrid_rank=list(order.index).index("hybrid") + 1))
RK = pd.DataFrame(rank); RK.to_csv(OUT / "ranking_by_scenario.csv", index=False)

# summary
md = lambda df, **kw: "```\n" + df.to_string(**kw) + "\n```"
pd.set_option("display.width", 220)
lines = ["# nb03 v3 post-hoc analysis", "",
         f"Maize price scenarios (USD/t, World Bank Pink Sheet US maize 2020-2024): {PRICES}. Penalty weights (USD/kWh): {VOLL}.",
         f"Corrected Ymax raised the reference above the 'full' schedule in {int((TE.ymax_corr > TE.ymax_full_schedule + 1e-9).groupby([TE.site, TE.year]).any().sum())} "
         f"of {TE.groupby(['site', 'year']).ngroups} site-years; mean raise {float((TE.ymax_corr - TE.ymax_full_schedule).groupby([TE.site, TE.year]).first().mean()):.2f} t/ha.", ""]
key = CMP[CMP.metric.isin(["tc_v0_base", "tc_v50_base"])].copy()
for c in ("mean", "lo95", "hi95", "pct_of_b"):
    key[c] = key[c].round(1)
lines += ["## Paired differences, base price, penalty 0 and 50 USD/kWh", "", md(key, index=False), ""]
best = RK.groupby(["scenario"]).apply(lambda g: ", ".join(f"{s}:{b}" for s, b in zip(g.site, g.best)), include_groups=False)
lines += ["## Best family per site by scenario", "", md(best.to_frame("best")), ""]
lines += ["## MPC rank / hybrid rank (1 = best, 7 families)", "",
          md(RK.pivot(index="scenario", columns="site", values="mpc_rank")), "",
          md(RK.pivot(index="scenario", columns="site", values="hybrid_rank")), ""]
(OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(phys[["shed_kwh", "irrigation_mm", "yield_t_ha", "yield_gap_t_ha", "opex_ex_penalty", "tc_v0_base", "tc_v50_base"]].to_string())
print(seed_sd.to_string())
