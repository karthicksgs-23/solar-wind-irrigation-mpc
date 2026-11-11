"""Results tables for the manuscripts, written as Markdown snippets to manuscript/parts/tables/*.md from the final
result files (no hand-copied numbers). Run with the system Python 3.11."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
T = ROOT / "manuscript" / "parts" / "tables"; T.mkdir(parents=True, exist_ok=True)
SITES = ["bushland_tx", "jodhpur_in", "souss_ma", "azraq_jo"]
SN = {"bushland_tx": "Bushland", "jodhpur_in": "Jodhpur", "souss_ma": "Souss", "azraq_jo": "Azraq"}
FAM = [("rule", "Rule"), ("two_stage", "Two-stage"), ("mpc_default", "MPC default"), ("mpc_uncal", "MPC uncalibrated"),
       ("mpc", "**Joint MPC**"), ("hybrid", "Hybrid PPO–MPC"), ("drl", "PPO only"), ("mpc_perfect", "MPC perfect")]


def num(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace("-", "−")


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def write(name, text):
    (T / name).write_text(text + "\n", encoding="utf-8"); print("wrote", name)


# validation (Bushland)
v = pd.read_csv(O / "nb01_v2_drive" / "validation_metrics.csv")
rows = []
for params, plab in (("default", "Default"), ("calibrated", "Calibrated")):
    for split, slab in (("calibration_2013_2016", "2013, 2016 (n = 8)"), ("validation_2018", "2018 held out (n = 4)")):
        g = v[(v.params == params) & (v.split == split)].set_index("quantity")
        rows.append([plab, slab,
                     f"{num(g.at['seasonal_ET_mm','rmse'],1)} ({num(g.at['seasonal_ET_mm','nrmse_pct'],1)} %)", num(g.at['seasonal_ET_mm', 'bias'], 1),
                     f"{num(g.at['storage_1p5m_mm','rmse'],1)} ({num(g.at['storage_1p5m_mm','nrmse_pct'],1)} %)",
                     f"{num(g.at['yield_Mg_ha','rmse'],2)} ({num(g.at['yield_Mg_ha','nrmse_pct'],1)} %)", num(g.at['yield_Mg_ha', 'bias'], 2)])
write("t_validation.md", md(["AquaCrop parameters", "Seasons", "Seasonal ET RMSE, mm (nRMSE)", "Seasonal ET bias (mm)",
                             "Soil water storage RMSE, mm (nRMSE)", "Yield RMSE, t ha⁻¹ (nRMSE)", "Yield bias (t ha⁻¹)"], rows))

# twin calibration
f = json.load(open(O / "nb03v3_colab" / "crop_calibration_fit.json"))
rows = [[SN[s], num(f[s]["p"], 3), num(f[s]["ky"], 2), num(f[s]["rmse_default"], 3), num(f[s]["rmse_cal"], 3), num(f[s]["r2"], 2),
         "yes" if f[s]["on_grid_edge"] else "no"] for s in SITES]
write("t_twin.md", md(["Site", "p", "Ky", "RMSE, default (p 0.55, Ky 1.25)", "RMSE, calibrated", "R², calibrated", "Optimum on grid edge"], rows))

# sizing
s = pd.read_csv(O / "nb02_local_test" / "sizing_summary.csv")
rows = []
for sid in SITES:
    for des, lab in (("least_cost", "least cost"), ("hybrid", "PV–wind hybrid")):
        r = s[(s.site == sid) & (s.design == des)].iloc[0]
        rows.append([SN[sid], lab, num(r.pump_kw, 1), num(r.pv_kwp, 1), num(r.wind_kw, 1), num(r.batt_kwh, 1), num(r.tank_m3, 0),
                     num(r.npc_usd / 1000, 1), num(r.lcoe_usd_kwh, 3), num(100 * r.curtail_frac, 0), num(100 * r.wlpsp_test, 1),
                     ("< 0.01" if 0 < 100 * r.elpsp_test < 0.005 else num(100 * r.elpsp_test, 2)), num(r.diesel_npc_usd / 1000, 1), num(r.npc_usd / r.diesel_npc_usd, 2)])
write("t_sizing.md", md(["Site", "Design", "Pump (kW)", "PV (kWp)", "Wind (kW)", "Battery (kWh)", "Tank (m³)", "NPC (k USD)",
                         "LCOE (USD kWh⁻¹)", "Curtailment (%)", "WLPSP 2020–24 (%)", "ELPSP 2020–24 (%)", "Diesel NPC (k USD)",
                         "NPC / diesel"], rows))

# forecast skill, lead 1
fs = pd.read_csv(O / "nb03v3_colab" / "forecast_skill.csv")
fs = fs[fs.lead == 1]
rows = []
for sid in SITES:
    g = fs[fs.site == sid].set_index("var")
    rows.append([SN[sid]] + [f"{num(g.at[k,'nmae_pred'],3)} ({num(100*g.at[k,'skill_vs_persist'],1)})" for k in ("pv", "wind", "load", "et0", "rain")])
write("t_forecast.md", md(["Site", "PV", "Wind", "Farm load", "ET₀", "Rain"], rows))

# controller outcomes (test means)
p = pd.read_csv(O / "nb03v3_colab" / "posthoc" / "table_physical_and_cost.csv")
rows = []
for sid in SITES:
    for k, lab in FAM:
        r = p[(p.site == sid) & (p.family == k)].iloc[0]
        rows.append([SN[sid], lab, num(r.shed_kwh, 0), num(100 * r.elpsp, 1), num(r.irrigation_mm, 0), num(r.yield_t_ha, 2),
                     num(r.wp_kg_m3, 2), num(r.batt_throughput_kwh, 0), num(r.tc_v0_base, 0), num(r.tc_v50_base, 0)])
write("t_controllers.md", md(["Site", "Controller", "Unserved load (kWh yr⁻¹)", "ELPSP (%)", "Net irrigation (mm)",
                              "AquaCrop yield (t ha⁻¹)", "WP (kg m⁻³)", "Battery throughput (kWh yr⁻¹)",
                              "TC, VoLL 0 (USD yr⁻¹)", "TC, VoLL 50 (USD yr⁻¹)"], rows))

# paired contrasts
b = pd.read_csv(O / "nb03v3_colab" / "posthoc" / "paired_bootstrap_all_scenarios.csv")
CON = [("mpc - rule", "Joint MPC − rule"), ("mpc - two_stage", "Joint MPC − two-stage"), ("two_stage - rule", "Two-stage − rule"),
       ("mpc - mpc_uncal", "Calibrated − uncalibrated MPC"), ("mpc - mpc_default", "Tuned − default MPC"),
       ("hybrid - mpc", "Hybrid PPO–MPC − joint MPC"), ("drl - mpc", "PPO only − joint MPC"), ("mpc_perfect - mpc", "Perfect − forecast MPC")]


def cell(g, sid):
    r = g[g.site == sid].iloc[0]
    return f"{num(r['mean'],0)} [{num(r.lo95,0)}, {num(r.hi95,0)}]{'*' if r.sig else ''}"


rows = []
for key, lab in CON:
    for met, mlab in (("tc_v0_base", "0"), ("tc_v50_base", "50")):
        g = b[(b.metric == met) & (b.contrast == key)]
        rows.append([lab, mlab] + [cell(g, sid) for sid in SITES])
write("t_contrasts.md", md(["Contrast", "VoLL (USD kWh⁻¹)"] + [SN[s] for s in SITES], rows))

rows = []
for key, lab in CON[:2] + CON[3:4] + CON[5:7]:
    for met, mlab, d in (("yield_t_ha", "yield (t ha⁻¹)", 2), ("irrigation_mm", "irrigation (mm)", 0)):
        g = b[(b.metric == met) & (b.contrast == key)]
        rows.append([lab, mlab] + [f"{num(g[g.site==sid].iloc[0]['mean'],d)} [{num(g[g.site==sid].iloc[0].lo95,d)}, {num(g[g.site==sid].iloc[0].hi95,d)}]{'*' if g[g.site==sid].iloc[0].sig else ''}" for sid in SITES])
write("t_contrasts_phys.md", md(["Contrast", "Quantity"] + [SN[s] for s in SITES], rows))

# seed spread of the learning controllers
sd = pd.read_csv(O / "nb03v3_colab" / "posthoc" / "table_seed_spread.csv", header=[0, 1], index_col=[0, 1])
rows = []
for sid in SITES:
    for fam, lab in (("hybrid", "Hybrid PPO–MPC"), ("drl", "PPO only")):
        r = sd.loc[(sid, fam)]
        rows.append([SN[sid], lab, f"{num(r[('tc_v0_base','mean')],0)} ± {num(r[('tc_v0_base','std')],0)}",
                     f"{num(r[('tc_v0_base','min')],0)}–{num(r[('tc_v0_base','max')],0)}",
                     f"{num(r[('irrigation_mm','min')],0)}–{num(r[('irrigation_mm','max')],0)}",
                     f"{num(r[('yield_t_ha','min')],2)}–{num(r[('yield_t_ha','max')],2)}"])
write("t_seeds.md", md(["Site", "Controller", "TC, VoLL 0 (mean ± sd over seeds)", "TC range over seeds", "Irrigation range (mm)",
                        "Yield range (t ha⁻¹)"], rows))

# sensitivity summary
se = pd.read_csv(O / "nb02_local_test" / "sensitivity.csv")
se["ratio"] = se.npc_usd / se.diesel_npc_usd
print("sensitivity cases", len(se), "renewable cheaper", int(se.renewable_cheaper.sum()), "with wind", int((se.wind_kw > 0).sum()))
print(se.groupby("scenario").ratio.agg(["min", "max"]).round(2).to_string())
