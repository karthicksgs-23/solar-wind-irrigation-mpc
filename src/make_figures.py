"""Manuscript figures from the final result files (no re-simulation).
Writes manuscript/figures/fig1..fig6 (PNG, 300 dpi). Run with the system Python 3.11 (matplotlib, pandas)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
FIG = ROOT / "manuscript" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

# reference palette, first three categorical slots (validated all-pairs) + text inks
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SITES = ["bushland_tx", "jodhpur_in", "souss_ma", "azraq_jo"]
SNAME = {"bushland_tx": "Bushland (USA)", "jodhpur_in": "Jodhpur (India)", "souss_ma": "Souss (Morocco)", "azraq_jo": "Azraq (Jordan)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(FIG / name); plt.close(fig); print("saved", name)


# ---------------------------------------------------------------- Fig. 1 framework
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.0)); ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(0, 60)
    def box(x, y, w, h, title, body, fc="#f3f2ef"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=6.6, color=INK2, linespacing=1.35)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    box(1, 36, 30, 22, "1  Datasets", "NASA POWER hourly/daily 2001–2024\nSoilGrids 2.0 soil water limits\nUSDA-ARS Bushland lysimeters\n(maize 2013, 2016, 2018)")
    box(35, 36, 30, 22, "2  Sizing (NPC)", "PV, wind, Li-ion battery, tank\ngrid search, 2001–2016\nWLPSP ≤ 5 %, ELPSP ≤ 2 %\ndiesel-pump reference")
    box(69, 36, 30, 22, "3  Crop layer", "AquaCrop-OSPy calibrated on\nBushland lysimeters\n(2018 held out)\nFAO-56 twin: p, Ky fitted\nto AquaCrop deficit runs")
    box(1, 4, 30, 26, "4  Day-ahead forecasts", "gradient boosting, 2-day lead\nPV, wind, farm load, ET0, rain\nblocked CV on training years")
    box(35, 4, 30, 26, "5  Controllers (48 h)", "rule · two-stage (decoupled)\njoint MPC (LP), tuned reserve\nMPC uncalibrated · MPC default\nhybrid PPO–MPC · PPO only\nMPC with perfect forecasts", fc="#e8f0fb")
    box(69, 4, 30, 26, "6  Evaluation 2020–2024", "AquaCrop yield of each schedule\nunserved load, water, cycling\ntotal cost = operating cost\n+ yield-loss value\npaired year bootstrap")
    arrow(31, 47, 35, 47); arrow(65, 47, 69, 47); arrow(16, 36, 16, 30); arrow(50, 36, 50, 30); arrow(84, 36, 84, 30)
    arrow(31, 17, 35, 17); arrow(65, 17, 69, 17)
    save(fig, "fig1_framework.png")


# ---------------------------------------------------------------- Fig. 2 Bushland validation
def fig2():
    d = O / "nb01_v2_drive"
    cal = pd.read_csv(d / "seasons_calibrated_calibration_2013_2016.csv"); val = pd.read_csv(d / "seasons_calibrated_validation_2018.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, (o, s, lab, lo, hi) in zip(axs, [("ET_obs", "ET_sim", "seasonal ET (mm)", 600, 880),
                                              ("yield_obs", "yield_sim", "grain yield (t ha$^{-1}$)", 6, 16)]):
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls=(0, (3, 3)))
        for df, c, mk, name in ((cal, C1, "o", "calibration 2013, 2016"), (val, C2, "s", "validation 2018 (held out)")):
            ax.scatter(df[o], df[s], s=34, color=c, marker=mk, edgecolor="white", linewidth=1.0, label=name, zorder=3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"observed {lab}"); ax.set_ylabel(f"AquaCrop {lab}")
    h, l = axs[0].get_legend_handles_labels()                 # one legend shared by (a) and (b)
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.06), handletextpad=0.3, columnspacing=2.0)
    axs[0].set_title("(a) seasonal evapotranspiration", loc="left", fontsize=9); axs[1].set_title("(b) grain yield", loc="left", fontsize=9)
    save(fig, "fig2_bushland_validation.png")


# ---------------------------------------------------------------- Fig. 3 twin calibration against AquaCrop
def fig3():
    pts = pd.read_csv(O / "nb03v3_colab" / "crop_calibration_points.csv")
    fit = json.load(open(O / "nb03v3_colab" / "crop_calibration_fit.json"))
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.35), sharey=True)
    for ax, sid in zip(axs, SITES):
        p = pts[pts.site == sid]; f = fit[sid]
        ax.plot([0, 1.05], [0, 1.05], color=INK2, lw=0.8, ls=(0, (3, 3)))
        ax.scatter(p.rel_aquacrop, p.twin_default, s=12, marker="x", color=C2, linewidth=0.9, label="default (p 0.55, Ky 1.25)")
        ax.scatter(p.rel_aquacrop, p.twin_calibrated, s=14, color=C1, edgecolor="white", linewidth=0.6, label="calibrated")
        ax.set_title(f"{SNAME[sid]}\np {f['p']:.3g}, Ky {f['ky']:.3g}, R² {f['r2']:.2f}", fontsize=7.8, loc="left")
        ax.set_xlim(0, 1.05); ax.set_ylim(-0.03, 1.08); ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel("AquaCrop relative yield")
    axs[0].set_ylabel("twin relative yield"); axs[0].legend(loc="lower right", fontsize=6.6, handletextpad=0.2)
    save(fig, "fig3_twin_calibration.png")


# ---------------------------------------------------------------- Fig. 4 sizing and NPC
def fig4():
    s = pd.read_csv(O / "nb02_local_test" / "sizing_summary.csv")
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    x = np.arange(len(SITES)); w = 0.26
    series = [("least_cost", "npc_usd", "PV + battery + tank (least cost)", C1),
              ("hybrid", "npc_usd", "PV + wind + battery + tank", C2),
              ("least_cost", "diesel_npc_usd", "diesel-pump reference", C3)]
    for k, (des, col, lab, c) in enumerate(series):
        v = [float(s[(s.site == sid) & (s.design == des)][col].iloc[0]) / 1000 for sid in SITES]
        bars = ax.bar(x + (k - 1) * w, v, w - 0.02, color=c, label=lab, edgecolor="white", linewidth=1.0)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 3, f"{val:.0f}", ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.set_xticks(x, [SNAME[sid] for sid in SITES]); ax.set_ylabel("net present cost, 25 yr (k USD)")
    ax.set_ylim(0, 300); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3, fontsize=7.2, bbox_to_anchor=(0, 1.12))
    save(fig, "fig4_sizing_npc.png")


# ---------------------------------------------------------------- Fig. 5 paired contrasts (forest plot)
CON = [("mpc - rule", "joint MPC − rule"), ("mpc - two_stage", "joint MPC − two-stage"),
       ("mpc - mpc_uncal", "calibrated − uncalibrated MPC"), ("hybrid - mpc", "hybrid PPO–MPC − MPC"),
       ("drl - mpc", "PPO only − MPC"), ("mpc_perfect - mpc", "perfect − forecast MPC")]


def fig5(metric="tc_v0_base", name="fig5_paired_contrasts.png", xlabel="difference in total cost (USD yr$^{-1}$), negative = first is cheaper"):
    b = pd.read_csv(O / "nb03v3_colab" / "posthoc" / "paired_bootstrap_all_scenarios.csv")
    b = b[b.metric == metric]
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.2), sharey=True)
    for ax, (key, lab) in zip(axs.flat, CON):
        sub = b[b.contrast == key].set_index("site").reindex(SITES)
        y = np.arange(len(SITES))[::-1]
        ax.axvline(0, color=INK2, lw=0.8)
        for yi, (sid, r) in zip(y, sub.iterrows()):
            ax.plot([r.lo95, r.hi95], [yi, yi], color=C1, lw=2, solid_capstyle="round")
            ax.scatter([r["mean"]], [yi], s=36, zorder=3, color=C1 if r.sig else "white", edgecolor=C1, linewidth=1.4)
        ax.set_yticks(y, [SNAME[s] for s in SITES]); ax.set_title(lab, fontsize=8.3, loc="left")
        ax.grid(axis="y", visible=False)
    for ax in axs[1]:
        ax.set_xlabel("USD yr$^{-1}$")
    fig.text(0.5, -0.02, xlabel + ";  filled = 95 % CI excludes 0, open = not significant", ha="center", fontsize=7.6, color=INK2)
    fig.tight_layout()
    save(fig, name)


# ---------------------------------------------------------------- Fig. 6 physical outcomes per controller
FAMS = [("rule", "rule"), ("two_stage", "two-stage"), ("mpc_uncal", "MPC uncal."), ("mpc", "joint MPC"),
        ("hybrid", "hybrid PPO–MPC"), ("drl", "PPO only")]


def fig6():
    t = pd.read_csv(O / "nb03v3_colab" / "posthoc" / "table_physical_and_cost.csv")
    mets = [("shed_kwh", "unserved farm load (kWh yr$^{-1}$)"), ("irrigation_mm", "net irrigation (mm)"),
            ("yield_t_ha", "AquaCrop yield (t ha$^{-1}$)")]
    fig, axs = plt.subplots(len(mets), 4, figsize=(7.4, 5.4), sharey=True)
    y = np.arange(len(FAMS))[::-1]
    for j, sid in enumerate(SITES):
        sub = t[t.site == sid].set_index("family")
        for i, (col, lab) in enumerate(mets):
            ax = axs[i, j]; v = [sub.at[f, col] for f, _ in FAMS]
            cols = [C2 if f == "mpc" else C1 for f, _ in FAMS]
            if col == "yield_t_ha":
                ax.scatter(v, y, s=26, color=cols, edgecolor="white", linewidth=0.8, zorder=3)
                ax.set_xlim(min(v) - 0.3, max(v) + 0.3)
            else:
                ax.barh(y, v, 0.62, color=cols, edgecolor="white", linewidth=1.0)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(SNAME[sid], fontsize=8.3, loc="left")
            if j == 0:
                ax.set_yticks(y, [n for _, n in FAMS])
            ax.set_xlabel(lab, fontsize=7.2); ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "fig6_physical_outcomes.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
