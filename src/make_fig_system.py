"""Two candidate system-architecture figures in the style of the user's reference images.
Option 1: layered (control / information / physical layers).  Option 2: power-flow block diagram with the controller
in a dotted box.  Content follows src/env.py (hourly energy balance, pump -> tank -> drip, 48 h joint LP MPC).
Run: python src/make_fig_system.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Ellipse, Circle

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif", "savefig.dpi": 400,
                     "savefig.bbox": "tight"})
INK, GREY, BLUE, RED = "#1a1a1a", "#8c8c8c", "#2a64b8", "#c62828"


def new_ax(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.set_axis_off(); ax.set_aspect("equal")
    return fig, ax


def rbox(ax, x, y, w, h, text, fc="#fff8dc", ec=INK, fs=7.2, lw=0.9, ls="-", weight="normal", r=1.2, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, color=color, linespacing=1.3)


def arr(ax, p0, p1, color=INK, lw=1.0, ls="-", style="-|>", ms=8, conn="arc3"):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, mutation_scale=ms,
                                                      shrinkA=0, shrinkB=0, connectionstyle=conn))


def line(ax, pts, color=INK, lw=1.0, ls="-"):
    xs, ys = zip(*pts); ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="butt")


# ------------------------------------------------------------------ icons
def pv_icon(ax, x, y, s=1.0):
    p = Polygon([(x, y), (x + 9 * s, y), (x + 11 * s, y + 5 * s), (x + 2 * s, y + 5 * s)], fc="#1f4e8c", ec=INK, lw=0.7)
    ax.add_patch(p)
    for k in range(1, 4):
        line(ax, [(x + k * 9 * s / 4, y), (x + k * 9 * s / 4 + 2 * s, y + 5 * s)], color="#9fc3ee", lw=0.5)
    for k in (1, 2):
        f = k / 3; line(ax, [(x + 2 * s * f, y + 5 * s * f), (x + 9 * s + 2 * s * f, y + 5 * s * f)], color="#9fc3ee", lw=0.5)
    line(ax, [(x + 5.5 * s, y), (x + 5.5 * s, y - 2.2 * s)], lw=0.9)


def wind_icon(ax, x, y, s=1.0, ls="-"):
    import numpy as np
    ax.add_patch(Polygon([(x - 0.6 * s, y), (x - 0.25 * s, y + 10 * s), (x + 0.25 * s, y + 10 * s), (x + 0.6 * s, y)],
                         fc="#e9e9e9", ec=INK, lw=0.7, ls=ls))
    hub = np.array([x, y + 10 * s])
    for ang in (90, 210, 330):                       # tapered blades
        a = np.deg2rad(ang); d = np.array([np.cos(a), np.sin(a)]); n = np.array([-d[1], d[0]])
        pts = [hub + 0.5 * s * n, hub + 2.0 * s * d + 0.75 * s * n, hub + 5.2 * s * d, hub - 0.3 * s * n]
        ax.add_patch(Polygon(pts, fc="#f4f4f4", ec=INK, lw=0.7, ls=ls))
    ax.add_patch(Circle(hub, 0.6 * s, fc="white", ec=INK, lw=0.7, zorder=5))


def batt_icon(ax, x, y, s=1.0):
    ax.add_patch(Rectangle((x, y), 8 * s, 4.5 * s, fc="#d9efd9", ec=INK, lw=0.8))
    ax.add_patch(Rectangle((x + 8 * s, y + 1.4 * s), 0.8 * s, 1.7 * s, fc=INK, ec=INK))
    for k in range(3):
        ax.add_patch(Rectangle((x + (0.7 + 2.4 * k) * s, y + 0.7 * s), 1.9 * s, 3.1 * s, fc="#3c9a4b", ec="none"))


def tank_icon(ax, x, y, s=1.0):
    w, h = 7 * s, 7 * s
    ax.add_patch(Rectangle((x, y), w, h * 0.6, fc="#7fb3e6", ec="none"))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec=INK, lw=0.8))
    ax.add_patch(Ellipse((x + w / 2, y + h), w, 1.6 * s, fc="#eef4fb", ec=INK, lw=0.8))
    ax.add_patch(Ellipse((x + w / 2, y), w, 1.6 * s, fc="#7fb3e6", ec=INK, lw=0.8, zorder=0))


def pump_icon(ax, x, y, s=1.0):
    ax.add_patch(Circle((x, y), 2.4 * s, fc="#f4d6a0", ec=INK, lw=0.8))
    ax.text(x, y, "M", ha="center", va="center", fontsize=6.5 * s, weight="bold")


def plant_icon(ax, x, y, s=1.0):
    line(ax, [(x, y), (x, y + 6 * s)], color="#2e7d32", lw=1.1)
    for k, sgn in ((1.5, 1), (2.8, -1), (4.1, 1)):
        ax.add_patch(Ellipse((x + sgn * 1.3 * s, y + k * s), 2.8 * s, 0.9 * s, angle=sgn * 30, fc="#66bb6a", ec="#2e7d32", lw=0.5))
    ax.add_patch(Ellipse((x + 0.5 * s, y + 5.6 * s), 0.9 * s, 1.8 * s, angle=-15, fc="#f2c94c", ec="#b58a12", lw=0.5))


def house_icon(ax, x, y, s=1.0):
    ax.add_patch(Rectangle((x, y), 6 * s, 4 * s, fc="#f0e0d0", ec=INK, lw=0.8))
    ax.add_patch(Polygon([(x - 0.8 * s, y + 4 * s), (x + 3 * s, y + 7 * s), (x + 6.8 * s, y + 4 * s)], fc="#b5533c", ec=INK, lw=0.8))
    ax.add_patch(Rectangle((x + 2.2 * s, y), 1.6 * s, 2.4 * s, fc="#7a5230", ec=INK, lw=0.5))


def t(ax, x, y, s, fs=6.2, color=INK, **kw):
    ax.text(x, y, s, fontsize=fs, color=color, **kw)


# ================================================================== OPTION 1: layered
def option1(path):
    fig, ax = new_ax(10.4, 5.0, 208, 100)
    for x, w, fc, ttl in [(1, 93, "#f3d9da", "Control layer"), (96, 41, "#e6e6ea", "Information layer"),
                          (139, 68, "#f8ebc6", "Physical layer")]:
        ax.add_patch(FancyBboxPatch((x, 1), w, 97, boxstyle="round,pad=0,rounding_size=2.5", fc=fc, ec="none"))
        t(ax, x + w / 2, 94.5, ttl, fs=9.5, ha="center", va="center", weight="bold")

    # ---- physical layer
    bx = 172
    line(ax, [(bx, 7.5), (bx, 86)], lw=2.6); t(ax, bx, 87.5, "DC bus", fs=6.6, ha="center")
    pv_icon(ax, 145, 77); t(ax, 151, 72.6, "PV array", fs=6.6, ha="center")
    line(ax, [(156.5, 79.5), (bx, 79.5)], lw=0.9); arr(ax, (163, 79.5), (166, 79.5), ms=7)
    wind_icon(ax, 152, 56, ls="--"); t(ax, 152, 53.6, "Wind turbine\n(sizing candidate)", fs=6.0, ha="center", va="top")
    line(ax, [(157.5, 64), (bx, 64)], lw=0.9, ls="--"); arr(ax, (163, 64), (166, 64), ms=7)
    batt_icon(ax, 147, 30); t(ax, 151.5, 27.8, "Li-ion battery", fs=6.6, ha="center", va="top")
    line(ax, [(156, 32.3), (bx, 32.3)], lw=0.9); arr(ax, (161, 32.3), (166, 32.3), ms=7, style="<|-|>")
    rbox(ax, 141, 3, 22, 9.5, "Converters, VFD,\ndrip valves", fc="white", fs=5.8)
    line(ax, [(163, 7.5), (bx, 7.5)], lw=0.9)

    house_icon(ax, 191, 80); t(ax, 194, 77.6, "Farm loads\n(priority)", fs=6.0, ha="center", va="top")
    line(ax, [(bx, 82), (191, 82)], lw=0.9); arr(ax, (179, 82), (183, 82), ms=7)
    pump_icon(ax, 180, 60); t(ax, 180, 63.2, "VFD pump", fs=6.0, ha="center", va="bottom")
    line(ax, [(bx, 60), (177.6, 60)], lw=0.9)
    tank_icon(ax, 190, 56.5, 0.95); t(ax, 193.3, 66.0, "Tank", fs=6.0, ha="center", va="bottom")
    line(ax, [(182.4, 60), (190, 60)], color=BLUE, lw=1.2); arr(ax, (184.5, 60), (188, 60), color=BLUE, ms=7)
    line(ax, [(193.3, 55.7), (193.3, 32)], color=BLUE, lw=1.2); arr(ax, (193.3, 45), (193.3, 40), color=BLUE, ms=7)
    t(ax, 194.5, 44, "drip", fs=6.0, color=BLUE)
    for k in range(4):
        plant_icon(ax, 181 + 6 * k, 17, 0.95)
    line(ax, [(178, 16.6), (205, 16.6)], color="#8d6e63", lw=2.0)
    line(ax, [(179, 31.6), (204, 31.6)], color=BLUE, lw=0.8, ls=":")
    t(ax, 191.5, 13.5, "Maize field ($D_r$)", fs=6.0, ha="center", va="top")
    line(ax, [(bx, 42), (177, 42)], color=GREY, lw=0.9); t(ax, 177.8, 42, "curtail", fs=5.8, color=GREY, va="center")

    # measurements (blue dashed): tank + soil share a trunk; SOC goes straight across
    line(ax, [(196.7, 61), (205, 61)], color=BLUE, lw=0.9, ls="--")
    line(ax, [(203.5, 22), (205, 22)], color=BLUE, lw=0.9, ls="--")
    line(ax, [(205, 22), (205, 91), (137, 91), (137, 34.5)], color=BLUE, lw=0.9, ls="--")
    arr(ax, (137, 34.5), (134.2, 34.5), color=BLUE, ms=7)
    t(ax, 204, 70, "$V_{tank}$, $D_r$", fs=6.0, color=BLUE, ha="right")
    line(ax, [(147, 31.2), (138, 31.2)], color=BLUE, lw=0.9, ls="--"); arr(ax, (138, 31.2), (134.2, 31.2), color=BLUE, ms=7)
    t(ax, 141.5, 32.4, "SoC", fs=6.0, color=BLUE, ha="center")

    # ---- information layer
    rbox(ax, 99, 70, 35, 17, "Weather data\nNASA POWER hourly\n2001–2024", fc="white", fs=6.6)
    rbox(ax, 99, 44, 35, 19, "Day-ahead forecaster\n(gradient boosting)\n$\\hat P_{PV},\\ \\hat P_{W},\\ \\hat L,$\n"
                             "$\\widehat{ET_0},\\ \\hat R$", fc="white", fs=6.6)
    rbox(ax, 99, 14, 35, 26, "Measured states\nbattery SOC\ntank volume $V_{tank}$\nsoil depletion $D_r$", fc="white", fs=6.6)
    arr(ax, (116.5, 70), (116.5, 63.2), ms=7)

    # ---- control layer
    rbox(ax, 5, 70, 40, 17, "Crop calibration\nAquaCrop-OSPy on Bushland\nlysimeters (2018 held out)", fs=6.6)
    rbox(ax, 50, 70, 40, 17, "FAO-56 crop twin\n$p,\\ K_y$ fitted to AquaCrop\ndeficit runs", fs=6.6)
    arr(ax, (45, 78.5), (50, 78.5), ms=7)
    rbox(ax, 5, 54, 40, 11, "Yield value\n$V$ = price × $Y_{max}$ × area", fs=6.6)
    ax.add_patch(FancyBboxPatch((5, 18), 85, 31, boxstyle="round,pad=0,rounding_size=1.5", fc="#fff8dc", ec=INK, lw=1.3))
    t(ax, 47.5, 45.5, "Crop-calibrated, yield-priced joint MPC", fs=7.6, weight="bold", ha="center")
    t(ax, 47.5, 41.5, "linear program, 48 h horizon, re-planned daily", fs=6.4, ha="center", style="italic")
    t(ax, 47.5, 36.2, "$\\min\\ \\sum_t\\ c_{shed}P_{shed}+V K_y(1-K_s)\\,ET_c/ET_{c,S}+c_w Q_{rel}+c_{cyc}P_{ch}$",
      fs=6.6, ha="center")
    t(ax, 47.5, 31.8, "+ terminal values on SOC, tank and $D_r$", fs=6.4, ha="center")
    t(ax, 47.5, 26.0, "s.t. energy balance · SOC limits + night reserve · tank balance\n"
                      "soil-water balance · VFD minimum power (30 %)", fs=6.2, ha="center", va="center", linespacing=1.4)
    t(ax, 47.5, 20.4, "tuned: reserve, $\\tau$ (end-of-horizon $D_r$ target)", fs=6.0, ha="center", color="#555")
    arr(ax, (70, 70), (70, 49.2), ms=7); arr(ax, (25, 54), (25, 49.2), ms=7)
    # grey information links
    line(ax, [(99, 53.5), (82, 53.5)], color=GREY, lw=2.4); arr(ax, (82, 53.7), (82, 49.6), color=GREY, lw=2.4, ms=10)
    t(ax, 90.5, 55.2, "forecasts", fs=5.8, color="#555", ha="center")
    arr(ax, (99, 27), (90.6, 27), color=GREY, lw=2.4, ms=10); t(ax, 94.8, 28.8, "states", fs=5.8, color="#555", ha="center")
    # set-points and control signals
    rbox(ax, 5, 3.5, 85, 10.5, "Hourly set-points (first 24 h applied): pump power $P_{pump}$, drip release $Q_{rel}$,\n"
                               "battery $P_{ch}/P_{dis}$, farm-load shed $P_{shed}$, curtailment $P_{cur}$", fc="white", fs=5.9)
    arr(ax, (47.5, 18), (47.5, 14.2), ms=7)
    line(ax, [(90, 7.5), (138, 7.5)], color=BLUE, lw=1.0, ls="--"); arr(ax, (138, 7.5), (140.8, 7.5), color=BLUE, ms=7)
    t(ax, 115, 9.2, "control signals", fs=6.0, color=BLUE, ha="center", style="italic")
    fig.savefig(path); plt.close(fig); print("saved", path)


# ================================================================== OPTION 2: block diagram
GFC, GEC = "#c8f0c8", "#2b6b2b"


def gbox(ax, x, y, w, h, s, fs=6.8, fc=GFC, ls="-"):
    rbox(ax, x, y, w, h, s, fc=fc, ec=GEC, fs=fs, lw=1.0, r=1.6, ls=ls)


def option2(path):
    fig, ax = new_ax(10.0, 5.4, 200, 108)
    ax.add_patch(Rectangle((1, 1), 198, 106, fc="white", ec=INK, lw=0.8))
    # sources
    pv_icon(ax, 7, 88, 1.15); t(ax, 13.5, 83.2, "PV array", fs=7, ha="center", va="top")
    wind_icon(ax, 13, 55, 1.15, ls="--"); t(ax, 13, 52.6, "Wind turbine\n(sizing candidate)", fs=6.4, ha="center", va="top")
    gbox(ax, 33, 86, 26, 12, "PV converter\n(PVWatts model)")
    gbox(ax, 33, 60, 26, 12, "Wind converter\n(power curve)", ls="--")
    line(ax, [(21, 92), (33, 92)]); arr(ax, (26, 92), (29, 92), ms=8)
    line(ax, [(19.5, 66), (33, 66)], ls="--"); arr(ax, (26, 66), (29, 66), ms=8)
    # DC bus
    line(ax, [(70, 45), (70, 100)], lw=3.0); t(ax, 70, 101.8, "DC bus", fs=7, ha="center")
    line(ax, [(59, 92), (70, 92)]); arr(ax, (62, 92), (66, 92), ms=8)
    line(ax, [(59, 66), (70, 66)], ls="--"); arr(ax, (62, 66), (66, 66), ms=8)
    line(ax, [(70, 47), (62, 47)], color=GREY); t(ax, 61, 47, "curtailment", fs=6.2, color=GREY, ha="right", va="center")
    # farm loads
    gbox(ax, 82, 86, 24, 12, "Inverter")
    line(ax, [(70, 92), (82, 92)]); arr(ax, (74, 92), (78, 92), ms=8)
    house_icon(ax, 119, 87, 1.25); t(ax, 122.8, 84.4, "Farm loads\n(priority, may be shed)", fs=6.5, ha="center", va="top")
    line(ax, [(106, 92), (118, 92)]); arr(ax, (110, 92), (114, 92), ms=8)
    # pump -> tank -> drip -> field
    gbox(ax, 82, 65, 24, 12, "VFD\n(min. 30 % power)")
    line(ax, [(70, 71), (82, 71)]); arr(ax, (74, 71), (78, 71), ms=8)
    pump_icon(ax, 117, 71, 1.15); t(ax, 117, 74.3, "Pump", fs=6.5, ha="center", va="bottom")
    line(ax, [(106, 71), (114.2, 71)])
    gbox(ax, 133, 65, 22, 12, "Storage tank\n$V_{tank}$", fc="#cfe3f7")
    line(ax, [(119.8, 71), (133, 71)], color=BLUE, lw=1.2); arr(ax, (124, 71), (129, 71), color=BLUE, ms=8)
    gbox(ax, 161, 65, 18, 12, "Drip\nvalves", fc="#cfe3f7")
    line(ax, [(155, 71), (161, 71)], color=BLUE, lw=1.2); arr(ax, (156, 71), (159.5, 71), color=BLUE, ms=8)
    for k in range(4):
        plant_icon(ax, 163 + 7 * k, 87, 1.1)
    line(ax, [(160, 86.6), (196, 86.6)], color="#8d6e63", lw=2.2)
    t(ax, 178, 96.5, "Maize field ($D_r$)", fs=6.8, ha="center")
    line(ax, [(179, 71), (186, 71), (186, 86)], color=BLUE, lw=1.2); arr(ax, (186, 78), (186, 83), color=BLUE, ms=8)
    # battery
    gbox(ax, 82, 44, 24, 12, "Bidirectional\nDC/DC converter")
    line(ax, [(70, 50), (82, 50)]); arr(ax, (72.5, 50), (79.5, 50), ms=8, style="<|-|>")
    line(ax, [(106, 50), (108, 50)]); batt_icon(ax, 108, 47.4, 1.15)
    t(ax, 120.3, 50, "Li-ion\nbattery", fs=6.2, va="center")

    # ---- controller (dotted red box)
    ax.add_patch(Rectangle((4, 4), 192, 32, fc="none", ec=RED, lw=1.6, ls=(0, (1, 1.5))))
    t(ax, 5, 37.5, "Controller", fs=7.2, color=RED, weight="bold", va="bottom")
    gbox(ax, 6, 21, 49, 12, "Day-ahead forecaster (gradient boosting)\nNASA POWER → PV, wind, load, $ET_0$, rain", fs=6.0)
    gbox(ax, 6, 6.5, 49, 12, "Crop twin (FAO-56): $p,\\ K_y$ calibrated\nto AquaCrop-OSPy (Bushland lysimeters)", fs=6.0)
    gbox(ax, 60, 8, 90, 22, "Crop-calibrated, yield-priced joint MPC\n(linear program, 48 h horizon, re-planned daily)\n"
                           "min: farm-load shed + yield-loss value + water + battery cycling", fs=7.0)
    arr(ax, (55, 27), (60, 27), ms=8); arr(ax, (55, 12.5), (60, 12.5), ms=8)
    # set-points (red)
    arr(ax, (90, 30), (90, 44), color=RED, ms=7); t(ax, 91, 39.5, "$P_{ch}/P_{dis}$", fs=6.0, color=RED)
    line(ax, [(130, 30), (130, 67), (107, 67)], color=RED, lw=1.0); arr(ax, (108, 67), (106.2, 67), color=RED, ms=7)
    t(ax, 131, 46, "$P_{pump}$", fs=6.0, color=RED)
    line(ax, [(150, 20), (172, 20), (172, 63)], color=RED, lw=1.0); arr(ax, (172, 62), (172, 64.8), color=RED, ms=7)
    t(ax, 173, 46, "$Q_{rel}$", fs=6.0, color=RED)
    # measured states (blue dashed)
    line(ax, [(112, 47.4), (112, 33)], color=BLUE, ls="--", lw=0.9); arr(ax, (112, 33), (112, 30.2), color=BLUE, ms=7)
    t(ax, 111, 39, "SoC", fs=6.0, color=BLUE, ha="right")
    line(ax, [(140, 65), (140, 33)], color=BLUE, ls="--", lw=0.9); arr(ax, (140, 33), (140, 30.2), color=BLUE, ms=7)
    t(ax, 141, 46, "$V_{tank}$", fs=6.0, color=BLUE)
    line(ax, [(194, 86.6), (194, 14), (153, 14)], color=BLUE, ls="--", lw=0.9); arr(ax, (153, 14), (150.2, 14), color=BLUE, ms=7)
    t(ax, 193, 46, "$D_r$", fs=6.4, color=BLUE, ha="right")
    fig.savefig(path); plt.close(fig); print("saved", path)


if __name__ == "__main__":
    option1(ROOT / "system_figure_option1_layered.png")
    option2(ROOT / "system_figure_option2_blockdiagram.png")


