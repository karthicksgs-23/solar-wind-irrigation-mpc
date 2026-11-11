"""Digital-twin environment and controllers for joint energy-water management.

Plant (what really happens, hourly):
  energy: PV + wind -> farm load (critical) -> pump -> battery -> curtailment; battery covers load shortfalls; load shed
          if the battery is empty.
  water:  pump fills the tank; drip irrigation releases water from the tank to the field.
  soil:   FAO-56 daily root-zone depletion with water-stress coefficient Ks; seasonal yield ratio by the FAO-33
          Ky relation 1 - Ya/Ym = Ky (1 - ETa/ETc), with p and Ky calibrated per site against AquaCrop (deficit_schedule,
          replay_ratio); daily yield loss valued in USD (Site.set_yield_value). Final yields are recomputed with AquaCrop.

Controllers decide, each hour, pump power and drip release (battery and curtailment follow from the energy balance):
  RuleController       - pump on renewable surplus; irrigate to refill when depletion exceeds RAW (FAO-56 practice).
  TwoStageController   - irrigation from the FAO-56 rule first, then a 48 h LP (MPC) for energy only.
  MPCController        - joint 48 h LP over pump, battery, release and soil water, re-planned daily; terminal values on
                         battery energy, tank water and soil depletion (the DRL agent of the hybrid sets these).
Forecasts come from forecast.py (or are perfect).
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

RHO_W, G = 1000.0, 9.81


# ---------------------------------------------------------------- site container
class Site:
    """All hourly/daily inputs of one site on a common local-solar-day calendar.
    hourly: DataFrame (UTC index) with pv_kw_per_kwp, wind_kw_per_kw, farm_load_kw.
    daily: DataFrame (date index) with et0_mm, rain_mm. design: dict pv_kwp, wind_kwr, batt_kwh, tank_m3, pump_kw.
    soil: dict theta_fc, theta_wp. sow_mmdd: maize sowing date."""

    def __init__(self, name, lon, hourly, daily, design, soil, sow_mmdd, head_m=60.0, area_ha=5.0, eta_pump=0.55,
                 drip_eff=0.90, zr_min=0.3, zr_max=1.2, p_depl=0.55, ky=1.25, stages=(25, 40, 45, 30), kc=(0.30, 1.20, 0.35)):
        self.name, self.lon = name, lon
        off = pd.Timedelta(hours=int(round(lon / 15)))
        loc = hourly.index.tz_localize(None) + off
        self.h = hourly.copy(); self.h.index = loc                 # local solar time
        self.d = daily.copy()
        self.design, self.soil, self.sow = design, soil, sow_mmdd
        self.head_m, self.area_ha, self.eta_pump, self.drip_eff = head_m, area_ha, eta_pump, drip_eff
        self.kwh_per_m3 = RHO_W * G * head_m / eta_pump / 3.6e6
        self.zr_min, self.zr_max, self.p, self.ky, self.stages, self.kc = zr_min, zr_max, p_depl, ky, stages, kc
        self.taw_per_m = 1000.0 * (soil["theta_fc"] - soil["theta_wp"])   # mm per m of root depth
        self.V = self.etcS = None                                          # set by set_yield_value

    def set_yield_value(self, price_usd_t, ymax_t_ha, years):
        """Value the twin's yield loss in USD: V = price x Ymax x area (full-season crop value) and etcS = mean seasonal
        ETc over 'years'. Yield-loss value of a day = V Ky (1 - Ks) ETc_day / etcS, so that the season total equals
        V Ky (1 - ETa/ETc) when the season's ETc equals its climatology (FAO-33 relation, linear in Ks)."""
        self.price, self.ymax = float(price_usd_t), float(ymax_t_ha)
        self.V = self.price * self.ymax * self.area_ha
        tot = []
        for y in years:
            c = self.season(y)
            et0 = self.d.et0_mm.reindex(c.index)
            if et0.notna().all():
                tot.append(float((c.kc * et0).sum()))
        self.etcS = float(np.mean(tot))
        return self

    def season(self, year):
        """Daily crop table for one season: date, Kc, Zr, TAW, RAW."""
        L = sum(self.stages); ini, dev, mid, late = self.stages
        days = pd.date_range(pd.Timestamp(f"{year}-{self.sow}"), periods=L, freq="D")
        k = np.empty(L); zr = np.empty(L)
        for i in range(L):
            if i < ini:
                k[i] = self.kc[0]
            elif i < ini + dev:
                k[i] = self.kc[0] + (self.kc[1] - self.kc[0]) * (i - ini) / dev
            elif i < ini + dev + mid:
                k[i] = self.kc[1]
            else:
                k[i] = self.kc[1] + (self.kc[2] - self.kc[1]) * (i - ini - dev - mid) / late
            zr[i] = self.zr_min + (self.zr_max - self.zr_min) * min(1.0, i / (ini + dev))
        taw = self.taw_per_m * zr
        return pd.DataFrame({"kc": k, "zr": zr, "taw": taw, "raw": self.p * taw}, index=days)


def peff(rain_mm):
    return 0.8 * max(rain_mm - 5.0, 0.0)


COST = dict(shed=50.0, water=0.05, cyc=0.02)   # USD per kWh shed, per m3 released, per kWh charged (same in the MPC)


def energy_cost(shed_kwh, water_m3, throughput_kwh, c=COST):
    """Daily energy-and-water part of the operating cost (USD)."""
    return c["shed"] * shed_kwh + c["water"] * water_m3 + c["cyc"] * throughput_kwh


def op_cost(shed_kwh, loss_usd, water_m3, throughput_kwh, c=COST):
    """Daily operating cost used to score every controller and as the (negative) DRL reward: energy and water cost plus
    the value of the yield lost that day according to the twin's (calibrated) crop model."""
    return energy_cost(shed_kwh, water_m3, throughput_kwh, c) + loss_usd


def _soil_day(site, c, dr, et0, rain_mm, irr_mm):
    """One FAO-56 root-zone day: returns (new depletion, Ks, ETc, ETa). Ks uses the depletion at the start of the day."""
    ks = 1.0 if dr <= c.raw else max(0.0, (c.taw - dr) / ((1 - site.p) * c.taw))
    etc = c.kc * et0; eta = ks * etc
    return min(max(dr - peff(rain_mm) - irr_mm + eta, 0.0), c.taw), ks, etc, eta


def deficit_schedule(site, year, trigger, refill=1.0):
    """Net irrigation (mm/day, Series over the season) of a soil-only deficit rule, used to calibrate the twin's crop
    model against AquaCrop: when the expected end-of-day depletion exceeds trigger x TAW, apply refill x that depletion.
    The schedule is generated once with the site's current p (before calibration) and then held fixed while p and Ky
    are fitted by replaying it (replay_ratio)."""
    crop = site.season(year); dr = 0.0; out = []
    for day, c in crop.iterrows():
        et0, rain = float(site.d.at[day, "et0_mm"]), float(site.d.at[day, "rain_mm"])
        want = dr + c.kc * et0
        irr = refill * want if want > trigger * c.taw else 0.0
        dr = _soil_day(site, c, dr, et0, rain, irr)[0]
        out.append(irr)
    return pd.Series(out, index=crop.index, name="irr_mm")


def replay_ratio(site, year, irr_mm, p, ky):
    """Twin yield ratio 1 - Ky (1 - ETa/ETc) of a given net irrigation schedule under stress parameters (p, Ky)."""
    p0 = site.p; site.p = p
    try:
        crop = site.season(year); dr = 0.0; E_ = C_ = 0.0
        for day, c in crop.iterrows():
            dr, _, etc, eta = _soil_day(site, c, dr, float(site.d.at[day, "et0_mm"]), float(site.d.at[day, "rain_mm"]),
                                        float(irr_mm.get(day, 0.0)))
            E_ += eta; C_ += etc
    finally:
        site.p = p0
    return max(1 - ky * (1 - E_ / C_), 0.0)


# ---------------------------------------------------------------- plant: day-stepping simulator (used by run_year and DRL)
class Plant:
    """Hour-by-hour plant for one calendar year, advanced one local day at a time.
    reset(year) -> state dict; step_day(plan) applies a 24 h plan (pump_kw, release_m3, allow_batt_pump, batt_floor),
    returns the day's metrics (shed_kwh, stress_mm, water_m3, throughput_kwh, ...) and updates the soil at day end."""

    def __init__(self, site, soc0=0.5, tank0=0.5):
        self.site, self.soc0, self.tank0 = site, soc0, tank0

    def reset(self, year):
        s, D = self.site, self.site.design
        self.B, self.T, self.P = D["batt_kwh"], D["tank_m3"], D["pump_kw"]
        self.soc_min, self.soc_max, self.eta_ch, self.eta_dis, self.c_rate = 0.10 * self.B, 0.90 * self.B, 0.96, 0.96, 0.5 * self.B
        self.soc = self.soc_min + self.soc0 * (self.soc_max - self.soc_min); self.tank = self.tank0 * self.T
        self.year, self.crop, self.dr = year, s.season(year), 0.0
        hh = s.h.loc[f"{year}-01-01":f"{year}-12-31 23:00"]
        full = hh.groupby(hh.index.normalize()).size()               # keep complete local days only (data edges)
        hh = hh[hh.index.normalize().isin(full[full == 24].index)]
        self.hidx = hh.index
        self.gen = (hh.pv_kw_per_kwp.values * D["pv_kwp"] + hh.wind_kw_per_kw.values * D["wind_kwr"])
        self.load = hh.farm_load_kw.values
        self.days = hh.index.normalize().unique()
        self.k = 0                                                   # index of the next day to simulate
        self.rec, self.seas = [], []
        self.cost_tot = self.energy_tot = self.loss_tot = 0.0
        self.eta_tot = self.etc_tot = 0.0
        assert s.V is not None, "call site.set_yield_value(...) first"
        return self.state()

    def state(self):
        return dict(soc=self.soc, tank=self.tank, dr=self.dr)

    @property
    def done(self):
        return self.k >= len(self.days)

    @property
    def today(self):
        return self.days[self.k]

    def step_day(self, plan):
        s, P, T = self.site, self.P, self.T
        day = self.days[self.k]; i0 = self.k * 24
        shed_day = rel_day = thr_day = cur_day = pump_day = 0.0
        for hr in range(24):
            i = i0 + hr
            g, L = self.gen[i], self.load[i]
            rel = min(plan["release_m3"][hr], self.tank); self.tank -= rel; rel_day += rel
            served = min(g, L); g -= served; short = L - served
            if short > 0:
                dis = min(short, max(self.soc - self.soc_min, 0) * self.eta_dis, self.c_rate); self.soc -= dis / self.eta_dis; short -= dis
            room_kw = max(T - self.tank, 0.0) * s.kwh_per_m3
            want = min(plan["pump_kw"][hr], P, room_kw)
            if 0 < want < 0.3 * P and room_kw >= 0.3 * P:            # small plans are rounded up to the VFD minimum
                want = 0.3 * P
            p_pump = min(g, want)
            if want > p_pump and plan.get("allow_batt_pump", True):
                top = min(want - p_pump, max(self.soc - plan.get("batt_floor", self.soc_min), 0) * self.eta_dis, self.c_rate)
                self.soc -= top / self.eta_dis; p_pump += top
            if p_pump < 0.3 * P:                                     # VFD minimum power
                if p_pump > min(g, want):                            # give back the battery share
                    self.soc += (p_pump - min(g, want)) / self.eta_dis
                p_pump = 0.0 if min(g, want) < 0.3 * P else min(g, want)
            g -= min(g, p_pump)
            ch = min(g, self.c_rate, max(self.soc_max - self.soc, 0) / self.eta_ch); self.soc += ch * self.eta_ch
            cur = g - ch
            self.tank += p_pump / s.kwh_per_m3
            self.rec.append((self.gen[i], L, short, p_pump, ch, cur, self.soc, self.tank, rel))
            shed_day += short; thr_day += ch; cur_day += cur; pump_day += p_pump
        stress_mm = loss = 0.0; ks = 1.0
        if day in self.crop.index:                                   # end of day: soil water balance
            c = self.crop.loc[day]
            et0, rain = float(s.d.at[day, "et0_mm"]), float(s.d.at[day, "rain_mm"])
            irr_mm = rel_day * s.drip_eff / (10.0 * s.area_ha)
            self.dr, ks, etc, eta = _soil_day(s, c, self.dr, et0, rain, irr_mm)
            self.etc_tot += etc; self.eta_tot += eta
            stress_mm = max(self.dr - c.raw, 0.0)
            loss = s.V * s.ky * (1 - ks) * etc / s.etcS
            self.seas.append((day, et0, peff(rain), irr_mm, self.dr, ks, etc, eta, c.taw, c.raw))
        self.k += 1
        e_cost = energy_cost(shed_day, rel_day, thr_day)
        cost = e_cost + loss
        self.cost_tot += cost; self.energy_tot += e_cost; self.loss_tot += loss
        return dict(day=day, shed_kwh=shed_day, stress_mm=stress_mm, ks=ks, water_m3=rel_day, throughput_kwh=thr_day,
                    curtail_kwh=cur_day, pump_kwh=pump_day, cost=cost, loss_usd=loss,
                    season_end=bool(len(self.crop) and day == self.crop.index[-1]))

    def season_irrigation(self):
        """Net irrigation (mm/day) applied so far this season, as a Series for AquaCrop."""
        return pd.Series({r[0]: r[3] for r in self.seas}, name="irr_mm")

    def summary(self):
        s = self.site
        H = pd.DataFrame(self.rec, index=self.hidx[:len(self.rec)],
                         columns=["gen_kw", "load_kw", "shed_kw", "pump_kw", "charge_kw", "curtail_kw", "soc_kwh", "tank_m3", "release_m3"])
        S = pd.DataFrame(self.seas, columns=["day", "et0", "peff", "irr_mm", "dr", "ks", "etc", "eta", "taw", "raw"]).set_index("day")
        e, c = self.eta_tot, self.etc_tot
        ratio = 1 - s.ky * (1 - e / c) if c > 0 else np.nan
        summ = dict(year=self.year, load_kwh=H.load_kw.sum(), shed_kwh=H.shed_kw.sum(), elpsp=H.shed_kw.sum() / H.load_kw.sum(),
                    pump_kwh=H.pump_kw.sum(), curtail_kwh=H.curtail_kw.sum(), gen_kwh=H.gen_kw.sum(), batt_throughput_kwh=H.charge_kw.sum(),
                    irrigation_mm=S.irr_mm.sum(), irrigation_m3=H.release_m3.sum(), eta_mm=e, etc_mm=c,
                    eta_etc=e / c if c else np.nan, yield_ratio=max(ratio, 0.0), stress_days=int((S.ks < 1).sum()),
                    op_cost=self.cost_tot, energy_cost=self.energy_tot, twin_loss_usd=self.loss_tot)
        return H, S, summ


def run_year(site, year, controller, fc=None, soc0=0.5, tank0=0.5):
    """Simulate one calendar year under a controller (daily plans applied hour by hour).
    fc: forecasts dict or None (perfect information). Returns (hourly DataFrame, seasonal soil DataFrame, summary)."""
    plant = Plant(site, soc0, tank0)
    plant.reset(year)
    controller.reset(site, year)
    while not plant.done:
        plan = controller.plan_day(site, plant.today, plant.state(), fc)
        m = plant.step_day(plan)
        controller.observe(m["day"], plant.dr)
    return plant.summary()


# ---------------------------------------------------------------- forecasts lookup
def make_forecast_lookup(site, fc_hourly=None, fc_daily=None):
    """Build day -> forecast arrays. fc_hourly: dict var -> long frame from forecast.forecast_hourly (local days);
    fc_daily: dict var -> long frame from forecast.forecast_daily. None -> perfect forecasts (actuals)."""
    return dict(hourly=fc_hourly, daily=fc_daily)


def _hourly_fc(site, day, fc, nh=48):
    """48 h forecast of generation (kW) and load (kW) from local midnight of 'day'."""
    D = site.design
    idx = pd.date_range(day, periods=nh, freq="h")
    act = site.h.reindex(idx)
    if fc is None or fc.get("hourly") is None:
        pv, wd, ld = act.pv_kw_per_kwp.values, act.wind_kw_per_kw.values, act.farm_load_kw.values
    else:
        out = {}
        for var, col in (("pv", "pv_kw_per_kwp"), ("wind", "wind_kw_per_kw"), ("load", "farm_load_kw")):
            f = fc["hourly"][var]
            v = np.empty(nh)
            for lead in (1, 2):
                dd = day + pd.Timedelta(days=lead - 1)
                r = f.get((dd, lead))
                v[(lead - 1) * 24:lead * 24] = r if r is not None else act[col].values[(lead - 1) * 24:lead * 24]
            out[var] = v
        pv, wd, ld = out["pv"], out["wind"], out["load"]
    gen = np.nan_to_num(pv) * D["pv_kwp"] + np.nan_to_num(wd) * D["wind_kwr"]
    return gen, np.nan_to_num(ld)


def _daily_fc(site, day, fc, nd=2):
    days = [day + pd.Timedelta(days=k) for k in range(nd)]
    if fc is None or fc.get("daily") is None:
        et0 = [float(site.d.et0_mm.get(x, 0.0)) for x in days]; rain = [float(site.d.rain_mm.get(x, 0.0)) for x in days]
    else:
        et0 = [fc["daily"]["et0"].get((x, k + 1), float(site.d.et0_mm.get(x, 0.0))) for k, x in enumerate(days)]
        rain = [fc["daily"]["rain"].get((x, k + 1), 0.0) for k, x in enumerate(days)]
    return days, np.array(et0), np.array(rain)


# ---------------------------------------------------------------- controllers
class RuleController:
    """Pump at the renewable surplus left after the farm load (no battery for pumping); irrigate when depletion exceeds
    RAW, refilling to field capacity (FAO-56 practice), released over the daytime drip window 06-18 h."""
    name = "rule"

    def reset(self, site, year):
        self.crop = site.season(year); self.dr = 0.0

    def observe(self, day, dr):
        self.dr = dr

    def irrigation_need_m3(self, site, day):
        if day not in self.crop.index:
            return 0.0
        c = self.crop.loc[day]
        et0 = float(site.d.at[day, "et0_mm"]) if day in site.d.index else 0.0
        dr_next = self.dr + c.kc * et0                                # expected depletion by end of today
        if dr_next <= c.raw:
            return 0.0
        return dr_next / site.drip_eff * 10.0 * site.area_ha

    def plan_day(self, site, day, state, fc):
        need = self.irrigation_need_m3(site, day)
        rel = np.zeros(24); rel[6:18] = need / 12.0
        return dict(pump_kw=np.full(24, site.design["pump_kw"]), release_m3=rel, allow_batt_pump=False)


C_DELIVER = 5.0    # USD per m3 of two-stage planned irrigation delivered (below the 50 USD/kWh shed cost / ~0.3 kWh/m3)


class MPCController:
    """Joint 48 h linear programme, solved at local midnight; the first 24 h of the plan are applied.
    Decision variables per hour: pump power, battery charge/discharge, curtailment, load shed, drip release, battery
    reserve shortfall; daily soil depletion with drainage and stress slack; terminal depletion excess. Objective (minimise):
        c_shed*shed + c_stress_d*stress + c_water*(release + drainage) + c_cyc*(charge+discharge) + c_cur*curtail
        + w_res*reserve_shortfall - w_soc*SoC_end - w_tank*tank_end + w_dr*max(depletion_end - dr_target*RAW_end, 0)
    c_stress_d = stress_mult * V Ky ETc_d / ((1-p) TAW_d etcS) is the yield-loss value (USD) of 1 mm of depletion beyond
    RAW on day d, from the same (calibrated) crop model the plant is scored with (Site.set_yield_value).
    Three operating parameters shape the plan: reserve (battery fraction held back for the farm load; the plant also
    refuses battery pumping below it), dr_target (end-of-horizon depletion target as a fraction of RAW) and stress_mult.
    They are tuned once on training years (tune_mpc) or set each day by the hybrid DRL agent (set_params).
    Tiny time-dependent tie-breaking costs make the optimum unique, so results do not depend on the solver version.
    fixed_release: if given (array of 24 h), irrigation is planned outside the LP (two-stage mode): the release is bounded
    by the plan and every delivered m3 earns C_DELIVER, so the plan is met unless that would shed farm load (a hard
    equality made the LP infeasible on ~20% of days, which then fell back to the rule plan)."""
    name = "mpc"

    def __init__(self, c_shed=50.0, stress_mult=1.0, c_cyc=0.02, c_cur=0.0, c_water=0.05, w_soc=0.10, w_tank=0.05, w_dr=5.0,
                 reserve=0.0, dr_target=1.0, w_res=0.5, horizon_h=48, perfect=False, name=None):
        # costs in USD, as in env.COST: 1 kWh of shed farm load = 50; 1 m3 of irrigation water = 0.05 (2.8 per net mm on
        # 5 ha); deep drainage is charged as the water it wastes; stress at its yield-loss value (see class docstring).
        self.c = dict(shed=c_shed, cyc=c_cyc, cur=c_cur, water=c_water)
        self.w = dict(soc=w_soc, tank=w_tank, dr=w_dr, res=w_res)
        self.reserve, self.dr_target, self.stress_mult = reserve, dr_target, stress_mult
        self.H, self.perfect = horizon_h, perfect
        self.name = name or ("mpc_perfect" if perfect else "mpc")
        self.last_status = None

    def reset(self, site, year):
        self.crop = site.season(year); self.dr = 0.0

    def observe(self, day, dr):
        self.dr = dr

    def set_params(self, reserve=None, dr_target=None, stress_mult=None):
        if reserve is not None:
            self.reserve = float(reserve)
        if dr_target is not None:
            self.dr_target = float(dr_target)
        if stress_mult is not None:
            self.stress_mult = float(stress_mult)

    def plan_day(self, site, day, state, fc, fixed_release=None):
        H, D = self.H, site.design
        nd = H // 24
        gen, load = _hourly_fc(site, day, None if self.perfect else fc, H)
        days, et0, rain = _daily_fc(site, day, None if self.perfect else fc, nd)
        B, T, P = D["batt_kwh"], D["tank_m3"], D["pump_kw"]
        smin, smax, ec, ed, cmax = 0.10 * B, 0.90 * B, 0.96, 0.96, 0.5 * B
        s_res = smin + self.reserve * (smax - smin)
        k = site.kwh_per_m3; mm_per_m3 = site.drip_eff / (10.0 * site.area_ha)
        # variable layout: per hour [pump, ch, dis, cur, shed, rel, res_short] ; soc[0..H] ; tank[0..H] ;
        # per day [dr_end, drain, stress] ; terminal depletion excess
        nv_h = 7
        i_soc = H * nv_h; i_tank = i_soc + H + 1; i_day = i_tank + H + 1; i_term = i_day + 3 * nd; nvar = i_term + 1
        hv = lambda t, j: t * nv_h + j
        lb = np.zeros(nvar); ub = np.full(nvar, np.inf)
        for t in range(H):
            ub[hv(t, 0)] = P; ub[hv(t, 1)] = cmax; ub[hv(t, 2)] = cmax; ub[hv(t, 4)] = load[t]
            ub[hv(t, 5)] = (fixed_release[t] if (fixed_release is not None and t < 24) else (0.0 if fixed_release is not None else np.inf))
        lb[i_soc:i_soc + H + 1] = smin; ub[i_soc:i_soc + H + 1] = smax
        lb[i_soc] = ub[i_soc] = min(max(state["soc"], smin), smax)
        ub[i_tank:i_tank + H + 1] = T; lb[i_tank] = ub[i_tank] = min(state["tank"], T)
        rows, cols, vals, beq = [], [], [], []
        r = 0
        def add(row, col, val):
            rows.append(row); cols.append(col); vals.append(val)
        for t in range(H):
            # energy: pump + ch + cur - dis - shed = gen - load
            add(r, hv(t, 0), 1); add(r, hv(t, 1), 1); add(r, hv(t, 3), 1); add(r, hv(t, 2), -1); add(r, hv(t, 4), -1)
            beq.append(gen[t] - load[t]); r += 1
            # battery: soc[t+1] - soc[t] - ec*ch + dis/ed = 0
            add(r, i_soc + t + 1, 1); add(r, i_soc + t, -1); add(r, hv(t, 1), -ec); add(r, hv(t, 2), 1 / ed); beq.append(0.0); r += 1
            # tank: tank[t+1] - tank[t] - pump/k + rel = 0
            add(r, i_tank + t + 1, 1); add(r, i_tank + t, -1); add(r, hv(t, 0), -1 / k); add(r, hv(t, 5), 1); beq.append(0.0); r += 1
        # soil (daily): dr_end_d = dr_start_d + Kc*ET0 - Peff - mm_per_m3*sum(rel_d) + drain_d   (drain >= 0 lets dr stay >= 0)
        in_season = [x in self.crop.index for x in days]
        for d in range(nd):
            kc = float(self.crop.at[days[d], "kc"]) if in_season[d] else 0.0
            add(r, i_day + 3 * d, 1)
            if d > 0:
                add(r, i_day + 3 * (d - 1), -1)
            for t in range(24 * d, 24 * (d + 1)):
                add(r, hv(t, 5), mm_per_m3 if in_season[d] else 0.0)
            add(r, i_day + 3 * d + 1, -1)
            beq.append((self.dr if d == 0 else 0.0) + kc * et0[d] - peff(rain[d])); r += 1
        A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nvar))
        # stress: stress_d >= dr_end_d - RAW_d
        ri, ci, vi, bub = [], [], [], []
        r = 0
        for d in range(nd):
            raw = float(self.crop.at[days[d], "raw"]) if in_season[d] else 1e6
            taw = float(self.crop.at[days[d], "taw"]) if in_season[d] else 1e6
            ri += [r, r]; ci += [i_day + 3 * d, i_day + 3 * d + 2]; vi += [1, -1]; bub.append(raw); r += 1
            ri += [r]; ci += [i_day + 3 * d]; vi += [1]; bub.append(taw); r += 1      # depletion cannot exceed TAW
        # terminal: excess >= dr_end_last - dr_target * RAW_last
        raw_end = float(self.crop.at[days[-1], "raw"]) if in_season[-1] else 1e6
        ri += [r, r]; ci += [i_day + 3 * (nd - 1), i_term]; vi += [1, -1]; bub.append(self.dr_target * raw_end); r += 1
        # battery reserve (soft): soc[t+1] + short_t >= s_res
        for t in range(H):
            ri += [r, r]; ci += [i_soc + t + 1, hv(t, 6)]; vi += [-1, -1]; bub.append(-s_res); r += 1
        A_ub = sparse.csr_matrix((vi, (ri, ci)), shape=(r, nvar))
        cvec = np.zeros(nvar)
        for t in range(H):
            eps = 1e-6 * (t + 1)                                     # tie-breakers: earlier action preferred, unique optimum
            cvec[hv(t, 1)] = cvec[hv(t, 2)] = self.c["cyc"] + eps; cvec[hv(t, 3)] = self.c["cur"] + 2 * eps
            cvec[hv(t, 4)] = self.c["shed"]
            cvec[hv(t, 0)] = 1e-4 + eps                              # do not pump without reason
            cvec[hv(t, 5)] = self.c["water"] + eps                   # irrigation water is scarce
            if fixed_release is not None:                            # two-stage: deliver the fixed plan where possible
                cvec[hv(t, 5)] -= C_DELIVER
            cvec[hv(t, 6)] = self.w["res"]
        for d in range(nd):
            if in_season[d]:
                kc, taw = float(self.crop.at[days[d], "kc"]), float(self.crop.at[days[d], "taw"])
                cvec[i_day + 3 * d + 2] = self.stress_mult * site.V * site.ky * kc * et0[d] / ((1 - site.p) * taw * site.etcS)
            cvec[i_day + 3 * d + 1] = self.c["water"] / mm_per_m3    # deep drainage wastes the water it carries
        cvec[i_soc + H] -= self.w["soc"]; cvec[i_tank + H] -= self.w["tank"]; cvec[i_term] = self.w["dr"]
        res = linprog(cvec, A_ub=A_ub, b_ub=bub, A_eq=A_eq, b_eq=beq, bounds=list(zip(lb, ub)), method="highs")
        self.last_status = res.status
        if res.status != 0:                                          # fall back to the rule plan if the LP fails
            return RuleController.plan_day(self, site, day, state, fc) if hasattr(self, "crop") else None
        x = res.x
        pump = np.array([x[hv(t, 0)] for t in range(24)]); rel = np.array([x[hv(t, 5)] for t in range(24)])
        dis_plan = np.array([x[hv(t, 2)] for t in range(24)])
        return dict(pump_kw=pump, release_m3=rel, allow_batt_pump=True, batt_floor=s_res, plan_dis=dis_plan)

    irrigation_need_m3 = RuleController.irrigation_need_m3


class TwoStageController(MPCController):
    """Decoupled baseline: stage 1 fixes irrigation by the FAO-56 rule; stage 2 optimises energy with the same LP."""

    def __init__(self, **kw):
        super().__init__(**kw); self.name = "two_stage"

    def plan_day(self, site, day, state, fc, fixed_release=None):
        need = self.irrigation_need_m3(site, day)
        rel = np.zeros(24); rel[6:18] = need / 12.0
        return super().plan_day(site, day, state, fc, fixed_release=rel)


# v2 tuning (2014-16) never preferred reserve < 0.5; the v2 optimum sat on the upper edges (reserve 0.9, target 1.0),
# so the grid is shifted up and widened.
TUNE_GRID = dict(reserve=(0.5, 0.75, 0.9, 1.0), dr_target=(0.5, 0.75, 1.0, 1.25, 1.5))


def tune_mpc(site, years, fc, grid=TUNE_GRID):
    """Fixed-parameter MPC baseline: pick (reserve, dr_target) with the lowest mean operating cost over training years
    (forecasts for those years are out-of-sample by blocked CV). Returns (best dict, table of all combinations)."""
    rows = []
    for res_ in grid["reserve"]:
        for tgt in grid["dr_target"]:
            costs = [run_year(site, y, MPCController(reserve=res_, dr_target=tgt), fc)[2]["op_cost"] for y in years]
            rows.append(dict(reserve=res_, dr_target=tgt, op_cost=float(np.mean(costs))))
    tab = pd.DataFrame(rows).sort_values("op_cost").reset_index(drop=True)
    return dict(reserve=float(tab.reserve[0]), dr_target=float(tab.dr_target[0])), tab
