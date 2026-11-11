"""DRL environments (Gymnasium) on top of env.Plant, one decision per local day.

HybridEnv  (proposed hybrid DRL-MPC): action a in [-1, 1]^3 moves the MPC operating parameters around the values tuned
           on the training years (a = 0 reproduces the tuned MPC exactly): battery reserve +/-0.4 (0-1), end-of-horizon
           depletion target +/-0.5 of RAW (0.1-1.5), stress weight x10**(0.5 a); the MPC then plans the day and the plant
           runs it.
DRLOnlyEnv (model-free DRL baseline): action a in [-1, 1]^3 sets the day's drip release as a fraction (0-2x) of the
           refill need (from the forecast ET0), the daytime pump target (0-100% of pump power, 08-17 h) and whether the
           battery may assist the pump.

Reward = -op_cost(day) / REWARD_SCALE, the same daily cost used to score every controller (env.op_cost: energy, water
and the twin's yield-loss value). With aq (AquaCrop in the loop), the last day of the season also carries the
correction twin_loss - AquaCrop_loss, so that over a season the agent is rewarded for the AquaCrop yield, the signal the
MPC's linear crop model cannot see. aq = dict(fn=callable(year, irr_mm Series) -> t/ha, ymax={year: t/ha}).
Observation (14 values): day-of-year sin/cos, in-season flag, Kc, depletion/TAW, RAW/TAW, SoC fraction, tank fraction,
forecast generation (next 24 h, 24-48 h) and load (next 24 h) normalised by design size, forecast ET0 day 1-2 / 10, and
forecast rain day 1 / 10.
"""
import numpy as np

import env as E

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:                     # lets the local test run without gymnasium
    gym = spaces = None
    _Base = object

REWARD_SCALE = 100.0


def observation(site, plant, fc, crop):
    day = plant.today
    D = site.design
    gen, load = E._hourly_fc(site, day, fc, 48)
    _, et0, rain = E._daily_fc(site, day, fc, 2)
    cap = max(D["pv_kwp"] + D["wind_kwr"], 1e-6)
    in_s = day in crop.index
    kc = float(crop.at[day, "kc"]) if in_s else 0.0
    taw = float(crop.at[day, "taw"]) if in_s else 1.0
    raw = float(crop.at[day, "raw"]) if in_s else 1.0
    doy = day.dayofyear
    return np.array([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), float(in_s), kc,
                     plant.dr / taw, raw / taw,
                     (plant.soc - plant.soc_min) / max(plant.soc_max - plant.soc_min, 1e-6), plant.tank / max(plant.T, 1e-6),
                     gen[:24].sum() / (24 * cap), gen[24:].sum() / (24 * cap), load[:24].sum() / (24 * max(D["pump_kw"], 1e-6)),
                     et0[0] / 10.0, et0[1] / 10.0, rain[0] / 10.0], dtype=np.float32)


class _DailyEnv(_Base):
    """Common episode logic: an episode is one calendar year drawn from 'years' (in order if sequential=True)."""
    metadata = {"render_modes": []}

    def __init__(self, site, years, fc, seed=0, sequential=False, aq=None):
        self.site, self.years, self.fc, self.sequential, self.aq = site, list(years), fc, sequential, aq
        self.rng = np.random.default_rng(seed)
        self.plant = E.Plant(site)
        self._i = -1
        if spaces is not None:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
            self.observation_space = spaces.Box(-10.0, 10.0, shape=(14,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.sequential:
            self._i = (self._i + 1) % len(self.years); y = self.years[self._i]
        else:
            y = int(self.rng.choice(self.years))
        self.plant.reset(y)
        self.year = y
        self.crop = self.site.season(y)
        self._reset_controller(y)
        return observation(self.site, self.plant, self.fc, self.crop), {"year": y}

    def step(self, action):
        plan = self._plan(np.clip(np.asarray(action, dtype=float), -1, 1))
        m = self.plant.step_day(plan)
        self._observe(m)
        cost = m["cost"]
        if self.aq is not None and m["season_end"]:
            s = self.site
            y_aq = float(self.aq["fn"](self.year, self.plant.season_irrigation()))
            aq_loss = s.price * s.area_ha * (self.aq["ymax"][self.year] - y_aq)
            cost += aq_loss - self.plant.loss_tot                    # season total becomes the AquaCrop yield-loss value
            m = dict(m, aq_yield_t_ha=y_aq, aq_loss_usd=aq_loss)
        done = self.plant.done
        obs = (observation(self.site, self.plant, self.fc, self.crop) if not done else np.zeros(14, dtype=np.float32))
        return obs, -cost / REWARD_SCALE, done, False, m


class HybridEnv(_DailyEnv):
    def __init__(self, site, years, fc, mpc_kwargs=None, **kw):
        super().__init__(site, years, fc, **kw)
        self.mpc = E.MPCController(**(mpc_kwargs or {}))
        self.p0 = dict(reserve=self.mpc.reserve, dr_target=self.mpc.dr_target, stress_mult=self.mpc.stress_mult)

    def _reset_controller(self, y):
        self.mpc.reset(self.site, y)

    def _plan(self, a):
        self.mpc.set_params(reserve=float(np.clip(self.p0["reserve"] + 0.4 * a[0], 0.0, 1.0)),
                            dr_target=float(np.clip(self.p0["dr_target"] + 0.5 * a[1], 0.1, 1.5)),
                            stress_mult=self.p0["stress_mult"] * 10.0 ** (0.5 * a[2]))
        return self.mpc.plan_day(self.site, self.plant.today, self.plant.state(), self.fc)

    def _observe(self, m):
        self.mpc.observe(m["day"], self.plant.dr)


class DRLOnlyEnv(_DailyEnv):
    def _reset_controller(self, y):
        self.rule = E.RuleController(); self.rule.reset(self.site, y)

    def _plan(self, a):
        s, day = self.site, self.plant.today
        need = 0.0
        if day in self.crop.index:                                   # refill need from the current depletion and forecast ETc
            et0 = float(E._daily_fc(s, day, self.fc, 1)[1][0])
            need = (self.plant.dr + float(self.crop.at[day, "kc"]) * et0) / s.drip_eff * 10.0 * s.area_ha
        rel = np.zeros(24); rel[6:18] = (a[0] + 1) / 2 * 2.0 * need / 12.0
        pump = np.zeros(24); pump[8:18] = (a[1] + 1) / 2 * s.design["pump_kw"]
        return dict(pump_kw=pump, release_m3=rel, allow_batt_pump=bool(a[2] > 0), batt_floor=self.plant.soc_min)

    def _observe(self, m):
        self.rule.observe(m["day"], self.plant.dr)


class PolicyController:
    """Wraps a trained policy (anything with .predict(obs, deterministic=True)) as a controller for env.run_year."""

    def __init__(self, policy, kind, name, mpc_kwargs=None):
        self.policy, self.kind, self.name, self.mpc_kwargs = policy, kind, name, mpc_kwargs

    def reset(self, site, year):
        self.site, self.year = site, year
        self.crop = site.season(year)
        self.inner = (HybridEnv(site, [year], None, mpc_kwargs=self.mpc_kwargs) if self.kind == "hybrid"
                      else DRLOnlyEnv(site, [year], None))
        self.inner.crop = self.crop
        self.inner._reset_controller(year)

    def observe(self, day, dr):
        self.inner.plant.dr = dr                                     # real end-of-day depletion from the plant
        self.inner._observe({"day": day})

    def plan_day(self, site, day, state, fc):
        self.inner.fc = fc
        plant = _PlantView(site, day, state)
        obs = observation(site, plant, fc, self.crop)
        a, _ = self.policy.predict(obs, deterministic=True)
        self.inner.plant = plant
        return self.inner._plan(np.clip(np.asarray(a, dtype=float), -1, 1))


class _PlantView:
    """Minimal read-only plant view (today, state, limits) for computing observations inside run_year."""

    def __init__(self, site, day, state):
        D = site.design
        self.today, self.dr, self.soc, self.tank = day, state["dr"], state["soc"], state["tank"]
        self.soc_min, self.soc_max, self.T = 0.10 * D["batt_kwh"], 0.90 * D["batt_kwh"], D["tank_m3"]

    def state(self):
        return dict(soc=self.soc, tank=self.tank, dr=self.dr)
