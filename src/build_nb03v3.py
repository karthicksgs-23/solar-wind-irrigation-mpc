"""Build notebooks/03_controllers_drl_mpc_v3.ipynb (Colab). Modules are NOT embedded: they are read from Drive
solar_wind_irrigation/src_v3 (forecast.py, env.py, drl.py, aquacrop_yield.py), and the hourly/daily inputs from the
processed tables written by notebook 02 (data/processed). Usage: python build_nb03v3.py <reference op_cost>"""
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent
OUT = SRC.parent / "notebooks" / "03_controllers_drl_mpc_v3.ipynb"
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md(r"""
# Notebook 03 v3 — Crop-model calibration, yield-aware MPC, and DRL agents rewarded with AquaCrop yield

**Why v3.** In v2 the hybrid DRL–MPC equalled the tuned MPC, and the model-free DRL reached higher AquaCrop yields with more water: the controllers' internal FAO-56/Ky crop model underrated the yield loss of deficit irrigation (about 2.5× less than AquaCrop).

**Changes from v2.** (i) **Stage 0:** the twin's stress onset *p* and yield response *Ky* are calibrated per site against AquaCrop on deficit-irrigation schedules (2009–2016). (ii) The stress term of the cost is now the **yield-loss value in USD** (maize price × yield loss), in both the plant and the MPC. (iii) Both learned agents receive the **AquaCrop yield at season end** as a reward correction, the signal the MPC's linear crop model cannot use; discount 0.995. (iv) Wider tuning grid (v2 optimum sat on its edge). (v) The two-stage baseline uses the tuned battery reserve (v2 used 0). (vi) New ablation `mpc_uncal`: tuned MPC with the uncalibrated crop model. (vii) **Headline score:** total cost = energy and water cost + AquaCrop yield-loss value.

**Design.** Train 2001–2016 · validate 2017–2019 · test 2020–2024 · 3 PPO seeds · AquaCrop yields for test schedules · paired year bootstrap. CPU runtime; the AquaCrop timing printed in Stage 0 gives the expected training time. **Checkpointing.** Forecasts, calibration runs, tuning, evaluation and AquaCrop yields are saved per site, and PPO saves every 7,300 steps, all to Drive `outputs/nb03_v3`. If Colab disconnects, reconnect and use *Runtime → Run all*: finished work loads and training continues from the last checkpoint. Progress is logged with timestamps in `outputs/nb03_v3/progress.log`. The run refuses to resume if the settings or modules changed. **Output:** `export_nb03_v3.zip`.
""")

code(r"""
!pip -q install gymnasium stable-baselines3 "aquacrop>=3,<4"
import os, sys, json, time, platform, shutil, glob, importlib, copy
import numpy as np, pandas as pd, matplotlib.pyplot as plt, scipy, sklearn
from google.colab import drive, files
drive.mount('/content/drive')
BASE = '/content/drive/MyDrive/solar_wind_irrigation'
SRC = f'{BASE}/src_v3'; PROC = f'{BASE}/data/processed'; OUT = f'{BASE}/outputs/nb03_v3'; MOD = f'{OUT}/models'
for d in (OUT, MOD, f'{OUT}/schedules'): os.makedirs(d, exist_ok=True)
import hashlib
SHA = __SHA__   # sha256 of the locally tested modules (newline-normalised)
for f, h in SHA.items():
    assert os.path.exists(f'{SRC}/{f}'), f'missing {SRC}/{f}'
    got = hashlib.sha256((open(f'{SRC}/{f}', encoding='utf-8').read().replace('\r\n', '\n').rstrip('\n') + '\n').encode()).hexdigest()
    print(f, 'OK' if got == h else 'MISMATCH ' + got[:12])
    assert got == h, f'{f} differs from the tested version'
assert os.path.exists(f'{BASE}/outputs/nb02/sizing_picks.json'), 'run notebook 02 first'
sys.path.insert(0, SRC)
import forecast as fc, env as E, drl, aquacrop_yield as aq
for m in (fc, E, drl, aq): importlib.reload(m)
assert hasattr(E, 'deficit_schedule'), 'src_v3/env.py is not the v3 file'
print('scipy', scipy.__version__, '| sklearn', sklearn.__version__)
""")

code(r"""
LON = {'bushland_tx': -102.094, 'jodhpur_in': 73.02, 'souss_ma': -8.88, 'azraq_jo': 36.82}
SOW = {'bushland_tx': '05-15', 'jodhpur_in': '07-01', 'souss_ma': '03-15', 'azraq_jo': '04-01'}
SOIL = {'bushland_tx': {'theta_fc': 0.3645, 'theta_wp': 0.25}, 'jodhpur_in': {'theta_fc': 0.3491, 'theta_wp': 0.103},
        'souss_ma': {'theta_fc': 0.2846, 'theta_wp': 0.1395}, 'azraq_jo': {'theta_fc': 0.3565, 'theta_wp': 0.2193}}   # SoilGrids 0-100 cm
PICKS = json.load(open(f'{BASE}/outputs/nb02/sizing_picks.json'))
TRAIN, VAL, TEST = list(range(2001, 2017)), [2017, 2018, 2019], [2020, 2021, 2022, 2023, 2024]
TUNE_YEARS = [2014, 2015, 2016]; CAL_YEARS = list(range(2009, 2017))
SEEDS = [0, 1, 2]; HYB_STEPS = 60_000; DRL_STEPS = 150_000; N_ENVS = 2; DESIGN = 'least_cost'
PRICE = 200.0          # USD per t maize grain: ASSUMED placeholder, to be replaced by a sourced price (with the costs)
YMAX_REF = 13.0        # t/ha, only for the reproducibility check before calibration
AQ_REWARD = True; GAMMA = 0.995
# deficit schedules for calibration: (name, trigger as fraction of TAW, refill fraction of the depletion)
SCHED = [('full', 0.10, 1.0), ('t30', 0.30, 1.0), ('t45', 0.45, 1.0), ('t60', 0.60, 1.0), ('t75', 0.75, 1.0),
         ('t90', 0.90, 1.0), ('t45_r50', 0.45, 0.5), ('t30_r70', 0.30, 0.7)]

# checkpointing: every stage saves per-site parts to Drive and PPO saves every CKPT_EVERY steps, so "Run all" after a
# disconnect resumes where the run stopped. Progress lines go to outputs/nb03_v3/progress.log.
import pickle
CKPT_EVERY = 7_300     # env steps between PPO checkpoints (10 rollouts of N_ENVS x 365 days)
RESUMED = {}           # agents whose training continued from a checkpoint (not bit-identical to an uninterrupted run)

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} UTC  {msg}"
    print(line, flush=True)
    with open(f'{OUT}/progress.log', 'a') as fh:
        fh.write(line + '\n')

def save_atomic(path, write):          # write(tmp); a half-written file never replaces a good one
    tmp = path + '.part'; write(tmp); os.replace(tmp, path)

def dump_json(obj, path):
    def w(t):
        with open(t, 'w') as fh: json.dump(obj, fh, indent=1, default=str)
    save_atomic(path, w)

def dump_pkl(obj, path):
    def w(t):
        with open(t, 'wb') as fh: pickle.dump(obj, fh)
    save_atomic(path, w)

def part(stage, sid, ext='csv'):
    d = f'{OUT}/parts/{stage}'; os.makedirs(d, exist_ok=True)
    return f'{d}/{sid}.{ext}'

CFG = dict(sha=globals().get('SHA'), picks=PICKS, seeds=SEEDS, hyb_steps=HYB_STEPS, drl_steps=DRL_STEPS, n_envs=N_ENVS, design=DESIGN,
           price=PRICE, aq_reward=AQ_REWARD, gamma=GAMMA, train=TRAIN, val=VAL, test=TEST, tune_years=TUNE_YEARS, cal_years=CAL_YEARS, sched=SCHED)
CFG = json.loads(json.dumps(CFG))
p_cfg = f'{OUT}/run_config.json'
if os.path.exists(p_cfg):
    diff = [k for k, v in json.load(open(p_cfg)).items() if CFG.get(k) != v]
    assert not diff, f'{OUT} holds checkpoints from a different configuration {diff}: rename that folder or restore the settings'
    log('resuming: configuration unchanged, completed stages load from checkpoints')
else:
    dump_json(CFG, p_cfg); log('new run')
""")

md("## 1. Sites (processed tables from notebook 02) and day-ahead forecasts (blocked CV for training years)")
code(r"""
SITE, LOOK, DAILY, skill = {}, {}, {}, []
for sid in LON:
    t0 = time.time()
    h = pd.read_parquet(f'{PROC}/{sid}_hourly.parquet'); d = pd.read_parquet(f'{PROC}/{sid}_daily.parquet')
    p_look = part('forecasts', sid, 'pkl')
    if os.path.exists(p_look):
        look, sk = pickle.load(open(p_look, 'rb'))
    else:
        look, sk = {'hourly': {}, 'daily': {}}, []
        for var, col in (('pv', 'pv_kw_per_kwp'), ('wind', 'wind_kw_per_kw'), ('load', 'farm_load_kw')):
            f = fc.forecast_hourly(h[col], LON[sid], TRAIN, VAL + TEST)
            sk.append(fc.skill_table(f, TEST).assign(site=sid, var=var))
            look['hourly'][var] = {(dd, ll): g.sort_values('hr').pred.values for (dd, ll), g in f.groupby(['day', 'lead']) if len(g) == 24}
        for var, col in (('et0', 'et0_mm'), ('rain', 'rain_mm')):
            f = fc.forecast_daily(d[col], TRAIN, VAL + TEST)
            sk.append(fc.skill_table(f, TEST).assign(site=sid, var=var))
            look['daily'][var] = {(r.day, r.lead): r.pred for r in f.itertuples()}
        dump_pkl((look, sk), p_look)
    skill += sk
    p = PICKS[sid][DESIGN]
    design = dict(pv_kwp=p['pv_kwp'], wind_kwr=p['wind_kwr'], batt_kwh=p['batt_kwh'], tank_m3=p['tank_m3'], pump_kw=PICKS[sid]['pump_kw'])
    SITE[sid] = E.Site(sid, LON[sid], h[['pv_kw_per_kwp', 'wind_kw_per_kw', 'farm_load_kw']], d[['et0_mm', 'rain_mm']], design, SOIL[sid], SOW[sid])
    SITE[sid].set_yield_value(PRICE, YMAX_REF, TRAIN)          # provisional; replaced after calibration
    LOOK[sid] = look
    DAILY[sid] = d[['tmin', 'tmax', 'rain_mm', 'et0_mm']]
    log(f"forecasts {sid} {({k: round(v, 1) for k, v in design.items()})} {time.time() - t0:.0f}s")
SK = pd.concat(skill); SK.to_csv(f'{OUT}/forecast_skill.csv', index=False)

# reproducibility check: perfect-information default MPC, Azraq 2020, uncalibrated crop model (no forecasts involved)
_, _, ref = E.run_year(SITE['azraq_jo'], 2020, E.MPCController(perfect=True), None)
print('reproducibility check op_cost', round(ref['op_cost'], 3), '| local reference __REF__')
""")

md("## 2. Stage 0 — calibrate the twin's crop model (p, Ky) against AquaCrop on deficit schedules")
code(r"""
cal_parts = []
for sid in LON:
    p = part('calibration', sid)
    if os.path.exists(p):
        cal_parts.append(pd.read_csv(p, float_precision='round_trip')); log(f'calibration runs {sid}: loaded'); continue
    t0, rows = time.time(), []
    for y in TRAIN + VAL + TEST:
        for name, trig, refill in (SCHED if y in CAL_YEARS else SCHED[:1]):   # full irrigation every year (Ymax)
            irr = E.deficit_schedule(SITE[sid], y, trig, refill)
            t1 = time.time()
            r = aq.season_yield(DAILY[sid], SOIL[sid], f'{y}-{SOW[sid]}', irr)
            rows.append(dict(site=sid, year=y, schedule=name, trigger=trig, refill=refill, sec=time.time() - t1, **r))
    df = pd.DataFrame(rows); save_atomic(p, lambda t: df.to_csv(t, index=False)); cal_parts.append(df)
    log(f'calibration runs {sid}: {len(df)} AquaCrop seasons {time.time() - t0:.0f}s')
CALT = pd.concat(cal_parts, ignore_index=True); CALT.to_csv(f'{OUT}/crop_calibration_runs.csv', index=False)
SEC_AQ = float(CALT.sec.median())
n_seasons = len(SEEDS) * len(LON) * (HYB_STEPS + DRL_STEPS) / 365
print(f'AquaCrop: median {SEC_AQ:.2f} s per season -> about {n_seasons * SEC_AQ / N_ENVS / 3600:.1f} h extra for AquaCrop rewards in training')

YMAX = {sid: CALT[CALT.site == sid].groupby('year').yield_t_ha.max().to_dict() for sid in LON}
P_GRID = np.round(np.arange(0.10, 0.801, 0.025), 3); KY_GRID = np.round(np.arange(0.50, 3.001, 0.05), 2)
FIT, pts = {}, []
for sid in LON:
    s = SITE[sid]
    sub = CALT[(CALT.site == sid) & CALT.year.isin(CAL_YEARS)].reset_index(drop=True)
    rel = (sub.yield_t_ha / sub.year.map(YMAX[sid])).values
    irrs = {(r.year, r.schedule): E.deficit_schedule(s, r.year, r.trigger, r.refill) for r in sub.itertuples()}
    ec = {p: np.array([E.replay_ratio(s, r.year, irrs[(r.year, r.schedule)], p, 1.0) for r in sub.itertuples()]) for p in P_GRID}
    best = (np.inf, None, None)
    for p in P_GRID:
        for ky in KY_GRID:
            sse = float(((np.clip(1 - ky * (1 - ec[p]), 0, None) - rel) ** 2).sum())
            if sse < best[0]:
                best = (sse, float(p), float(ky))
    _, p_b, ky_b = best
    ec_def = np.array([E.replay_ratio(s, r.year, irrs[(r.year, r.schedule)], 0.55, 1.0) for r in sub.itertuples()])
    pred_b, pred_d = np.clip(1 - ky_b * (1 - ec[p_b]), 0, None), np.clip(1 - 1.25 * (1 - ec_def), 0, None)
    rmse = lambda a: float(np.sqrt(np.mean((a - rel) ** 2)))
    edge = p_b in (P_GRID[0], P_GRID[-1]) or ky_b in (KY_GRID[0], KY_GRID[-1])
    FIT[sid] = dict(p=p_b, ky=ky_b, rmse_cal=rmse(pred_b), rmse_default=rmse(pred_d), n=len(rel),
                    r2=float(1 - ((pred_b - rel) ** 2).sum() / ((rel - rel.mean()) ** 2).sum()),
                    ymax_train_mean=float(np.mean([YMAX[sid][y] for y in TRAIN])), on_grid_edge=bool(edge))
    pts.append(pd.DataFrame(dict(site=sid, rel_aquacrop=rel, twin_calibrated=pred_b, twin_default=pred_d, schedule=sub.schedule)))
    print(sid, FIT[sid], '<- ON GRID EDGE' if edge else '', flush=True)
json.dump(FIT, open(f'{OUT}/crop_calibration_fit.json', 'w'), indent=1)
PTS = pd.concat(pts); PTS.to_csv(f'{OUT}/crop_calibration_points.csv', index=False)

# calibrated sites (used by every controller) and the uncalibrated copies (ablation mpc_uncal)
SITE_UNCAL = {}
for sid in LON:
    SITE_UNCAL[sid] = copy.copy(SITE[sid])
    SITE_UNCAL[sid].p, SITE_UNCAL[sid].ky = 0.55, 1.25
    SITE_UNCAL[sid].set_yield_value(PRICE, FIT[sid]['ymax_train_mean'], TRAIN)
    SITE[sid].p, SITE[sid].ky = FIT[sid]['p'], FIT[sid]['ky']
    SITE[sid].set_yield_value(PRICE, FIT[sid]['ymax_train_mean'], TRAIN)

fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for a, sid in zip(ax, LON):
    q = PTS[PTS.site == sid]
    a.scatter(q.rel_aquacrop, q.twin_default, s=12, marker='x', label='default p=0.55, Ky=1.25')
    a.scatter(q.rel_aquacrop, q.twin_calibrated, s=12, label=f"calibrated p={FIT[sid]['p']}, Ky={FIT[sid]['ky']}")
    a.plot([0, 1.05], [0, 1.05], 'k:', lw=0.8); a.set_title(sid, fontsize=9); a.set_xlabel('AquaCrop relative yield'); a.legend(fontsize=7)
ax[0].set_ylabel('twin yield ratio'); plt.tight_layout(); plt.savefig(f'{OUT}/fig_crop_calibration.png', dpi=200); plt.show()
""")

md("## 3. Tune the fixed MPC baseline on training years (reserve × depletion target grid, calibrated crop model)")
code(r"""
TUNED, tabs = {}, []
for sid in LON:
    p_t, p_tab = part('tuning', sid, 'json'), part('tuning', sid)
    if os.path.exists(p_t):
        TUNED[sid] = json.load(open(p_t)); tabs.append(pd.read_csv(p_tab, float_precision='round_trip')); log(f'tuning {sid}: loaded {TUNED[sid]}'); continue
    t0 = time.time()
    TUNED[sid], tab = E.tune_mpc(SITE[sid], TUNE_YEARS, LOOK[sid])
    tab = tab.assign(site=sid); tabs.append(tab)
    save_atomic(p_tab, lambda t: tab.to_csv(t, index=False)); dump_json(TUNED[sid], p_t)     # json last: it marks the site done
    log(f'tuning {sid}: {TUNED[sid]} {time.time() - t0:.0f}s')
dump_json(TUNED, f'{OUT}/mpc_tuned.json'); pd.concat(tabs).to_csv(f'{OUT}/mpc_tuning.csv', index=False)
for sid, t in TUNED.items():
    edge = [k for k in ('reserve', 'dr_target') if t[k] in (E.TUNE_GRID[k][0], E.TUNE_GRID[k][-1])]
    print(sid, t, ('ON GRID EDGE: ' + ', '.join(edge)) if edge else 'interior')
""")

md("## 4. Train PPO agents (hybrid starts at the tuned MPC; model-free DRL from scratch; both get the AquaCrop season reward)")
code(r"""
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

class DriveCheckpoint(BaseCallback):
    # every CKPT_EVERY steps: reward-normalisation statistics, then the model (its zip carries num_timesteps)
    def __init__(self, stem, every):
        super().__init__(); self.stem, self.every = stem, every
    def _on_training_start(self):
        self.last = self.num_timesteps
    def _on_step(self):
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            save_atomic(f'{self.stem}_ckpt_vecnorm.pkl', self.training_env.save)
            save_atomic(f'{self.stem}_ckpt.zip', self.model.save)
            log(f'  checkpoint {os.path.basename(self.stem)} at {self.num_timesteps} steps')
        return True

def aq_fn(sid):
    return lambda y, irr: aq.season_yield(DAILY[sid], SOIL[sid], f'{y}-{SOW[sid]}', irr)['yield_t_ha']

def make(kind, sid, seed, rank):
    def _f():
        a = dict(fn=aq_fn(sid), ymax=YMAX[sid]) if AQ_REWARD else None
        if kind == 'hybrid':
            e = drl.HybridEnv(SITE[sid], TRAIN, LOOK[sid], mpc_kwargs=TUNED[sid], seed=seed * 100 + rank, aq=a)
        else:
            e = drl.DRLOnlyEnv(SITE[sid], TRAIN, LOOK[sid], seed=seed * 100 + rank, aq=a)
        return Monitor(e)
    return _f

def train(kind, sid, seed):
    path = f'{MOD}/{kind}_{sid}_s{seed}.zip'; stem = path[:-4]
    ck, ck_vn = f'{stem}_ckpt.zip', f'{stem}_ckpt_vecnorm.pkl'
    if os.path.exists(path):
        return PPO.load(path, device='cpu'), 'loaded'
    steps = HYB_STEPS if kind == 'hybrid' else DRL_STEPS
    for vec in ('subproc', 'dummy'):
        try:
            fns = [make(kind, sid, seed, r) for r in range(N_ENVS)]
            venv = SubprocVecEnv(fns, start_method='fork') if vec == 'subproc' else DummyVecEnv(fns)
            if os.path.exists(ck) and os.path.exists(ck_vn):
                venv = VecNormalize.load(ck_vn, venv)
                model = PPO.load(ck, env=venv, device='cpu')
                RESUMED[f'{kind}_{sid}_s{seed}'] = int(model.num_timesteps)
                log(f'{kind} {sid} s{seed}: resuming at {model.num_timesteps} of {steps} steps')
            else:
                venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=GAMMA)    # policy sees raw observations
                model = PPO('MlpPolicy', venv, n_steps=365, batch_size=365, n_epochs=10, learning_rate=3e-4, gamma=GAMMA, gae_lambda=0.95,
                            policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.5 if kind == 'hybrid' else 0.0), seed=seed, verbose=0, device='cpu')
            t0 = time.time()
            model.learn(total_timesteps=max(steps - model.num_timesteps, 0), reset_num_timesteps=False, callback=DriveCheckpoint(stem, CKPT_EVERY))
            venv.close()
            save_atomic(path, model.save)
            for f in (ck, ck_vn):
                if os.path.exists(f): os.remove(f)
            return model, f'trained to {model.num_timesteps} steps, {time.time() - t0:.0f}s this session ({vec})'
        except Exception as ex:
            print(kind, sid, seed, vec, 'vec env failed:', repr(ex)[:200], flush=True)
    raise RuntimeError('training failed with both vec envs')

AGENTS = {}
for sid in LON:
    for kind in ('hybrid', 'drl'):
        for seed in SEEDS:
            AGENTS[(kind, sid, seed)], msg = train(kind, sid, seed)
            log(f'{kind} {sid} s{seed}: {msg}')
""")

md("## 5. Evaluate all controllers on validation (2017–2019) and test (2020–2024) years")
code(r"""
def controllers(sid):
    t = TUNED[sid]
    yield E.RuleController(), None, SITE[sid]
    yield E.TwoStageController(reserve=t['reserve']), LOOK[sid], SITE[sid]
    yield E.MPCController(name='mpc_default'), LOOK[sid], SITE[sid]
    yield E.MPCController(**t), LOOK[sid], SITE[sid]
    yield E.MPCController(name='mpc_uncal', **t), LOOK[sid], SITE_UNCAL[sid]
    yield E.MPCController(perfect=True, **t), None, SITE[sid]
    for kind in ('drl', 'hybrid'):
        for seed in SEEDS:
            yield drl.PolicyController(AGENTS[(kind, sid, seed)], kind, f'{kind}_s{seed}', mpc_kwargs=t), LOOK[sid], SITE[sid]

ev_parts = []
for sid in LON:
    p = part('evaluation', sid)
    if os.path.exists(p):
        ev_parts.append(pd.read_csv(p, float_precision='round_trip')); log(f'evaluation {sid}: loaded'); continue
    t0, rows = time.time(), []
    for ctrl, look, site in controllers(sid):
        for y in VAL + TEST:
            H, S, sm = E.run_year(site, y, ctrl, look)
            sm.update(site=sid, controller=ctrl.name, split='val' if y in VAL else 'test')
            rows.append(sm)
            S[['irr_mm']].to_csv(f'{OUT}/schedules/{sid}_{ctrl.name}_{y}.csv')
    df = pd.DataFrame(rows); save_atomic(p, lambda t: df.to_csv(t, index=False)); ev_parts.append(df)
    log(f'evaluation {sid}: {len(df)} controller-years {time.time() - t0:.0f}s')
EV = pd.concat(ev_parts, ignore_index=True); EV.to_csv(f'{OUT}/evaluation_years.csv', index=False)
EV['family'] = EV.controller.str.replace(r'_s\d+$', '', regex=True)
EV[EV.split == 'test'].groupby(['site', 'family']).energy_cost.mean().unstack(0).round(0)
""")

md("## 6. AquaCrop grain yield of every test-year schedule, and the headline total cost")
code(r"""
aq_parts = []
for sid in LON:
    p = part('aquacrop', sid)
    if os.path.exists(p):
        aq_parts.append(pd.read_csv(p, float_precision='round_trip')); log(f'AquaCrop test yields {sid}: loaded'); continue
    t0, yrows = time.time(), []
    for f in sorted(glob.glob(f'{OUT}/schedules/{sid}_*_202[0-4].csv')):
        name = os.path.basename(f)[len(sid) + 1:-9]; y = int(f[-8:-4])
        irr = pd.read_csv(f, index_col=0, parse_dates=True)['irr_mm']
        try:
            r = aq.season_yield(DAILY[sid], SOIL[sid], f'{y}-{SOW[sid]}', irr)
        except Exception as e:
            r = dict(yield_t_ha=np.nan, irrigation_mm=float(irr.sum()), eta_mm=np.nan, error=str(e)[:120])
        yrows.append(dict(site=sid, controller=name, year=y, **r))
    df = pd.DataFrame(yrows); save_atomic(p, lambda t: df.to_csv(t, index=False)); aq_parts.append(df)
    log(f'AquaCrop test yields {sid}: {len(df)} seasons {time.time() - t0:.0f}s')
YL = pd.concat(aq_parts, ignore_index=True)
YL['wp_kg_m3'] = YL.yield_t_ha * 1000 / (YL.irrigation_mm * 10).where(YL.irrigation_mm > 0)
YL.to_csv(f'{OUT}/aquacrop_yields_test.csv', index=False)

TE = EV[EV.split == 'test'].merge(YL[['site', 'controller', 'year', 'yield_t_ha', 'wp_kg_m3']], on=['site', 'controller', 'year'], how='left')
TE['aq_loss_usd'] = [PRICE * SITE[s].area_ha * (YMAX[s][y] - yl) for s, y, yl in zip(TE.site, TE.year, TE.yield_t_ha)]
TE['total_cost'] = TE.energy_cost + TE.aq_loss_usd
TE.to_csv(f'{OUT}/evaluation_test_with_aquacrop.csv', index=False)
COLS = ['total_cost', 'energy_cost', 'aq_loss_usd', 'twin_loss_usd', 'elpsp', 'shed_kwh', 'irrigation_mm', 'yield_t_ha', 'wp_kg_m3',
        'yield_ratio', 'stress_days', 'curtail_kwh', 'batt_throughput_kwh']
SUM = TE.groupby(['site', 'family'])[COLS].agg(['mean', 'std']).round(3)
SUM.to_csv(f'{OUT}/evaluation_test_summary.csv')
TE.groupby(['site', 'family'])[['total_cost', 'energy_cost', 'aq_loss_usd', 'shed_kwh', 'irrigation_mm', 'yield_t_ha']].mean().round(1)
""")

md("## 7. Paired comparisons on the test years (bootstrap over years; agents averaged over seeds)")
code(r"""
rng = np.random.default_rng(42)
T = TE.groupby(['site', 'family', 'year'])[['total_cost', 'energy_cost', 'shed_kwh', 'irrigation_mm', 'yield_t_ha', 'wp_kg_m3']].mean().reset_index()
LOWER = ('total_cost', 'energy_cost', 'shed_kwh', 'irrigation_mm')
comp = []
for sid in LON:
    for a, b in (('hybrid', 'mpc'), ('hybrid', 'drl'), ('hybrid', 'rule'), ('mpc', 'mpc_default'), ('mpc', 'mpc_uncal'), ('mpc', 'rule'),
                 ('drl', 'mpc'), ('mpc_perfect', 'mpc'), ('two_stage', 'rule'), ('mpc', 'two_stage')):
        A = T[(T.site == sid) & (T.family == a)].set_index('year'); B_ = T[(T.site == sid) & (T.family == b)].set_index('year')
        for m in LOWER + ('yield_t_ha', 'wp_kg_m3'):
            d = (A[m] - B_[m]).dropna().values
            if len(d) == 0: continue
            boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            comp.append(dict(site=sid, contrast=f'{a} - {b}', metric=m, mean=d.mean(), lo95=np.percentile(boots, 2.5),
                             hi95=np.percentile(boots, 97.5), share_years_better=np.mean(d < 0) if m in LOWER else np.mean(d > 0)))
CMP = pd.DataFrame(comp); CMP.to_csv(f'{OUT}/paired_comparisons.csv', index=False)
CMP[CMP.metric == 'total_cost'].round(1)
""")
code(r"""
fam = ['rule', 'two_stage', 'mpc_default', 'mpc_uncal', 'mpc', 'drl', 'hybrid', 'mpc_perfect']
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
for a, (m, lab) in zip(ax, (('total_cost', 'Total cost incl. AquaCrop yield loss (USD/yr)'), ('shed_kwh', 'Unserved farm load (kWh/yr)'),
                            ('yield_t_ha', 'AquaCrop yield (t/ha)'))):
    piv = T.groupby(['site', 'family'])[m].mean().unstack().reindex(columns=fam)
    piv.plot.bar(ax=a, width=0.85, legend=(a is ax[0])); a.set_title(lab, fontsize=9); a.set_xlabel('')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_controller_comparison.png', dpi=200); plt.show()
""")

md("## 8. Export for checking (`export_nb03_v3.zip`; trained models and schedules stay on Drive)")
code(r"""
import stable_baselines3, gymnasium, importlib.metadata as im
info = {'run_utc': pd.Timestamp.utcnow().isoformat(), 'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
        'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'sb3': stable_baselines3.__version__, 'gymnasium': gymnasium.__version__,
        'aquacrop': im.version('aquacrop'), 'seeds': SEEDS, 'hyb_steps': HYB_STEPS, 'drl_steps': DRL_STEPS, 'n_envs': N_ENVS, 'gamma': GAMMA,
        'aq_reward': AQ_REWARD, 'sec_per_aquacrop_season': SEC_AQ, 'price_usd_t': PRICE, 'design': DESIGN, 'tuned': TUNED,
        'tune_grid': E.TUNE_GRID, 'tune_years': TUNE_YEARS, 'cal_years': CAL_YEARS, 'crop_fit': FIT, 'soil': SOIL, 'sow': SOW,
        'cost_weights': E.COST, 'repro_check_op_cost': ref['op_cost'], 'train': TRAIN, 'val': VAL, 'test': TEST,
        'resumed_agents_at_step': RESUMED, 'ckpt_every': CKPT_EVERY}
json.dump(info, open(f'{OUT}/run_info.json', 'w'), indent=1, default=str)
log('finished; writing export_nb03_v3.zip')
tmp = '/content/export_nb03_v3'; shutil.rmtree(tmp, ignore_errors=True)
shutil.copytree(OUT, tmp, ignore=shutil.ignore_patterns('models', 'schedules', 'parts', '*.part'))
shutil.make_archive(f'{BASE}/outputs/export_nb03_v3', 'zip', tmp)
files.download(f'{BASE}/outputs/export_nb03_v3.zip')
""")

if __name__ == "__main__":
    import sys
    ref = sys.argv[1] if len(sys.argv) > 1 else "n/a"
    import hashlib
    sha = {f: hashlib.sha256(((SRC / f).read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n") + "\n").encode()).hexdigest()
           for f in ("forecast.py", "env.py", "drl.py", "aquacrop_yield.py")}
    for c in cells:
        c["source"] = c["source"].replace("__REF__", ref).replace("__SHA__", json.dumps(sha))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
                                        "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
    for c in nb["cells"]:
        c["source"] = c["source"].splitlines(keepends=True)
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(OUT, len(cells), "cells", OUT.stat().st_size, "bytes")
