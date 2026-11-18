"""Write expected points for the next 3 gameweeks to data/predictions.parquet (the app reads this file).

Weekly refresh, after a gameweek finishes:
  python fetch_current.py          # new results from the FPL API
  python train_mlp.py              # retrain (a few minutes)
  python make_predictions.py       # new predictions
"""
import json, joblib
from pathlib import Path
import fpl_data as fd, fpl_model as fm, fpl_engine as fe

ROOT = Path(__file__).parent
boot = json.load(open(ROOT / 'data' / '2026-27' / 'bootstrap.json'))
h = fd.build(horizon=38)          # every remaining gameweek (later weeks are rough: form is held at today's level)
nets, sc = fm.load(ROOT / 'models' / 'mlp.pt'); gb = joblib.load(ROOT / 'models' / 'gbm.joblib')
P, gws = fe.predictions(h, nets, sc, gb, boot, horizon=38)
# official fixture difficulty (1 easy to 5 hard) for every player-gameweek; blank gameweek = 0, double = the harder one
fx = json.load(open(ROOT / 'data' / '2026-27' / 'fixtures.json'))
import pandas as pd
fd_rows = [(f['event'], f['team_h'], f['team_h_difficulty']) for f in fx if f['event']] + [(f['event'], f['team_a'], f['team_a_difficulty']) for f in fx if f['event']]
fdr = pd.DataFrame(fd_rows, columns=['gw', 'team_id', 'fdr']).groupby(['gw', 'team_id']).fdr.max()
for g in gws: P[f'fdr_{g}'] = P.team_id.map(lambda t: fdr.get((g, t), 0))
P['code'] = P.index.map({e['id']: e['code'] for e in boot['elements']}); P['team_code'] = P.team_id.map({t['id']: t['code'] for t in boot['teams']})
P['photo'] = P.index.map({e['id']: e['photo'].replace('.jpg', '') for e in boot['elements']})
P['form'] = P.index.map({e['id']: float(e['form']) for e in boot['elements']}); P['total_points'] = P.index.map({e['id']: e['total_points'] for e in boot['elements']})
# ---- chance of a goal, an assist, a clean sheet and 60+ minutes, for the next 6 gameweeks (double gameweek: chance of at least one)
import numpy as np, train_events as tev
ev = joblib.load(ROOT / 'models' / 'events.joblib'); EM, EF = ev['models'], ev['features']
fut = h[h.future & h.gw.isin(gws[:6])].copy()
X = fut[EF].fillna(-1)
fut['p_goal'] = EM['goal'].predict_proba(X)[:, 1]; fut['p_assist'] = EM['assist'].predict_proba(X)[:, 1]; fut['p_60'] = EM['mins60'].predict_proba(X)[:, 1]
fut['p_bonus'] = EM['bonus'].predict_proba(X)[:, 1]; fut['xb'] = fut.p_bonus * EM['bonus_size'].predict(X).clip(1, 3)
fut['p_dc'] = np.where(fut.pos == 1, 0.0, EM['dc'].predict_proba(X)[:, 1])        # goalkeepers cannot earn defensive contribution points
T = tev.team_table(h); fut = fut.merge(T[['season', 'fixture', 'team_id', 'league_cs']], on=['season', 'fixture', 'team_id'], how='left')
fut['p_team_cs'] = EM['team_cs'].predict_proba(fut[ev['team_features']].fillna(-1))[:, 1]
fut['p_cs'] = fut.p_team_cs * fut.p_60
el = pd.DataFrame(boot['elements']).set_index('id')
for k, g in enumerate(gws[:6]):
    f = fut[fut.gw == g]; av = fe.availability(el, k)
    for c in ('p_goal', 'p_assist', 'p_cs', 'p_60', 'p_bonus', 'p_dc'):
        p_any = f.groupby('element')[c].apply(lambda x: 1 - np.prod(1 - x.values))
        P[f'{c}_{g}'] = (p_any.reindex(P.index).fillna(0) * av.reindex(P.index)).values
    P[f'xb_{g}'] = (f.groupby('element').xb.sum().reindex(P.index).fillna(0) * av.reindex(P.index)).values   # expected bonus points (doubles add up)
# ---- team form and matchup facts for the next fixture (shown in the app)
nx = fut[fut.gw == gws[0]].drop_duplicates('element').set_index('element')
for c in ('dc_hit_r10', 'dc_p90_r10', 'bonus_rate_r10', 'own_form5', 'own_gd5', 'own_xgf', 'own_xga', 'opp_form5', 'opp_xgf', 'opp_xga', 'h2h_adj', 'h2h_n', 'venue_adj', 'p_team_cs'):
    P[c] = nx[c].reindex(P.index)
P.to_parquet(ROOT / 'data' / 'predictions.parquet')
nxt = next(e for e in boot['events'] if e['is_next'])
json.dump({'gws': gws, 'next_gw': nxt['id'], 'deadline': nxt['deadline_time']}, open(ROOT / 'data' / 'meta.json', 'w'))
print('gameweeks', gws)
print(P.sort_values(f'xp_{gws[0]}', ascending=False).head(15)[['name', 'team', 'pos', 'price', f'opp_{gws[0]}', f'fdr_{gws[0]}', f'xp_{gws[0]}', f'xp_{gws[1]}', f'xp_{gws[-1]}']].round(2).to_string())
