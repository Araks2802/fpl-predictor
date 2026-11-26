"""Experiment: mix real goals into the team attack/defence numbers (currently expected goals only).
Team goals against = the most goals any of the team's players saw conceded in that match (a 90-minute player sees them all).
Same pre-match rolling window (10 games) and shrinkage as fpl_data. Calibration fitted on seasons before 2025-26;
scored on 2025-26 and 2026-27, which it never saw."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import fpl_data as fd
h = fd.build()
LG, K = 1.35, 5.0
codes = fd._team_codes(); h['_tc'] = [codes.get(k) for k in zip(h.season, h.team_id)]; h['_oc'] = [codes.get(k) for k in zip(h.season, h.opponent_team)]
tf = h.groupby(['season', 'si', 'fixture', '_tc', '_oc'], dropna=False).agg(
    ko=('kickoff_time', 'first'), xg=('expected_goals', lambda x: x.sum(min_count=1)), home=('was_home', 'first'),
    done=('future', lambda x: not x.any())).reset_index().drop_duplicates(['season', 'fixture', '_tc'])
fx = pd.concat([fd._norm_fixtures(fd._load_current()[1] if s_ == '2026-27' else fd._load_old(s_)[1], s_) for s_ in fd.SEASONS])
fx = fx.rename(columns={'id': 'fixture'})[['season', 'fixture', 'team_h_score', 'team_a_score']]
tf = tf.merge(fx, on=['season', 'fixture'], how='left')
tf['ga'] = np.where(tf.home.astype(bool), tf.team_a_score, tf.team_h_score).astype(float)
tf.loc[~tf.done | (tf.xg <= 0), 'xg'] = np.nan; tf.loc[~tf.done, 'ga'] = np.nan
o = tf[['season', 'fixture', '_tc', 'xg', 'ga']].rename(columns={'_tc': '_oc', 'xg': 'xga', 'ga': 'gf'})
tf = tf.merge(o, on=['season', 'fixture', '_oc'], how='left').sort_values('ko')
def pre(g, c):
    s = g[c]; pl = s.notna()
    ps = pd.Series(np.nan, index=g.index); pn = pd.Series(np.nan, index=g.index)
    ps[pl] = s[pl].rolling(10, min_periods=1).sum(); pn[pl] = s[pl].rolling(10, min_periods=1).count()
    ps, pn = ps.shift(1).ffill().fillna(0), pn.shift(1).ffill().fillna(0)
    return (ps + K * LG) / (pn + K)
out = []
for _, g in tf.groupby('_tc'):
    g = g.copy()
    for c in ('xg', 'xga', 'gf', 'ga'): g[c + '_pre'] = pre(g, c)
    out.append(g)
tf = pd.concat(out)
oo = tf[['season', 'fixture', '_tc', 'xg_pre', 'gf_pre']].rename(columns={'_tc': '_oc', 'xg_pre': 'o_xg_pre', 'gf_pre': 'o_gf_pre'})
tf = tf.merge(oo, on=['season', 'fixture', '_oc'], how='left')
d = tf[tf.done & tf.ga.notna() & tf.xga_pre.notna() & tf.o_xg_pre.notna()].copy()
d['cs'] = (d.ga == 0).astype(int)
lg = lambda p: np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))[:, None]
tr, te = d[d.si < 3], d[d.si >= 3]
print(f'train {len(tr)} team-matches, test {len(te)} (2025-26 + 2026-27)')
rows = []
for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    def lam(x): return ((1 - w) * x.o_xg_pre + w * x.o_gf_pre) * ((1 - w) * x.xga_pre + w * x.ga_pre) / LG   # expected goals against this match
    cal = LogisticRegression().fit(lg(np.exp(-lam(tr)).values), tr.cs)
    p = cal.predict_proba(lg(np.exp(-lam(te)).values))[:, 1]
    ll = -np.mean(te.cs * np.log(p) + (1 - te.cs) * np.log(1 - p))
    l_ = lam(te).values; y = te.ga.values; dev = 2 * np.mean(np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / l_), 0) - (y - l_))
    rows.append((w, round(ll, 4), round(np.mean((p - te.cs) ** 2), 4), round(dev, 4)))
print(pd.DataFrame(rows, columns=['weight on real goals', 'clean-sheet log loss', 'clean-sheet Brier', 'goals Poisson deviance']).to_string(index=False))
print('(lower is better on all three)')
