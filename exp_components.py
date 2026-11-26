"""Experiment: build expected points from the chances, using FPL's scoring for each position, and test it against the current model.

expected points = appearance (1 if he plays, +1 more for 60+ minutes)
                + goal points by position x expected goals            (GK 10, DEF 6, MID 5, FWD 4)
                + 3 x expected assists
                + clean-sheet points by position x chance of a clean sheet while he plays 60+   (GK 4, DEF 4, MID 1)
                + 2 x chance of defensive-contribution points          (DEF, MID, FWD)
                + expected bonus points
                + GK: expected save points (1 per 3 saves)
                - GK/DEF: expected goals-conceded deductions (1 per 2 conceded)
                + a small per-position adjustment for cards, penalties and own goals (learned on the validation weeks)

Same split as train_mlp.py: models fitted up to 2025-26 GW28, adjustment learned on 2025-26 GW29-38, tested on 2026-27 GW3 onwards.
Run:  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python exp_components.py
"""
import numpy as np, pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
import fpl_data as fd, train_mlp as tm, exp_team_cs as xt
from fpl_model import predict

F = fd.FEATURES
GOAL = {1: 10, 2: 6, 3: 5, 4: 4}; CS = {1: 4, 2: 4, 3: 1, 4: 0}; LG = 1.35


def clf(d, y): return HC(max_iter=250, learning_rate=0.05, max_leaf_nodes=24, l2_regularization=1.0, random_state=0).fit(d[F].fillna(-1), y.astype(int))
def pois(d, y): return HR(loss='poisson', max_iter=250, learning_rate=0.05, max_leaf_nodes=24, l2_regularization=1.0, random_state=0).fit(d[F].fillna(-1), y)
def floor_mean(lam, k):
    """E[floor(X / k)] for X ~ Poisson(lam): the sum over n of P(X >= k n)."""
    lam = np.asarray(lam, float); return sum(poisson.sf(k * n - 1, lam) for n in range(1, 25))


def components(tr, d, team_cs):
    """Expected points from the chances for the rows d, with models fitted on tr. team_cs: column with the team's clean-sheet chance."""
    X = d[F].fillna(-1); c = pd.DataFrame(index=d.index); pos = d.pos.values
    c['p_play'] = clf(tr, tr.minutes > 0).predict_proba(X)[:, 1]
    c['p_60'] = np.minimum(clf(tr, tr.minutes >= 60).predict_proba(X)[:, 1], c.p_play)
    c['x_goals'] = pois(tr, tr.goals_scored).predict(X)
    c['x_assists'] = pois(tr, tr.assists).predict(X)
    dc_rows = (tr.dc_era == 1) & (tr.pos != 1) & tr.defensive_contribution.notna()
    trd = tr[dc_rows]
    c['p_dc'] = 0.0 if len(trd) < 1000 else np.where(pos == 1, 0.0, clf(trd, trd.defensive_contribution >= trd.pos.map({2: 10, 3: 12, 4: 12})).predict_proba(X)[:, 1])
    pany = clf(tr, tr.bonus >= 1).predict_proba(X)[:, 1]; b = tr[tr.bonus > 0]
    c['x_bonus'] = pany * HR(max_iter=150, learning_rate=0.05, max_leaf_nodes=12, random_state=0).fit(b[F].fillna(-1), b.bonus).predict(X).clip(1, 3)
    c['p_cs'] = d[team_cs].values * c.p_60
    lam_ga = -np.log(d[team_cs].clip(0.01, 0.99).values)                         # goals against, from the same clean-sheet chance
    c['x_gc_pts'] = np.where(np.isin(pos, (1, 2)), c.p_60 * floor_mean(lam_ga, 2), 0.0)
    g = tr[(tr.pos == 1) & (tr.minutes >= 60)]; lam_sv = pois(g, g.saves).predict(X)
    c['x_save_pts'] = np.where(pos == 1, c.p_60 * floor_mean(lam_sv, 3), 0.0)
    c['xp'] = (c.p_play + c.p_60 + d.pos.map(GOAL).values * c.x_goals + 3 * c.x_assists + d.pos.map(CS).values * c.p_cs
               + 2 * c.p_dc + c.x_bonus + c.x_save_pts - c.x_gc_pts)
    return c


if __name__ == '__main__':
    h = fd.build()
    h = xt.add_team_cs(h, lambda T: (T.si < 3) | ((T.season == '2025-26') & (T.gw <= 28)))
    h['cs_rough'] = np.exp(-h.opp_xgf * h.own_xga / LG)                           # same as cs_chance
    train, val, test = tm.split(h); test = test.copy(); val = val.copy()
    # current model
    tm.F = F; nets, sc = tm.fit_nets(train, val, verbose=False); gb = tm.fit_gb(train)
    test['blend'] = (gb.predict(test[F].fillna(-1)) + predict(nets, sc, test)) / 2
    test['last5'] = test.total_points_r5.fillna(0)
    both = pd.concat([val, test]); c0 = components(train, both, 'cs_rough')
    for tag, col in (('rough cs', 'cs_rough'), ('team-model cs', 'team_cs')):
        c = c0.copy(); pos = both.pos.values
        if col != 'cs_rough':     # swap only the clean-sheet and goals-conceded parts
            old = both.pos.map(CS).values * c.p_cs - c.x_gc_pts
            c['p_cs'] = both[col].values * c.p_60
            c['x_gc_pts'] = np.where(np.isin(pos, (1, 2)), c.p_60 * floor_mean(-np.log(both[col].clip(0.01, 0.99).values), 2), 0.0)
            c['xp'] = c.xp - old + both.pos.map(CS).values * c.p_cs - c.x_gc_pts
        # per-position adjustment (cards, penalties, own goals, anything the parts miss), learned on the validation weeks only
        adj = (both.loc[val.index, 'total_points'] - c.loc[val.index, 'xp']).groupby(both.loc[val.index, 'pos']).mean()
        test[f'parts ({tag})'] = (c.loc[test.index, 'xp'] + test.pos.map(adj)).clip(lower=0)
        test[f'avg(blend, parts {tag})'] = (test.blend + test[f'parts ({tag})']) / 2
        print(f'{tag}: adjustment per position', adj.round(2).to_dict())
        if tag == 'rough cs': cr = c.loc[test.index]
    cols = ['last5', 'blend', 'parts (rough cs)', 'parts (team-model cs)', 'avg(blend, parts rough cs)', 'avg(blend, parts team-model cs)']
    st = test.minutes_r3 >= 45
    tm.report(test, cols, 'All players, 2026-27 GW3 onwards')
    tm.report(test[st], cols, 'Regular starters (45+ mins over the last 3)')
    for p_, nm in {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.items():
        tm.report(test[st & (test.pos == p_)], ['blend', 'parts (rough cs)', 'avg(blend, parts rough cs)'], f'{nm} starters')
    s_ = test[test.minutes_r3 >= 60]
    print('\nCALIBRATION starters (60+): actual', round(s_.total_points.mean(), 2), {c_: round(s_[c_].mean(), 2) for c_ in ['blend', 'parts (rough cs)']})
    pd.concat([test, cr.add_prefix('c_')], axis=1).to_csv('/tmp/exp_components_test.csv', index=False)
