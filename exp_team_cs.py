"""Experiment: does the points model get better if it sees the dedicated team clean-sheet chance?

Same split as train_mlp.py (train to 2025-26 GW28, test 2026-27 GW3 onwards). Runs the current feature set and the
current set + 'team_cs' side by side, with the same seeds, and prints the same scores.

team_cs is never taken from a model that saw that match:
  - training rows: leave-one-season-out (each season's chances come from a team model fitted on the other seasons)
  - validation and test rows: a team model fitted on the training period only

Run:  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python exp_team_cs.py
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import fpl_data as fd, train_events as tev, train_mlp as tm
from fpl_model import predict


def team_model(): return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=12, l2_regularization=1.0, random_state=0)


def add_team_cs(h, train_mask):
    """train_mask(T) picks the team rows of the training period. Returns h with a 'team_cs' column."""
    T = tev.team_table(h); TF = tev.TEAM_F
    played = ~T.future & T.cs.notna(); tr = played & train_mask(T)
    T['team_cs'] = np.nan
    for s in T[tr].season.unique():                                   # out-of-fold for training rows
        fit_ = tr & (T.season != s); m = team_model().fit(T.loc[fit_, TF], T.loc[fit_, 'cs'].astype(int))
        rows = tr & (T.season == s); T.loc[rows, 'team_cs'] = m.predict_proba(T.loc[rows, TF].fillna(-1))[:, 1]
    m = team_model().fit(T.loc[tr, TF], T.loc[tr, 'cs'].astype(int))   # everything after the training period
    rest = ~tr; T.loc[rest, 'team_cs'] = m.predict_proba(T.loc[rest, TF].fillna(-1))[:, 1]
    return h.merge(T[['season', 'fixture', 'team_id', 'team_cs']], on=['season', 'fixture', 'team_id'], how='left')


if __name__ == '__main__':
    h = fd.build()
    h = add_team_cs(h, lambda T: (T.si < 3) | ((T.season == '2025-26') & (T.gw <= 28)))
    train, val, test = tm.split(h); test = test.copy()
    st = test.minutes_r3 >= 45; res = {}
    for tag, F in (('current', fd.FEATURES), ('+team_cs', fd.FEATURES + ['team_cs'])):
        tm.F = F
        nets, sc = tm.fit_nets(train, val, verbose=False); gb = tm.fit_gb(train)
        test[f'gbm_{tag}'] = gb.predict(test[F].fillna(-1)); test[f'mlp_{tag}'] = predict(nets, sc, test)
        test[f'blend_{tag}'] = (test[f'gbm_{tag}'] + test[f'mlp_{tag}']) / 2
    cols = [f'{m}_{t}' for m in ('gbm', 'mlp', 'blend') for t in ('current', '+team_cs')]
    tm.report(test, cols, 'All players, 2026-27 GW3 onwards')
    tm.report(test[st], cols, 'Regular starters (45+ mins over the last 3)')
    for p_, nm in ((1, 'GK'), (2, 'DEF')):
        g = test[st & (test.pos == p_)]
        tm.report(g, ['blend_current', 'blend_+team_cs'], f'{nm} starters only')
        # does a higher clean-sheet chance now mean more predicted points? correlation of prediction with team_cs among starters
        print(f'  {nm}: corr(prediction, team_cs)  current {g[["blend_current", "team_cs"]].corr().iloc[0, 1]:.3f} | '
              f'+team_cs {g[["blend_+team_cs", "team_cs"]].corr().iloc[0, 1]:.3f} | actual points {g[["total_points", "team_cs"]].corr().iloc[0, 1]:.3f}')
    test.to_csv('/tmp/exp_team_cs_test.csv', index=False)
