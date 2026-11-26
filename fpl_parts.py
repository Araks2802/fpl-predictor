"""Expected points built from the parts, using FPL's scoring rules for each position.

  expected points = appearance (1 if he plays, +1 more for 60+ minutes)
                  + goal points by position x expected goals              (GK 10, DEF 6, MID 5, FWD 4)
                  + 3 x expected assists
                  + clean-sheet points by position x clean-sheet chance   (GK 4, DEF 4, MID 1), only if he plays 60+
                  + 2 x chance of defensive-contribution points           (DEF, MID, FWD)
                  + expected bonus
                  + GK: expected save points (1 per 3 saves)
                  - GK/DEF: expected goals-conceded deductions (1 per 2 conceded)
                  + a small per-position adjustment for cards, penalties and own goals

Why: the main points model predicts a player's total directly from his history. That is accurate on average, but for
goalkeepers it leans on a keeper's own past points and saves more than on this week's opponent, so a keeper facing a strong
attack could be ranked above one with a much better clean-sheet chance. Built from the parts, the opponent counts through the
clean-sheet chance, goals conceded and saves. The app's expected points are the average of the two. In testing (exp_components.py,
2026-27 GW3-11, never seen) the average was as accurate as the direct model or better on every overall measure.

fit(h) is run by train_events.py after each gameweek and saved to models/parts.joblib; make_predictions.py applies it.
"""
import numpy as np, pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
import fpl_data as fd

GOAL = {1: 10, 2: 6, 3: 5, 4: 4}; CS = {1: 4, 2: 4, 3: 1, 4: 0}


def _clf(d, y, F): return HC(max_iter=250, learning_rate=0.05, max_leaf_nodes=24, l2_regularization=1.0, random_state=0).fit(d[F].fillna(-1), y.astype(int))
def _pois(d, y, F): return HR(loss='poisson', max_iter=250, learning_rate=0.05, max_leaf_nodes=24, l2_regularization=1.0, random_state=0).fit(d[F].fillna(-1), y)


def floor_mean(lam, k):
    """E[floor(X / k)] for X ~ Poisson(lam): the sum over n of P(X >= k n). Save points (k=3) and goals-conceded points (k=2)."""
    lam = np.asarray(lam, float); return sum(poisson.sf(k * n - 1, lam) for n in range(1, 25))


def _fit_models(tr, F):
    m = {'play': _clf(tr, tr.minutes > 0, F), 'm60': _clf(tr, tr.minutes >= 60, F),
         'goals': _pois(tr, tr.goals_scored, F), 'assists': _pois(tr, tr.assists, F),
         'bonus_any': _clf(tr, tr.bonus >= 1, F)}
    b = tr[tr.bonus > 0]; m['bonus_size'] = HR(max_iter=150, learning_rate=0.05, max_leaf_nodes=12, random_state=0).fit(b[F].fillna(-1), b.bonus)
    trd = tr[(tr.dc_era == 1) & (tr.pos != 1) & tr.defensive_contribution.notna()]
    m['dc'] = _clf(trd, trd.defensive_contribution >= trd.pos.map({2: 10, 3: 12, 4: 12}), F) if len(trd) >= 1000 else None
    g = tr[(tr.pos == 1) & (tr.minutes >= 60)]; m['saves'] = _pois(g, g.saves, F)
    return m


def predict(M, d):
    """Points from the parts for the rows d (needs fd.FEATURES and cs_chance). Returns a DataFrame with each part and xp_parts."""
    F = M['features']; m = M['models']; X = d[F].fillna(-1); pos = d.pos.values; c = pd.DataFrame(index=d.index)
    c['p_play'] = m['play'].predict_proba(X)[:, 1]
    c['p_60'] = np.minimum(m['m60'].predict_proba(X)[:, 1], c.p_play)
    c['x_goals'] = m['goals'].predict(X); c['x_assists'] = m['assists'].predict(X)
    c['p_dc'] = 0.0 if m['dc'] is None else np.where(pos == 1, 0.0, m['dc'].predict_proba(X)[:, 1])
    c['x_bonus'] = m['bonus_any'].predict_proba(X)[:, 1] * m['bonus_size'].predict(X).clip(1, 3)
    cs = d.cs_chance.clip(0.01, 0.99).values
    c['p_cs'] = cs * c.p_60
    c['x_gc_pts'] = np.where(np.isin(pos, (1, 2)), c.p_60 * floor_mean(-np.log(cs), 2), 0.0)   # goals against, from the same chance
    c['x_save_pts'] = np.where(pos == 1, c.p_60 * floor_mean(m['saves'].predict(X), 3), 0.0)
    xp = (c.p_play + c.p_60 + d.pos.map(GOAL).values * c.x_goals + 3 * c.x_assists + d.pos.map(CS).values * c.p_cs
          + 2 * c.p_dc + c.x_bonus + c.x_save_pts - c.x_gc_pts)
    c['xp_parts'] = (xp + d.pos.map(M['adjust']).fillna(0).values).clip(lower=0)
    return c


def fit(h, recent_gws=6):
    """Fit on every played match. The per-position adjustment is the average miss on the latest recent_gws gameweeks,
    measured with models fitted on the weeks before them (so it is not flattered by the models having seen those weeks)."""
    F = fd.FEATURES
    hist = h[~h.future & h.total_points.notna() & h.pos.isin([1, 2, 3, 4])]
    order = hist.si * 100 + hist.gw; cut = np.sort(order.unique())[-recent_gws]
    early, late = hist[order < cut], hist[order >= cut]
    M0 = {'models': _fit_models(early, F), 'features': F, 'adjust': {}}
    miss = (late.total_points - predict(M0, late).xp_parts).groupby(late.pos).mean()
    return {'models': _fit_models(hist, F), 'features': F, 'adjust': {int(k): float(v) for k, v in miss.items()}}
