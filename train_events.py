"""Event models: the chance a player scores, assists, keeps a clean sheet and plays 60+ minutes in a fixture.

Each is a gradient-boosting classifier on the same pre-match features as the points model.
Tested on 2026-27 GW3 onwards (never seen), then retrained on everything and saved to models/events.joblib.

Run:  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python train_events.py
"""
import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
import fpl_data as fd

OUT = Path(__file__).parent / 'models'
F = fd.FEATURES
EVENTS = {'goal': lambda d: d.goals_scored >= 1, 'assist': lambda d: d.assists >= 1,
          'cs': lambda d: (d.clean_sheets >= 1) & (d.minutes >= 60), 'mins60': lambda d: d.minutes >= 60,
          'bonus': lambda d: d.bonus >= 1,
          # 2 defensive-contribution points: 10+ CBIT for defenders, 12+ CBIRT for midfielders and forwards
          'dc': lambda d: d.defensive_contribution >= d.pos.map({2: 10, 3: 12, 4: 12})}
# defensive contributions were only recorded from 2025-26 and goalkeepers cannot earn them, so that model learns from those rows only
ROWS = {'dc': lambda d: (d.dc_era == 1) & (d.pos != 1) & d.defensive_contribution.notna()}


TEAM_F = ['own_xga', 'opp_xgf', 'own_def', 'opp_att', 'home', 'cs_chance', 'own_form5', 'opp_form5', 'own_gd5', 'opp_gd5', 'league_cs']


def team_table(h):
    """One row per team per fixture. Team clean sheet = a player who played 89+ minutes got clean-sheet points."""
    t = h.groupby(['season', 'si', 'gw', 'fixture', 'team_id'], as_index=False).agg(
        **{c: (c, 'first') for c in TEAM_F if c != 'league_cs'}, future=('future', 'first'),
        cs=('clean_sheets', lambda x: np.nan), ko=('kickoff_time', 'first'))
    full = h[h.minutes >= 89].groupby(['season', 'fixture', 'team_id']).clean_sheets.max().rename('cs_').reset_index()
    t = t.drop(columns='cs').merge(full, on=['season', 'fixture', 'team_id'], how='left').rename(columns={'cs_': 'cs'})
    t.loc[t.future, 'cs'] = np.nan
    # the league's clean-sheet rate over the previous 6 gameweeks, so the model follows changes in scoring between seasons
    lv = t[~t.future].groupby(['si', 'gw']).cs.mean().rename('m').reset_index().sort_values(['si', 'gw'])
    lv['league_cs'] = lv.m.shift(1).rolling(6, min_periods=1).mean()
    t = t.merge(lv[['si', 'gw', 'league_cs']], on=['si', 'gw'], how='left'); t['league_cs'] = t.league_cs.fillna(lv.m.tail(6).mean())
    return t


def fit(df, y):
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=24, l2_regularization=1.0, random_state=0)
    return m.fit(df[F].fillna(-1), y.astype(int))


def cs_like_app(h, train, d):
    """The clean-sheet chance the app shows (see make_predictions.py): the team's chance from expected goals (cs_chance),
    rescaled to the real clean-sheet rate on the training seasons only, times the player's chance of playing 60+ minutes."""
    from sklearn.linear_model import LogisticRegression
    lg = lambda p: np.log(p.clip(0.01, 0.99) / (1 - p.clip(0.01, 0.99))).values[:, None]
    T = team_table(h); Tp = T[~T.future & T.cs.notna()]
    Ttr = Tp[(Tp.si < 3) | ((Tp.season == '2025-26') & (Tp.gw <= 28))]
    cal = LogisticRegression().fit(lg(Ttr.cs_chance), Ttr.cs.astype(int))
    p60 = fit(train, train.minutes >= 60).predict_proba(d[F].fillna(-1))[:, 1]
    return cal.predict_proba(lg(d.cs_chance))[:, 1] * p60


if __name__ == '__main__':
    h = fd.build(horizon=1)
    hist = h[~h.future & h.total_points.notna() & h.pos.isin([1, 2, 3, 4])]
    train = hist[(hist.si < 3) | ((hist.season == '2025-26') & (hist.gw <= 28))]
    test = hist[(hist.season == '2026-27') & (hist.gw >= 3)]
    rows = []
    rv = hist[hist.season == '2026-27']                                   # back-test chances for every played week of the season (Review tab)
    test_probs = rv[['season', 'gw', 'element', 'fixture']].copy()
    for ev, fn in EVENTS.items():
        keep = ROWS.get(ev, lambda d: d.minutes.notna())
        tr_, te_ = train[keep(train)], test[keep(test)]
        m = fit(tr_, fn(tr_)); p = m.predict_proba(te_[F].fillna(-1))[:, 1]; y = fn(te_).astype(int).values
        if ev == 'cs': p = cs_like_app(h, tr_, te_)                     # score the clean-sheet chance the app shows, not this player-level model
        test_probs[f'p_{ev}'] = m.predict_proba(rv[F].fillna(-1))[:, 1] if ev != 'dc' else np.where(rv.pos == 1, 0.0, m.predict_proba(rv[F].fillna(-1))[:, 1])
        base = np.full(len(y), fn(tr_).mean())                           # baseline: everyone gets the average rate
        brier, brier0 = np.mean((p - y) ** 2), np.mean((base - y) ** 2)
        st = te_.minutes_r3.values >= 60
        top = np.argsort(-p)[:200]
        rows.append((ev, y.mean(), p.mean(), 100 * (1 - brier / brier0), p[st].mean(), y[st].mean(), p[top].mean(), y[top].mean()))
    t = pd.DataFrame(rows, columns=['event', 'actual rate', 'predicted rate', 'skill vs average %', 'starters predicted', 'starters actual', 'top 200 predicted', 'top 200 actual'])
    print('Test on 2026-27 GW3 onwards'); print(t.round(3).to_string(index=False)); t.to_csv(OUT / 'event_scores.csv', index=False)
    # ---- team clean-sheet model, then player chance = team chance x chance he plays 60+
    T = team_table(h); Tp = T[~T.future & T.cs.notna()]
    ttr = Tp[(Tp.si < 3) | ((Tp.season == '2025-26') & (Tp.gw <= 28))]
    tcm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=12, l2_regularization=1.0, random_state=0).fit(ttr[TEAM_F], ttr.cs.astype(int))
    m60 = fit(train, EVENTS['mins60'](train))
    te = test.merge(T[['season', 'fixture', 'team_id'] + TEAM_F[-1:]], on=['season', 'fixture', 'team_id'], how='left')
    p_team = tcm.predict_proba(te[TEAM_F].fillna(-1))[:, 1]; p60 = m60.predict_proba(te[F].fillna(-1))[:, 1]
    p = p_team * p60; y = EVENTS['cs'](te).astype(int).values; st = te.minutes_r3.values >= 60
    keep = te.pos.isin([1, 2]).values
    print(f"\nteam-based clean sheet (GK/DEF): predicted {p[keep].mean():.3f} vs actual {y[keep].mean():.3f} | starters {p[keep & st].mean():.3f} vs {y[keep & st].mean():.3f}"
          f" | skill vs average {100 * (1 - np.mean((p - y) ** 2) / np.mean((y.mean() - y) ** 2)):.1f}%")
    # expected bonus = chance of any bonus x how many (1, 2 or 3) he gets when he does. Splitting it keeps the two numbers consistent.
    from sklearn.ensemble import HistGradientBoostingRegressor
    cond = lambda d: HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=12, random_state=0).fit(d[d.bonus > 0][F].fillna(-1), d[d.bonus > 0].bonus)
    pany = fit(train, train.bonus >= 1).predict_proba(test[F].fillna(-1))[:, 1]
    pb = pany * cond(train).predict(test[F].fillna(-1)).clip(1, 3); st = (test.minutes_r3 >= 60).values
    print(f"expected bonus: predicted {pb.mean():.3f} vs actual {test.bonus.mean():.3f} | starters {pb[st].mean():.3f} vs {test.bonus.values[st].mean():.3f} | top 100: {np.sort(pb)[-100:].mean():.2f} vs {test.bonus.values[np.argsort(pb)[-100:]].mean():.2f}")
    test_probs.to_csv(OUT / 'event_test_predictions.csv', index=False)
    final = {ev: fit(hist[ROWS.get(ev, lambda d: d.minutes.notna())(hist)], fn(hist[ROWS.get(ev, lambda d: d.minutes.notna())(hist)])) for ev, fn in EVENTS.items()}
    final['bonus_size'] = cond(hist)
    final['team_cs'] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=12, l2_regularization=1.0, random_state=0).fit(Tp[TEAM_F], Tp.cs.astype(int))
    joblib.dump({'models': final, 'features': F, 'team_features': TEAM_F}, OUT / 'events.joblib'); print('saved models/events.joblib')
    # points built from the parts (fpl_parts.py): averaged with the main points model in make_predictions.py
    import fpl_parts
    Mp = fpl_parts.fit(h); joblib.dump(Mp, OUT / 'parts.joblib'); print('saved models/parts.joblib | adjustment per position', {k: round(v, 2) for k, v in Mp['adjust'].items()})
