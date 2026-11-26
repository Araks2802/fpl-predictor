"""Build the player-by-fixture table and the features used by the models.

Every feature for a fixture is computed only from games played BEFORE it (we shift by one),
so the model never sees the answer. Rows for future fixtures are added with no stats, so the same
code produces the features used to predict next week.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path

D = Path(__file__).parent / 'data'
SEASONS = ['2022-23', '2023-24', '2024-25', '2025-26', '2026-27']
POS = {'GK': 1, 'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
STATS = ['total_points', 'minutes', 'goals_scored', 'assists', 'bps', 'bonus', 'clean_sheets', 'saves', 'expected_goals',
         'expected_assists', 'expected_goals_conceded', 'defensive_contribution', 'threat', 'creativity', 'influence']
# the 2022-23 to 2024-25 files have no 'defensive_contribution' (the rule began in 2025-26), so it is 0 there
ERA_DC = {'2022-23': 0, '2023-24': 0, '2024-25': 0, '2025-26': 1, '2026-27': 1}


def season_index(s): return SEASONS.index(s)


def _load_old(s):
    g = pd.read_csv(D / s / 'merged_gw.csv', on_bad_lines='skip', low_memory=False)
    pr = pd.read_csv(D / s / 'players_raw.csv', low_memory=False)[['id', 'code', 'team', 'element_type', 'web_name']]
    pr.columns = ['element', 'code', 'team_id', 'pos', 'web_name']
    g = g.rename(columns={'round': 'gw'}) if 'gw' not in g else g
    g['gw'] = g['GW'] if 'GW' in g else g['gw']
    g = g.merge(pr, on='element', how='left')
    g['was_home'] = g['was_home'].astype(bool)
    for c in ('expected_goals', 'expected_assists', 'expected_goals_conceded', 'defensive_contribution'):
        if c not in g: g[c] = 0.0
    g['season'] = s
    return g, pd.read_csv(D / s / 'fixtures.csv')


def _load_current(s='2026-27'):
    b = json.load(open(D / s / 'bootstrap.json'))
    h = pd.read_csv(D / s / 'history.csv', low_memory=False)
    el = pd.DataFrame(b['elements'])[['id', 'code', 'team', 'element_type', 'web_name']]
    el.columns = ['element', 'code', 'team_id', 'pos', 'web_name']
    h = h.merge(el, on='element', how='left'); h['gw'] = h['round']; h['season'] = s
    for c in ('expected_goals', 'expected_assists', 'expected_goals_conceded', 'threat', 'creativity', 'influence'):
        h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0)
    fx = pd.DataFrame(json.load(open(D / s / 'fixtures.json')))
    return h, fx


def _norm_fixtures(fx, s):
    f = fx.rename(columns={'event': 'gw'})[['id', 'gw', 'team_h', 'team_a', 'team_h_score', 'team_a_score', 'kickoff_time']].copy()
    f['season'] = s; f['kickoff_time'] = pd.to_datetime(f['kickoff_time'], utc=True, errors='coerce'); return f


def team_strength(fixtures):
    """Rolling attack/defence strength for every team before each fixture (goals scored / conceded per game, shrunk to the league mean)."""
    rows = []
    for r in fixtures.itertuples():
        rows.append((r.season, r.gw, r.kickoff_time, r.id, r.team_h, r.team_a, r.team_h_score, r.team_a_score, True))
        rows.append((r.season, r.gw, r.kickoff_time, r.id, r.team_a, r.team_h, r.team_a_score, r.team_h_score, False))
    t = pd.DataFrame(rows, columns=['season', 'gw', 'ko', 'fixture', 'team', 'opp', 'gf', 'ga', 'home']).sort_values(['ko', 'fixture'])
    t['si'] = t.season.map(season_index)
    # team ids are renumbered each season (promotion / relegation), so follow each club by its permanent code
    codes = _team_codes(); t['tc'] = [codes.get(k) for k in zip(t.season, t.team)]
    t['res'] = np.where(t.gf > t.ga, 3.0, np.where(t.gf == t.ga, 1.0, 0.0)); t.loc[t.gf.isna(), 'res'] = np.nan
    out = []
    PRIOR, LG = 6.0, 1.45

    def before(s, w, how):
        """Value over the last w PLAYED games before each fixture. Future fixtures carry the latest value forward."""
        pl = s.notna(); v = pd.Series(np.nan, index=s.index)
        v[pl] = getattr(s[pl].rolling(w, min_periods=1), how)()
        return v.shift(1).ffill()

    for tc, g in t.groupby('tc'):
        g = g.sort_values('ko').copy(); gf = g.gf.astype(float); ga = g.ga.astype(float)
        pn = before(gf, 8, 'count').fillna(0)
        g['att'] = (before(gf, 8, 'sum').fillna(0) + PRIOR * LG) / (pn + PRIOR); g['def_'] = (before(ga, 8, 'sum').fillna(0) + PRIOR * LG) / (pn + PRIOR)
        # recent form: league points per game and goal difference per game over the last 5, shrunk a little towards average (1.37 ppg, 0 GD)
        n5 = before(gf, 5, 'count').fillna(0)
        g['form5'] = (before(g.res, 5, 'sum').fillna(0) + 2 * 1.37) / (n5 + 2)
        g['gd5'] = (before(gf - ga, 5, 'sum').fillna(0)) / (n5 + 2)
        out.append(g)
    return pd.concat(out)[['season', 'fixture', 'team', 'att', 'def_', 'form5', 'gd5']]


def build(horizon=3):
    """Return the full table (history + fixtures for the next `horizon` gameweeks) with features."""
    parts, fxs = [], []
    for s in SEASONS:
        if s == '2026-27': g, fx = _load_current()
        else: g, fx = _load_old(s)
        keep = ['season', 'gw', 'element', 'code', 'team_id', 'pos', 'web_name', 'fixture', 'opponent_team', 'was_home', 'value'] + STATS
        for c in keep:
            if c not in g: g[c] = 0
        g['kickoff_time'] = pd.to_datetime(g['kickoff_time'], utc=True, errors='coerce')
        parts.append(g[keep + ['kickoff_time']]); fxs.append(_norm_fixtures(fx, s))
    fixtures = pd.concat(fxs, ignore_index=True)
    ts = team_strength(fixtures)
    h = pd.concat(parts, ignore_index=True)
    h['future'] = False
    # a player's team must come from the fixture itself: the season files list his END-of-season club, which is wrong for players who moved mid-season
    fk = fixtures[['season', 'id', 'team_h', 'team_a']].rename(columns={'id': 'fixture'})
    h = h.merge(fk, on=['season', 'fixture'], how='left')
    hb = h.was_home.astype(bool); ok = h.team_h.notna()
    h.loc[ok, 'team_id'] = np.where(hb[ok], h.team_h[ok], h.team_a[ok]); h.loc[ok, 'opponent_team'] = np.where(hb[ok], h.team_a[ok], h.team_h[ok])
    h = h.drop(columns=['team_h', 'team_a'])
    h = h.drop_duplicates(['season', 'code', 'fixture'])   # a handful of repeated rows in the source files
    # ---- future rows: the next unplayed gameweek of the current season
    b = json.load(open(D / '2026-27' / 'bootstrap.json')); nxt = next(e['id'] for e in b['events'] if e['is_next'])
    el = pd.DataFrame(b['elements'])
    cf = fixtures[(fixtures.season == '2026-27') & (fixtures.gw >= nxt) & (fixtures.gw < nxt + horizon)]
    fut = []
    for r in cf.itertuples():
        for home, tid, opp in ((True, r.team_h, r.team_a), (False, r.team_a, r.team_h)):
            for p in el[el.team == tid].itertuples():
                fut.append(dict(season='2026-27', gw=r.gw, element=p.id, code=p.code, team_id=tid, pos=p.element_type, web_name=p.web_name,
                                fixture=r.id, opponent_team=opp, was_home=home, value=p.now_cost, kickoff_time=r.kickoff_time, future=True))
    fut = pd.DataFrame(fut)
    for c in STATS: fut[c] = np.nan
    h = pd.concat([h, fut], ignore_index=True)
    h['was_home'] = h.was_home.astype(bool)
    # goals the team conceded in that match (from the fixture's score; blank for future matches). Used to mix real goals into team xG.
    sc_ = fixtures[['season', 'id', 'team_h_score', 'team_a_score']].rename(columns={'id': 'fixture'})
    h = h.merge(sc_, on=['season', 'fixture'], how='left')
    h['_ga'] = np.where(h.was_home, h.team_a_score, h.team_h_score).astype(float); h.loc[h.future, '_ga'] = np.nan
    h = h.drop(columns=['team_h_score', 'team_a_score'])
    # ---- team strength of own team and opponent
    ts = ts.drop_duplicates(['season', 'fixture', 'team'])
    own = ts.rename(columns={'team': 'team_id', 'att': 'own_att', 'def_': 'own_def', 'form5': 'own_form5', 'gd5': 'own_gd5'})
    opp = ts.rename(columns={'team': 'opponent_team', 'att': 'opp_att', 'def_': 'opp_def', 'form5': 'opp_form5', 'gd5': 'opp_gd5'})
    # fixture ids restart every season, so the season must be part of the key
    h = h.merge(own, on=['season', 'fixture', 'team_id'], how='left').merge(opp, on=['season', 'fixture', 'opponent_team'], how='left')
    for c in ('own_att', 'own_def', 'opp_att', 'opp_def'): h[c] = h[c].fillna(1.45)
    for c in ('own_form5', 'opp_form5'): h[c] = h[c].fillna(1.37)
    for c in ('own_gd5', 'opp_gd5'): h[c] = h[c].fillna(0.0)
    # ---- order in time and compute form features per player (shifted so only past games count)
    h['si'] = h.season.map(season_index); h['dc_era'] = h.season.map(ERA_DC)
    h = h.sort_values(['code', 'si', 'gw', 'kickoff_time']).reset_index(drop=True)
    gcode = h.groupby('code', sort=False)
    for c in STATS:
        h[c] = pd.to_numeric(h[c], errors='coerce')
    # defensive contributions only exist from 2025-26. Earlier seasons show 0, which means 'not recorded', so make them missing.
    h.loc[h.dc_era == 0, 'defensive_contribution'] = np.nan
    for w in (3, 5, 10):
        for c in ('total_points', 'minutes', 'bps', 'expected_goals', 'expected_assists', 'defensive_contribution', 'saves', 'bonus'):
            h[f'{c}_r{w}'] = gcode[c].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    h['start_rate5'] = gcode['minutes'].transform(lambda x: (x.shift(1) >= 60).astype(float).where(x.shift(1).notna()).rolling(5, min_periods=1).mean())
    h['games_seen'] = gcode.cumcount()
    h['season_pts_mean'] = h.groupby(['code', 'season'], sort=False)['total_points'].transform(lambda x: x.shift(1).expanding().mean())
    h['gap_days'] = gcode['kickoff_time'].transform(lambda x: x.diff().dt.total_seconds() / 86400)
    h['n_fix_gw'] = h.groupby(['code', 'season', 'gw'])['fixture'].transform('count')
    h = add_form_and_team_xg(h, gcode)
    # future weeks have no results yet, so rolling windows would slowly fill with blanks. Hold each player's form at today's level instead.
    fut = h.future.values
    if fut.any():
        form_cols = [c for c in h.columns if any(c.startswith(s + '_r') for s in STATS)] + ['start_rate5', 'games_seen', 'season_pts_mean'] + HOLD_EXTRA
        form_cols = list(dict.fromkeys(form_cols))   # some names match two patterns (e.g. bonus_rate_r10); keep each once
        first = h[fut].groupby('code')[form_cols].transform('first')
        h.loc[fut, form_cols] = first.values
    h['price'] = h['value'] / 10.0
    h['home'] = h.was_home.astype(float)
    return h


HOLD_EXTRA = ['pts_p90_r10', 'xgi_p90_r10', 'pts_p90_r30', 'xgi_p90_r30', 'season_xgi_mean',
              'dc_hit_r5', 'dc_hit_r10', 'dc_p90_r10', 'bonus_rate_r10', 'bps_p90_r10']

BASE_FEATURES = ['pos', 'price', 'home', 'own_att', 'own_def', 'opp_att', 'opp_def', 'dc_era', 'n_fix_gw', 'games_seen', 'start_rate5', 'season_pts_mean'] + \
    [f'{c}_r{w}' for w in (3, 5, 10) for c in ('total_points', 'minutes', 'bps', 'expected_goals', 'expected_assists', 'defensive_contribution', 'saves', 'bonus')]
EXTRA_FEATURES = [f'{c}_r20' for c in ('total_points', 'minutes', 'bps', 'expected_goals', 'expected_assists')] + HOLD_EXTRA[:5] + \
    ['prev_ppg', 'prev_xgi90', 'own_xgf', 'own_xga', 'opp_xgf', 'opp_xga', 'att_vs_def', 'cs_chance', 'pos_level']
TEAM_FORM = ['own_form5', 'opp_form5', 'own_gd5', 'opp_gd5']
MATCHUP = ['h2h_adj', 'h2h_n', 'venue_adj']
DC_BONUS = ['dc_hit_r5', 'dc_hit_r10', 'dc_p90_r10', 'bonus_rate_r10', 'bps_p90_r10']
FEATURES = BASE_FEATURES + EXTRA_FEATURES + TEAM_FORM + MATCHUP + DC_BONUS


def _team_codes():
    m = {}
    for s in SEASONS:
        if s == '2026-27':
            for t in json.load(open(D / s / 'bootstrap.json'))['teams']: m[(s, t['id'])] = t['code']
        else:
            for t in pd.read_csv(D / s / 'teams.csv').itertuples(): m[(s, t.id)] = t.code
    return m


def add_form_and_team_xg(h, gcode):
    """Longer form windows, per-90 rates, last season's level, and team strength from expected goals."""
    for c in ('total_points', 'minutes', 'bps', 'expected_goals', 'expected_assists'):
        h[f'{c}_r20'] = gcode[c].transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    h['_xgi'] = h.expected_goals + h.expected_assists
    for w in (10, 30):   # per-90 rates need enough minutes to mean anything, so require 270
        mins = gcode['minutes'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).sum())
        for c, nm in (('total_points', 'pts'), ('_xgi', 'xgi')):
            tot = gcode[c].transform(lambda x: x.shift(1).rolling(w, min_periods=1).sum())
            h[f'{nm}_p90_r{w}'] = (90 * tot / mins).where(mins >= 270)
    h['season_xgi_mean'] = h.groupby(['code', 'season'], sort=False)['_xgi'].transform(lambda x: x.shift(1).expanding().mean())
    # last season: points per appearance and xGI per 90
    played = h[h.minutes > 0]
    prev = played.groupby(['code', 'si']).agg(ppg=('total_points', 'mean'), xgi=('_xgi', 'sum'), mins=('minutes', 'sum')).reset_index()
    prev['xgi90'] = 90 * prev.xgi / prev.mins.clip(lower=1); prev['si'] = prev.si + 1
    h = h.merge(prev[['code', 'si', 'ppg', 'xgi90']].rename(columns={'ppg': 'prev_ppg', 'xgi90': 'prev_xgi90'}), on=['code', 'si'], how='left')
    # team xG for and against per fixture, rolling over the last 10 played games, shrunk towards the league mean
    codes = _team_codes(); h['_tc'] = [codes.get(k) for k in zip(h.season, h.team_id)]; h['_oc'] = [codes.get(k) for k in zip(h.season, h.opponent_team)]
    tf = h.groupby(['season', 'fixture', '_tc', '_oc'], dropna=False).agg(ko=('kickoff_time', 'first'), xg=('expected_goals', lambda x: x.sum(min_count=1)),
                                                                       ga=('_ga', 'first'), done=('future', lambda x: not x.any())).reset_index()
    tf.loc[~tf.done | (tf.xg <= 0), 'xg'] = np.nan     # 0 team xG means the data is missing (early 2022-23), not a real 0
    tf.loc[~tf.done, 'ga'] = np.nan
    tf = tf.drop_duplicates(['season', 'fixture', '_tc'])
    opp = tf[['season', 'fixture', '_tc', 'xg', 'ga']].rename(columns={'_tc': '_oc', 'xg': 'xga', 'ga': 'gf'})
    tf = tf.merge(opp, on=['season', 'fixture', '_oc'], how='left').sort_values('ko')
    LG, K = 1.35, 5.0
    # team attack / defence = 75% expected goals + 25% real goals. xG alone ignores finishing that is partly real; tested on 980
    # team-matches in 2025-26 and 2026-27 (never seen when the weight was chosen), a quarter real goals predicted both clean sheets
    # and goals better than xG alone, and better than giving goals more weight.
    W_GOALS = 0.25
    out = []
    for tc, g in tf.groupby('_tc'):
        g = g.copy()
        for c in ('xg', 'xga', 'gf', 'ga'):
            s = g[c]; pl = s.notna()
            post_sum = s[pl].rolling(10, min_periods=1).sum(); post_n = s[pl].rolling(10, min_periods=1).count()
            ps = pd.Series(np.nan, index=g.index); pn = pd.Series(np.nan, index=g.index); ps[pl] = post_sum; pn[pl] = post_n
            ps, pn = ps.shift(1).ffill().fillna(0), pn.shift(1).ffill().fillna(0)
            g[c + '_pre'] = (ps + K * LG) / (pn + K)
        g['xg_pre'] = (1 - W_GOALS) * g.xg_pre + W_GOALS * g.gf_pre; g['xga_pre'] = (1 - W_GOALS) * g.xga_pre + W_GOALS * g.ga_pre
        out.append(g)
    tf = pd.concat(out)[['season', 'fixture', '_tc', 'xg_pre', 'xga_pre']]
    h = h.merge(tf.rename(columns={'xg_pre': 'own_xgf', 'xga_pre': 'own_xga'}), on=['season', 'fixture', '_tc'], how='left')
    h = h.merge(tf.rename(columns={'_tc': '_oc', 'xg_pre': 'opp_xgf', 'xga_pre': 'opp_xga'}), on=['season', 'fixture', '_oc'], how='left')
    for c in ('own_xgf', 'own_xga', 'opp_xgf', 'opp_xga'): h[c] = h[c].fillna(LG)
    h['att_vs_def'] = h.own_xgf * h.opp_xga / LG          # how many goals his team should create in this match
    h['cs_chance'] = np.exp(-h.opp_xgf * h.own_xga / LG)  # rough clean-sheet chance: P(0 goals) for a Poisson with that mean
    # ---- defensive contributions and bonus
    # threshold for the 2 points: 10 actions (CBIT) for defenders, 12 (CBIRT) for midfielders and forwards; goalkeepers cannot score them
    thr = h.pos.map({2: 10, 3: 12, 4: 12})
    hit = (h.defensive_contribution >= thr).astype(float).where(h.defensive_contribution.notna() & thr.notna())
    gc = h.groupby('code', sort=False)
    for w in (5, 10):
        h[f'dc_hit_r{w}'] = hit.groupby(h.code).transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
    dmins = h.minutes.where(h.defensive_contribution.notna())
    dsum = h.defensive_contribution.groupby(h.code).transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    msum = dmins.groupby(h.code).transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    h['dc_p90_r10'] = (90 * dsum / msum).where(msum >= 270)
    played = h.minutes.where(h.minutes > 0)
    bany = (h.bonus > 0).astype(float).where(played.notna())
    h['bonus_rate_r10'] = bany.groupby(h.code).transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    bsum = h.bps.groupby(h.code).transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    m10 = h.minutes.groupby(h.code).transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    h['bps_p90_r10'] = (90 * bsum / m10).where(m10 >= 270)
    # does he do better against this opponent, or at home / away, than he normally does? Only games he played count.
    # Samples are tiny (2 to 8 games), so the difference is shrunk: with K=4 a player needs several games before it moves much.
    K = 4.0
    tp = h.total_points.where(h.minutes > 0)
    base = tp.groupby(h.code).transform(lambda x: x.shift(1).expanding().mean())
    for keys, nm in ((['code', '_oc'], 'h2h'), (['code', 'was_home'], 'venue')):
        grp = tp.groupby([h[k] for k in keys])
        s_ = grp.transform(lambda x: x.shift(1).expanding().sum()); n_ = grp.transform(lambda x: x.shift(1).expanding().count())
        h[f'{nm}_n'] = n_.fillna(0)
        h[f'{nm}_adj'] = ((s_.fillna(0) + K * base) / (n_.fillna(0) + K) - base).fillna(0)
    # scoring level: how many points 60+ minute players in this position have scored league-wide over the previous 6 gameweeks.
    # This lets the model adjust when the rules change scoring (2025-26 defensive contributions, 2026-27 bonus changes).
    lv = h[(~h.future) & (h.minutes >= 60)].groupby(['si', 'gw', 'pos']).total_points.mean().rename('m').reset_index().sort_values(['si', 'gw'])
    lv['pos_level'] = lv.groupby('pos').m.transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    last = lv.groupby('pos').apply(lambda g: g.m.tail(6).mean())       # for future gameweeks
    h = h.merge(lv[['si', 'gw', 'pos', 'pos_level']], on=['si', 'gw', 'pos'], how='left')
    h['pos_level'] = h.pos_level.fillna(h.pos.map(last))
    return h.drop(columns=['_xgi', '_tc', '_oc'])


def make_matrix(h):
    X = h[FEATURES].copy()
    X['pos'] = X['pos'].astype(float)
    return X
