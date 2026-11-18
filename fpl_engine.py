"""Everything the app needs: predictions for upcoming gameweeks, loading a manager's team, and the transfer optimiser.

FPL rules used here (2026/27):
  squad 15 = 2 GK, 5 DEF, 5 MID, 3 FWD | max 3 per club | £100.0m budget
  XI: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD | captain x2 (Triple Captain x3)
  1 free transfer per gameweek, roll up to 5 | each extra transfer costs 4 points
  selling price = buy price + half of any rise (rounded down to £0.1m); falls are passed on in full
  Wildcard / Free Hit: unlimited transfers with no hits; saved free transfers are kept
"""
import json, math, time
import numpy as np, pandas as pd, requests, pulp
from pathlib import Path

ROOT = Path(__file__).parent
API = 'https://fantasy.premierleague.com/api/'
POS_NAME = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
SQUAD_NEED = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}; XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
HIT = 4
_s = requests.Session(); _s.headers['User-Agent'] = 'Mozilla/5.0 (personal FPL project)'


def get(path):
    r = _s.get(API + path, timeout=30); r.raise_for_status(); return r.json()


# ------------------------------------------------------------------ predictions
def availability(el, k):
    """Chance of playing. Gameweek k=0 is next week and uses the official flag; later weeks ease halfway back towards fit each week."""
    c = el.chance_of_playing_next_round.astype(float).fillna(100) / 100
    c = c.where(el.status != 'u', 0.0)              # 'u' = left the league / unavailable for the season
    return 1 - (1 - c) * (0.5 ** k)


def predictions(h, nets, sc, gb, boot, horizon=3):
    from fpl_model import predict
    import fpl_data as fd
    fut = h[h.future].copy()
    fut['xp'] = (predict(nets, sc, fut) + gb.predict(fut[getattr(gb, 'feature_list_', fd.FEATURES)].fillna(-1))) / 2
    fut['xp'] = fut.xp.clip(lower=0)
    el = pd.DataFrame(boot['elements']).set_index('id')
    gws = sorted(fut.gw.unique())[:horizon]
    out = pd.DataFrame(index=el.index)
    teams = {t['id']: t['short_name'] for t in boot['teams']}
    for k, g in enumerate(gws):
        f = fut[fut.gw == g]
        out[f'xp_{int(g)}'] = f.groupby('element').xp.sum().reindex(out.index).fillna(0) * availability(el, k)
        opp = f.assign(o=f.opponent_team.map(teams) + np.where(f.was_home, ' (H)', ' (A)')).groupby('element').o.agg(', '.join)
        out[f'opp_{int(g)}'] = opp.reindex(out.index).fillna('-')
    out['name'] = el.web_name; out['pos'] = el.element_type; out['team'] = el.team.map(teams); out['team_id'] = el.team
    out['price'] = el.now_cost / 10; out['status'] = el.status; out['news'] = el.news; out['chance'] = el.chance_of_playing_next_round
    out['selected'] = el.selected_by_percent.astype(float)
    return out[out.pos.isin([1, 2, 3, 4])], [int(g) for g in gws]


# ------------------------------------------------------------------ a manager's team
def sell_price(buy, now):
    return now if now <= buy else buy + math.floor((now - buy) / 2)   # in tenths of £m


def load_entry(eid, boot):
    """Return squad (with buying and selling prices), bank, free transfers and chips left, from public endpoints."""
    ev = [e for e in boot['events'] if e['finished'] or e['is_current']]
    cur = max(e['id'] for e in ev if e['finished']) if any(e['finished'] for e in ev) else 1
    entry = get(f'entry/{eid}/'); hist = get(f'entry/{eid}/history/')
    chips = {c['event']: c['name'] for c in hist['chips']}
    # the squad: use the last finished gameweek, but skip back past a Free Hit week (its squad is temporary)
    gw = cur
    while chips.get(gw) == 'freehit' and gw > 1: gw -= 1
    picks = get(f'entry/{eid}/event/{gw}/picks/')
    bank = hist['current'][-1]['bank']
    transfers = get(f'entry/{eid}/transfers/')
    fh_weeks = {e for e, n in chips.items() if n == 'freehit'}
    el = {e['id']: e for e in boot['elements']}
    start = entry['started_event']
    try:
        h = pd.read_csv(ROOT / 'data' / '2026-27' / 'history.csv', usecols=['element', 'round', 'value'])
        start_val = h[h['round'] == start].groupby('element').value.first().to_dict()
    except Exception:
        start_val = {}
    rows = []
    for p in picks['picks']:
        pid = p['element']; now = el[pid]['now_cost']
        tin = [t for t in transfers if t['element_in'] == pid and t['event'] not in fh_weeks]
        buy = max(tin, key=lambda t: t['time'])['element_in_cost'] if tin else start_val.get(pid, now - el[pid]['cost_change_start'])
        rows.append(dict(id=pid, buy=buy, now=now, sell=sell_price(buy, now), pos=el[pid]['element_type']))
    squad = pd.DataFrame(rows)
    return dict(name=entry['name'], manager=f"{entry['player_first_name']} {entry['player_last_name']}", squad=squad, bank=bank,
                free_transfers=free_transfers(hist, start, cur), chips_left=chips_left(hist, cur + 1), points=entry['summary_overall_points'],
                rank=entry['summary_overall_rank'], squad_gw=gw)


def free_transfers(hist, start, cur):
    """Replay the season: +1 free transfer each week (max 5), minus transfers used. Wildcard / Free Hit weeks use none."""
    chips = {c['event']: c['name'] for c in hist['chips']}
    used = {r['event']: r['event_transfers'] for r in hist['current']}
    ft = 1                                             # after the first deadline you have 1 for the next week
    for g in range(start + 1, cur + 1):
        if chips.get(g) not in ('wildcard', 'freehit'): ft = max(ft - used.get(g, 0), 0)
        ft = min(ft + 1, 5)
    return ft


def chips_left(hist, next_gw):
    half = (2, 19) if next_gw <= 19 else (20, 38)
    used = {c['name'] for c in hist['chips'] if half[0] <= c['event'] <= half[1]}
    names = {'wildcard': 'Wildcard', 'freehit': 'Free Hit', 'bboost': 'Bench Boost', '3xc': 'Triple Captain'}
    return [v for k, v in names.items() if k not in used]


# ------------------------------------------------------------------ optimiser
def decay(n):
    """Later gameweeks count a little less: predictions get less certain and plans change."""
    return [0.85 ** k for k in range(n)]


def best_xi(ids, P, gw):
    """Best legal starting XI, captain and vice-captain for one gameweek from a 15-man squad."""
    s = P.loc[ids].copy(); s['x'] = s[f'xp_{gw}']
    m = pulp.LpProblem('xi', pulp.LpMaximize); v = {i: pulp.LpVariable(f's{i}', cat='Binary') for i in ids}
    m += pulp.lpSum(v[i] * s.x[i] for i in ids); m += pulp.lpSum(v.values()) == 11
    for p in (1, 2, 3, 4):
        grp = [v[i] for i in ids if s.pos[i] == p]; m += pulp.lpSum(grp) >= XI_MIN[p]; m += pulp.lpSum(grp) <= XI_MAX[p]
    m.solve(pulp.PULP_CBC_CMD(msg=0))
    xi = [i for i in ids if v[i].value() > 0.5]
    order = s.loc[xi].sort_values('x', ascending=False).index
    bench_gk = [i for i in ids if i not in xi and s.pos[i] == 1]
    bench = bench_gk + list(s.loc[[i for i in ids if i not in xi and s.pos[i] != 1]].sort_values('x', ascending=False).index)
    return list(xi), int(order[0]), int(order[1]), bench


def optimise(P, gws, squad, bank, ft, max_transfers=1, chip=None, weights=None, roll_value=1.0, bench_weight=0.1, banned=(), forced=()):
    """Pick transfers that maximise expected points over the next gameweeks.

    squad: DataFrame with id, sell (tenths of £m). bank in tenths. chip: None, 'Wildcard' or 'Free Hit'.
    roll_value: points a saved free transfer is worth (so the optimiser won't make a 0.1-point transfer just because it is free).
    """
    if chip == 'Free Hit': gws = gws[:1]
    weights = weights or decay(len(gws))
    unlimited = chip in ('Wildcard', 'Free Hit')
    owned = set(squad.id); sellp = squad.set_index('id').sell
    w = dict(zip(gws, weights))
    score = sum(P[f'xp_{g}'] * w[g] for g in gws)
    pool = set(P[(score > 0.8 * len(gws)) & (P.status != 'u')].index) | owned
    pool = [i for i in pool if i not in banned or i in owned]
    Q = P.loc[pool]
    m = pulp.LpProblem('fpl', pulp.LpMaximize)
    x = {i: pulp.LpVariable(f'x{i}', cat='Binary') for i in pool}
    s = {(i, g): pulp.LpVariable(f's{i}_{g}', cat='Binary') for i in pool for g in gws}
    c = {(i, g): pulp.LpVariable(f'c{i}_{g}', cat='Binary') for i in pool for g in gws}
    n_in = pulp.lpSum(x[i] for i in pool if i not in owned)
    hits = pulp.LpVariable('hits', lowBound=0, upBound=15, cat='Integer'); used_ft = pulp.LpVariable('used_ft', lowBound=0, upBound=5)
    for p, n in SQUAD_NEED.items(): m += pulp.lpSum(x[i] for i in pool if Q.pos[i] == p) == n
    for t in Q.team_id.unique(): m += pulp.lpSum(x[i] for i in pool if Q.team_id[i] == t) <= 3
    m += pulp.lpSum(x[i] * int(round(Q.price[i] * 10)) for i in pool if i not in owned) <= bank + pulp.lpSum((1 - x[i]) * int(sellp[i]) for i in owned)
    for i in forced: m += x[i] == 1
    for i in banned:
        if i in x: m += x[i] == 0
    if not unlimited:
        m += n_in <= max_transfers
        m += hits >= n_in - ft
        m += used_ft >= n_in - hits; m += used_ft <= ft
    else:
        m += hits == 0; m += used_ft == 0
    obj = []
    for g in gws:
        m += pulp.lpSum(s[i, g] for i in pool) == 11; m += pulp.lpSum(c[i, g] for i in pool) == 1
        for p in (1, 2, 3, 4):
            grp = [s[i, g] for i in pool if Q.pos[i] == p]; m += pulp.lpSum(grp) >= XI_MIN[p]; m += pulp.lpSum(grp) <= XI_MAX[p]
        for i in pool:
            m += s[i, g] <= x[i]; m += c[i, g] <= s[i, g]
            xp = float(Q.loc[i, f'xp_{g}'])
            obj.append(w[g] * xp * (s[i, g] + c[i, g] + bench_weight * (x[i] - s[i, g])))
    rv = roll_value if ft < 5 else 0.0
    m += pulp.lpSum(obj) - HIT * hits - rv * used_ft
    m.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    if pulp.LpStatus[m.status] != 'Optimal':
        raise ValueError('No legal squad can be reached with these settings (check the 3-per-club rule, the budget and any players you forced or banned).')
    new = [i for i in pool if x[i].value() > 0.5]
    out_ = sorted(owned - set(new)); in_ = sorted(set(new) - owned)
    bank_after = bank + sum(int(sellp[i]) for i in out_) - sum(int(round(P.price[i] * 10)) for i in in_)
    return dict(squad=new, out=out_, inn=in_, hits=int(round(hits.value() or 0)), bank_after=bank_after, gws=gws)


def squad_points(P, ids, gws, weights=None, bench_weight=0.1):
    """Expected points of a squad played optimally each week (best XI + captain), weighted like the optimiser."""
    weights = weights or decay(len(gws)); tot, per = 0.0, {}
    for g, w in zip(gws, weights):
        xi, cap, vice, bench = best_xi(ids, P, g)
        pts = P.loc[xi, f'xp_{g}'].sum() + P.loc[cap, f'xp_{g}'] + bench_weight * P.loc[bench, f'xp_{g}'].sum()
        per[g] = P.loc[xi, f'xp_{g}'].sum() + P.loc[cap, f'xp_{g}']; tot += w * pts
    return tot, per
