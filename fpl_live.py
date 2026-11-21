"""Live gameweek scores from FPL's public data: every player's points so far, provisional bonus, and a team's live total.

FPL adds the official bonus (3-2-1 by BPS in each match) about an hour after a match. Until then we work it out from the live BPS,
using FPL's tie rules, and mark it provisional. A team's score follows FPL's rules: captain x2 (x3 with Triple Captain), Bench Boost
counts the bench, the vice-captain takes the armband if the captain does not play, and a starter who did not play is replaced by the
first bench player who did, keeping a legal formation (1 GK, 3+ DEF, 2+ MID, 1+ FWD).
"""
import fpl_engine as fe

LO = {1: 1, 2: 3, 3: 2, 4: 1}


def bonus_from_bps(bps):
    """{player: bps} for one match -> {player: bonus}. Rank 1 gets 3, rank 2 gets 2, rank 3 gets 1, ties share the rank
    (tie for first: 3, 3, then 1; tie for second: 3, 2, 2; tie for third: 3, 2, 1, 1), as in FPL's rules."""
    vals = sorted(bps.values(), reverse=True)
    out = {}
    for p, v in bps.items():
        rank = 1 + sum(1 for x in vals if x > v)
        if rank <= 3: out[p] = {1: 3, 2: 2, 3: 1}[rank]
    return out


def _stat(fx, name, side):
    return {r['element']: r['value'] for s in fx.get('stats', []) if s['identifier'] == name for r in s[side]}


def fixture_state(fx):
    if not fx.get('started'): return 'not started'
    if fx.get('finished') or fx.get('finished_provisional'): return 'finished'
    return 'live'


def players(live, fixtures, team_of):
    """Per player for the gameweek: points so far (official, plus provisional bonus where FPL has not added it), minutes, status and how he
    got his points. team_of: {player id: team id}. Returns {id: dict(pts, mins, status, bonus_prov, lines, fixtures)}."""
    by_team = {}
    for f in fixtures:
        by_team.setdefault(f['team_h'], []).append(f); by_team.setdefault(f['team_a'], []).append(f)
    prov = {}                                                     # fixture id -> {player: provisional bonus}
    for f in fixtures:
        if not f.get('started'): continue
        official = _stat(f, 'bonus', 'h') | _stat(f, 'bonus', 'a')
        if official: continue                                     # FPL has added the bonus for this match: it is in total_points already
        bps = _stat(f, 'bps', 'h') | _stat(f, 'bps', 'a')
        if bps: prov[f['id']] = bonus_from_bps(bps)
    names = {'minutes': 'played', 'goals_scored': 'goal', 'assists': 'assist', 'clean_sheets': 'clean sheet', 'goals_conceded': 'conceded',
             'saves': 'saves', 'bonus': 'bonus', 'yellow_cards': 'yellow', 'red_cards': 'red', 'own_goals': 'own goal', 'penalties_saved': 'pen saved',
             'penalties_missed': 'pen missed', 'defensive_contribution': 'def. contribution'}
    out = {}
    for e in live['elements']:
        i = e['id']; fxs = by_team.get(team_of.get(i), [])
        states = [fixture_state(f) for f in fxs]
        status = 'no match' if not fxs else ('finished' if all(s == 'finished' for s in states) else ('live' if 'live' in states or ('finished' in states) else 'not started'))
        bp = sum(prov.get(f['id'], {}).get(i, 0) for f in fxs)
        lines = []
        for x in e.get('explain', []):
            for s in x['stats']:
                if s['points']: lines.append((names.get(s['identifier'], s['identifier'].replace('_', ' ')), s['points']))
        if bp: lines.append(('bonus (provisional)', bp))
        out[i] = dict(pts=e['stats']['total_points'] + bp, mins=e['stats']['minutes'], status=status, bonus_prov=bp, lines=lines,
                      fixtures=[f['id'] for f in fxs])
    return out


def team_score(picks, chip, pl, pos_of, hits=0):
    """A team's live score. picks: FPL picks (element, position, multiplier, is_captain, is_vice_captain); chip: active chip or None;
    pl: from players(); pos_of: {player: 1-4}. Returns dict(total, rows, subs, left, playing, cap_now)."""
    picks = sorted(picks, key=lambda p: p['position'])
    get = lambda i, k, d=0: pl.get(i, {}).get(k, d)
    xi = [p['element'] for p in picks if p['position'] <= 11]; bench = [p['element'] for p in picks if p['position'] > 11]
    bb = chip == 'bboost'; cm = 3 if chip == '3xc' else 2
    played = lambda i: get(i, 'mins') > 0
    out_done = lambda i: get(i, 'status') in ('finished', 'no match') and not played(i)      # his matches are over and he did not play
    counting = list(xi) + (list(bench) if bb else [])
    subs = []
    if not bb:
        # FPL goes through the bench in order: each bench player who played replaces the first starter (in team order) who did not,
        # as long as the formation stays legal; the goalkeeper only swaps with the goalkeeper
        done_out = set()
        for b in bench:
            if not played(b): continue
            for k, s in enumerate(list(counting)):
                if s in done_out or not out_done(s) or (pos_of.get(b) == 1) != (pos_of.get(s) == 1): continue
                trial = counting[:k] + [b] + counting[k + 1:]
                cnt = {p: sum(1 for x in trial if pos_of.get(x) == p) for p in (1, 2, 3, 4)}
                if all(cnt[p] >= LO[p] for p in (1, 2, 3, 4)):
                    counting = trial; done_out.add(s); subs.append((s, b)); break
    cap = next((p['element'] for p in picks if p['is_captain']), None); vice = next((p['element'] for p in picks if p['is_vice_captain']), None)
    cap_now = cap
    if cap is not None and out_done(cap) and vice is not None and vice in counting and played(vice): cap_now = vice
    mult = {i: (1 if i in counting else 0) for i in xi + bench}
    if cap_now in mult and mult[cap_now]: mult[cap_now] = cm
    rows = [dict(id=i, mult=mult[i], pts=get(i, 'pts'), status=get(i, 'status', 'no match'), mins=get(i, 'mins'), bench=i in bench,
                 sub_in=any(b == i for _, b in subs), sub_out=any(s == i for s, _ in subs), cap=i == cap, vice=i == vice, cap_now=i == cap_now)
            for i in xi + bench]
    total = sum(r['pts'] * r['mult'] for r in rows) - hits
    left = sum(1 for i in counting if get(i, 'status') == 'not started')
    playing = sum(1 for i in counting if get(i, 'status') == 'live')
    return dict(total=total, rows=rows, subs=subs, left=left, playing=playing, cap_now=cap_now, cap=cap)


def load(gw):
    """Live data and fixtures for one gameweek (FPL's public endpoints)."""
    return fe.get(f'event/{gw}/live/'), fe.get(f'fixtures/?event={gw}')
