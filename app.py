"""FPL Points Predictor: your projected points for the coming gameweek, the best XI and captain, and transfer suggestions.

Run locally:  streamlit run app.py
The app only reads data/predictions.parquet, data/meta.json and data/2026-27/bootstrap.json (made by make_predictions.py).
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
UK = ZoneInfo('Europe/London')           # deadlines shown in UK time (GMT in winter, BST in summer)
import numpy as np, pandas as pd, plotly.graph_objects as go, requests, streamlit as st
import fpl_engine as fe, ui
try:                                     # browser storage: remembers your team, league and confirmed transfers between visits
    from streamlit_js_eval import streamlit_js_eval as js_eval
except ImportError:
    js_eval = None
LS_KEY = 'fpl_predictor_v1'

ROOT = Path(__file__).parent
st.set_page_config(page_title='FPL Points Predictor', page_icon='⚽', layout='wide', initial_sidebar_state='expanded')
st.markdown(ui.CSS, unsafe_allow_html=True)
# ask the browser for what it saved last time, as the very first element of the page (its answer triggers one rerun)
if js_eval is not None and not st.session_state.get('_ls_ok'):
    st.session_state['_ls_raw'] = js_eval(js_expressions=f"localStorage.getItem('{LS_KEY}') || ''", key='ls_get')
POS = ui.POS


@st.cache_data(ttl=3600)
def load():
    P = pd.read_parquet(ROOT / 'data' / 'predictions.parquet'); meta = json.load(open(ROOT / 'data' / 'meta.json'))
    boot = json.load(open(ROOT / 'data' / '2026-27' / 'bootstrap.json'))
    gws = sorted(int(c[3:]) for c in P.columns if c.startswith('xp_'))
    return P, meta, boot, gws


@st.cache_data(ttl=3600, show_spinner=False)
def season_stats():
    """This season per player, from match-by-match history: totals, and the points each part earned (scored per match, as FPL does)."""
    H = pd.read_csv(ROOT / 'data' / '2026-27' / 'history.csv', low_memory=False)
    H['pos'] = H.element.map(P.pos)
    H['save_pts'] = H.saves // 3                                                     # 1 point per 3 saves in a match
    H['gc_pts'] = np.where(H.pos.isin([1, 2]), H.goals_conceded // 2, 0)             # GK / DEF lose 1 point per 2 conceded in a match
    H['cs_pts'] = H.clean_sheets * H.pos.map({1: 4, 2: 4, 3: 1, 4: 0}).fillna(0)
    thr = H.pos.map({2: 10, 3: 12, 4: 12})
    H['dc_pts'] = np.where(thr.notna() & (H.defensive_contribution >= thr), 2, 0)  # 2 points for reaching the threshold
    for c in ('expected_goals', 'expected_assists'): H[c] = pd.to_numeric(H[c], errors='coerce').fillna(0)
    g = H.groupby('element')
    S = pd.DataFrame({'minutes': g.minutes.sum(), 'starts': g.starts.sum(), 'apps': g.minutes.apply(lambda x: (x > 0).sum()),
                      'goals': g.goals_scored.sum(), 'assists': g.assists.sum(), 'xg': g.expected_goals.sum(), 'xa': g.expected_assists.sum(),
                      'cs': g.clean_sheets.sum(), 'cs_pts': g.cs_pts.sum(), 'gc': g.goals_conceded.sum(), 'gc_pts': g.gc_pts.sum(),
                      'saves': g.saves.sum(), 'save_pts': g.save_pts.sum(), 'pen_saved': g.penalties_saved.sum(), 'dc_pts': g.dc_pts.sum(),
                      'bonus': g.bonus.sum(), 'pts': g.total_points.sum(), 'yc': g.yellow_cards.sum(), 'rc': g.red_cards.sum()})
    S['ppg'] = S.pts / S.apps.where(S.apps > 0); S['saves90'] = 90 * S.saves / S.minutes.where(S.minutes >= 90)
    S['cards'] = S.yc.astype(int).astype(str) + ' / ' + S.rc.astype(int).astype(str)
    return S


@st.cache_data(ttl=3600, show_spinner=False)
def last5_cards():
    """{player id: html} with his last 5 matches: opponent, minutes and how he got his points, using FPL's scoring for his position."""
    H = pd.read_csv(ROOT / 'data' / '2026-27' / 'history.csv', low_memory=False).sort_values('kickoff_time')
    sc = boot['game_config']['scoring']; pk = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    sn = {t['id']: t['short_name'] for t in boot['teams']}
    out = {}
    for i, g in H.groupby('element'):
        if i not in P.index: continue
        p_ = int(P.pos[i]); k = pk[p_]; rows = ''
        for r in g.tail(5).iloc[::-1].itertuples():
            m = int(r.minutes); parts = []
            if m > 0: parts.append(('played', sc['long_play'] if m >= 60 else sc['short_play']))
            parts += [('goal' + ('s' if r.goals_scored > 1 else ''), r.goals_scored * sc['goals_scored'][k]), ('assist' + ('s' if r.assists > 1 else ''), r.assists * sc['assists']),
                      ('clean sheet', r.clean_sheets * sc['clean_sheets'][k]), ('saves', (r.saves // 3) * sc['saves']),
                      ('pen saved', r.penalties_saved * sc['penalties_saved']), ('conceded', (r.goals_conceded // 2) * sc['goals_conceded'][k]),
                      ('def. contribution', sc['defensive_contribution'][k] if p_ != 1 and pd.notna(r.defensive_contribution)
                       and r.defensive_contribution >= {2: 10, 3: 12, 4: 12}[p_] else 0),
                      ('bonus', r.bonus * sc['bonus']), ('yellow', r.yellow_cards * sc['yellow_cards']), ('red', r.red_cards * sc['red_cards']),
                      ('own goal', r.own_goals * sc['own_goals']), ('pen missed', r.penalties_missed * sc['penalties_missed'])]
            parts = [(n_, int(v)) for n_, v in parts if v]
            other = int(r.total_points) - sum(v for _, v in parts)
            if other: parts.append(('other', other))
            how = ' · '.join(f'{n_} {v:+d}' if v < 0 else f'{n_} {v}' for n_, v in parts) or 'did not play'
            gf, ga = (r.team_h_score, r.team_a_score) if r.was_home else (r.team_a_score, r.team_h_score)
            rows += (f'<tr><td>GW{r.round}</td><td style="white-space:nowrap">{sn.get(r.opponent_team, "?")} ({"H" if r.was_home else "A"}) {gf:.0f}-{ga:.0f}</td>'
                     f'<td>{m}\'</td><td class="how">{how}</td><td class="tot">{int(r.total_points)}</td></tr>')
        out[int(i)] = f'<table><tr><th>GW</th><th>Match</th><th>Mins</th><th>How he scored</th><th style="text-align:right">Pts</th></tr>{rows}</table>'
    return out


@st.cache_data(ttl=600, show_spinner='Loading your team from FPL...')
def load_team(eid, _boot):
    return fe.load_entry(eid, _boot)


@st.cache_data(show_spinner=False)
def lineup(ids, gw, boost=0.0):     # boost is part of the cache key: projections change with it
    return fe.best_xi(list(ids), P, gw)


@st.cache_data(show_spinner=False)
def run_opt(squad_json, bank, ft, k, chip, roll, banned, keep, gws, boost=0.0):
    from io import StringIO
    sq = pd.read_json(StringIO(squad_json))
    return fe.optimise(P, list(gws), sq, bank, ft, max_transfers=k, chip=chip, roll_value=roll, banned=list(banned), forced=list(keep))


P, meta, boot, ALL_GWS = load(); P = P.copy()
# ---- which gameweek is next, from the deadlines: the moment a deadline passes the app moves on to the next gameweek by itself,
# without waiting for the update job (the prediction file already covers every remaining gameweek).
# FPL_NOW=2026-11-21T14:00:00Z pretends it is another time (for testing).
import os
NOW = datetime.fromisoformat(os.environ['FPL_NOW'].replace('Z', '+00:00')) if os.environ.get('FPL_NOW') else datetime.now(timezone.utc)
DL = {e['id']: datetime.fromisoformat(e['deadline_time'].replace('Z', '+00:00')) for e in boot['events'] if e.get('deadline_time')}
ALL_GWS = [g for g in ALL_GWS if g not in DL or DL[g] > NOW] or ALL_GWS[-1:]
G0 = ALL_GWS[0]
LIVE_GW = max((g for g, d in DL.items() if d <= NOW), default=None)     # the latest gameweek whose deadline has passed (live, or just finished)
FPL_UPDATING = bool(LIVE_GW) and (NOW - DL[LIVE_GW]).total_seconds() < 75 * 60   # FPL's own site is down for about 30 minutes after a deadline


@st.cache_data(ttl=900, show_spinner=False)
def live_news():
    """Injury news, chance of playing, prices and ownership straight from FPL, refreshed every 15 minutes (no update job needed)."""
    b_ = fe.get('bootstrap-static/')
    return pd.DataFrame(b_['elements']).set_index('id'), datetime.now(timezone.utc)


# ---- the latest team news and prices: reapply each player's chance of playing to the model's predictions, so a new injury flag
# (or a player back to fit) shows straight away. The model's points before the chance-of-playing adjustment are in the file (xpr_ / _r_).
NEWS_AT = None
if any(c.startswith('xpr_') for c in P.columns) and not os.environ.get('FPL_NOW'):
    try:
        el_live, NEWS_AT = live_news()
        el_live = el_live.reindex(P.index)
        ok_ = el_live.status.notna()
        for k_, g_ in enumerate(ALL_GWS):
            av_ = fe.availability(el_live, k_).where(ok_, np.nan)
            if f'xpr_{g_}' in P: P[f'xp_{g_}'] = (P[f'xpr_{g_}'] * av_).fillna(P[f'xp_{g_}'])
            for c_ in ('p_goal', 'p_assist', 'p_cs', 'p_60', 'p_bonus', 'p_dc', 'xb'):
                if f'{c_}_r_{g_}' in P: P[f'{c_}_{g_}'] = (P[f'{c_}_r_{g_}'] * av_).fillna(P[f'{c_}_{g_}'])
        P['chance'] = el_live.chance_of_playing_next_round.where(ok_, P.chance)
        P['news'] = el_live.news.where(ok_, P.news); P['status'] = el_live.status.where(ok_, P.status)
        P['price'] = (el_live.now_cost / 10).where(ok_, P.price)
        P['selected'] = el_live.selected_by_percent.astype(float).where(ok_, P.selected)
        if 'form' in P: P['form'] = el_live.form.astype(float).where(ok_, P.form)
    except Exception:
        NEWS_AT = None                                        # FPL unreachable (or updating after a deadline): keep the last saved news
with st.sidebar.expander('Model settings'):
    BOOST = st.slider('Form boost', 0, 50, 0, 5, format='%d%%', key='boost',
                      help='Mixes FPL form (points per match over the last 30 days) into every projection. Testing on unseen gameweeks found 0% predicts best: hot streaks mostly fade.') / 100
if BOOST:
    c = (P.chance.astype(float).fillna(100) / 100).where(P.status != 'u', 0.0)
    for k, g in enumerate(ALL_GWS):
        nfix = P[f'opp_{g}'].map(lambda o: 0 if o == '-' else o.count(',') + 1)
        P[f'xp_{g}'] = P[f'xp_{g}'] + BOOST * (P.form * nfix * (1 - (1 - c) * 0.5 ** k) - P[f'xp_{g}'])
@st.cache_data(ttl=60, show_spinner=False)
def live_data(g_):
    """FPL's live data and fixtures for a gameweek. Shared by every visitor and fetched at most once a minute."""
    import fpl_live as fl
    return fl.load(g_)


@st.cache_data(ttl=120, show_spinner='Loading your league\'s live scores...')
def live_league(lid_, g_, n_=20):
    """The top n_ of a classic league with each manager's picks for gameweek g_ (public once its deadline has passed)."""
    from concurrent.futures import ThreadPoolExecutor
    d_ = fe.get(f'leagues-classic/{int(lid_)}/standings/'); rows_ = d_['standings']['results'][:n_]
    def one(r):
        try: pk_ = fe.get(f'entry/{r["entry"]}/event/{g_}/picks/')
        except Exception: pk_ = None
        return dict(entry=r['entry'], entry_name=r['entry_name'], player_name=r['player_name'], rank=r['rank'], total=r['total'], event_total=r['event_total'], picks=pk_)
    with ThreadPoolExecutor(max_workers=6) as pool_: ms_ = list(pool_.map(one, rows_))
    return dict(name=d_['league']['name'], managers=ms_)


# is the latest gameweek still being played? (deadline passed and some of its matches not finished)
LIVE_WINDOW = LIVE_ON = False
if LIVE_GW:
    try:
        import fpl_live as _fl
        _st = [_fl.fixture_state(f_) for f_ in live_data(LIVE_GW)[1]]
        LIVE_WINDOW = any(s_ != 'finished' for s_ in _st); LIVE_ON = 'live' in _st
    except Exception:
        LIVE_WINDOW = FPL_UPDATING              # FPL down for its post-deadline update: the gameweek has started
deadline = DL.get(G0) or datetime.fromisoformat(meta['deadline'].replace('Z', '+00:00'))
left = deadline - NOW
countdown = f'{left.days}d {left.seconds // 3600}h' if left.total_seconds() > 0 else 'deadline passed'
HORIZONS = {'Next GW': 1, 'Next 2': 2, 'Next 3': 3, 'Next 6': 6, 'Season': len(ALL_GWS)}   # Season = all remaining gameweeks
for n in HORIZONS.values(): P[f'h{n}'] = sum(P[f'xp_{g}'] for g in ALL_GWS[:n])
LAB = {i: f'{r["name"]} ({r.team}, £{r.price:.1f}m)' for i, r in P.iterrows()}
html = lambda s: st.markdown(s, unsafe_allow_html=True)
sec = lambda t, s='': html(f'<div class="sec">{t}<small>{s}</small></div>')

# ------------------------------------------------------------------ sidebar: the team
def click_pitch(xi, bench, cap, vice, g, ins=(), outs=(), moved=(), clickable=True, tag='r', order=None, diffs=(), tip='Change {name}: transfer him, bench him or make him captain'):
    """A formation made of Streamlit columns, styled like the Transfers pitches. Both Planner pitches use it, so they are the same size.
    clickable: each name tag is a button (returns the id clicked). ins / outs / moved: highlighted green / red / cyan (lineup change)."""
    xp = P[f'xp_{g}']; clicked = None

    def cell(i, col):
        nonlocal clicked
        r = P.loc[i]
        b = '<div class="badge">C</div>' if i == cap else ('<div class="badge v">V</div>' if i == vice else '')
        f = '' if pd.isna(r.chance) or r.chance >= 100 else f'<div class="flag {"r" if r.chance <= 25 else ""}">!</div>'
        kind = 'in' if i in ins else ('out' if i in outs else ('mv' if i in moved else ('diff' if i in diffs else '')))
        shirt_ = f'<div class="pshirt {"faded" if i in outs else ""}">{b}{f}<img src="{ui.shirt(r.team_code, r.pos == 1)}"></div>'
        v = xp[i] * (2 if i == cap else 1)
        below = f'<div class="ppt">{v:.1f} pts</div><div class="pfx">{ui.esc(r[f"opp_{g}"])}</div>'
        if clickable:
            col.markdown(shirt_, unsafe_allow_html=True)
            if col.button(r['name'], key=f'pp{kind}_{tag}_{g}_{i}', help=tip.format(name=r['name']), use_container_width=True): clicked = i
            col.markdown(below, unsafe_allow_html=True)
        else:   # one block: shirt, name, points, fixture
            col.markdown(shirt_ + f'<div class="pnm {kind}">{ui.esc(r["name"])}</div>' + below, unsafe_allow_html=True)
    with st.container(key=f'plpitch_{tag}_{g}'):
        for p in (1, 2, 3, 4):
            ids_ = [i for i in xi if P.pos[i] == p]
            if order: ids_ = sorted(ids_, key=lambda i: order.index(i) if i in order else 99)   # same spot on both pitches
            cols = st.columns(len(ids_))
            for i, col in zip(ids_, cols): cell(i, col)
    with st.container(key=f'plbench_{tag}_{g}'):
        html('<div class="blab">BENCH</div>')
        cols = st.columns(len(bench))
        for i, col in zip(bench, cols): cell(i, col)
    return clicked


def set_made(lst):
    """Save the transfers you already made, for THIS team only (session + page address, so a refresh keeps them)."""
    tid_ = str(st.session_state.get('eid', 'manual')).strip()
    st.session_state['made'] = lst
    st.session_state.setdefault('made_by_team', {})[tid_] = lst
    if lst: st.query_params['made'] = f'{tid_}~' + ','.join(f'{o}-{i}' for o, i in lst)
    elif 'made' in st.query_params: del st.query_params['made']


def load_made(tid_):
    """Confirmed transfers for this team: from this session, or from the page address if it was saved for this same team."""
    by = st.session_state.setdefault('made_by_team', {})
    if tid_ in by: return list(by[tid_])
    raw = st.query_params.get('made', '')
    if '~' not in raw: return []                                   # old links (no team) are ignored, so they can never leak into another team
    owner, pairs_ = raw.split('~', 1)
    if owner != tid_: return []
    return [tuple(int(x) for x in p_.split('-')) for p_ in pairs_.split(',') if '-' in p_]


def club_over(team_, new):
    """Clubs with more than 3 players in your squad after these transfers (FPL allows at most 3 per club): {club: count}."""
    ids_ = [int(i) for i in team_['squad'].id]
    for o, i in new:
        if o in ids_: ids_[ids_.index(o)] = i
    c = pd.Series([P.team[i] for i in ids_ if i in P.index]).value_counts()
    return c[c > 3].to_dict()


def bank_after(team_, new):
    """Bank after these transfers, from your current team (selling prices for players you own, today's price for players bought)."""
    sellp = team_['squad'].set_index('id').sell; bank = team_['bank']
    for o, i in new:
        bank += int(sellp[o]) if o in sellp.index else int(round(P.price[o] * 10))
        bank -= int(round(P.price[i] * 10))
    return bank


def confirm_block(new, made_now, team_, gw, after=None, key='cf'):
    """The 'are you sure?' content: the moves, free transfers used, any hit, the money, and that it cannot be undone."""
    ft_left = team_['free_transfers']; n = len(new); free_used = min(n, ft_left); hits = n - free_used
    bk = bank_after(team_, new)
    card = lambda i, kind: (f'<div class="mv {kind}"><img src="{ui.shirt(P.team_code[i], P.pos[i] == 1)}"><div><div class="bigname">{ui.esc(P.name[i])}</div>'
                            f'<div class="m">{P.team[i]} · {POS[P.pos[i]]} · £{P.price[i]:.1f}m</div></div></div>')
    html('<div class="note" style="margin-bottom:6px">You are telling the app you have made these transfers for GW%d in the FPL app:</div>' % gw)
    for o, i in new:
        html(f'<div class="move" style="margin:4px 0">{card(o, "out")}<div class="arrow">→</div>{card(i, "in")}</div>')
    if bk < 0:
        html(f'<div class="card" style="padding:12px 14px;border-color:#e90052;margin:8px 0"><div class="v" style="color:#e90052;font-size:1.2rem">Over budget by £{-bk / 10:.1f}m</div>'
             f'<div class="l" style="font-size:.88rem">You have £{team_["bank"] / 10:.1f}m in the bank. '
             + ('Together these transfers cost more than you have, so FPL would not allow them. Go back and change one, or add a transfer that frees up money.'
                if n > 1 else 'FPL would not allow this transfer, so it cannot be confirmed. Go back and pick a cheaper player, or add a transfer that frees up money.')
             + '</div></div>')
    else:
        html(f'<div class="note" style="margin:6px 0">Bank afterwards: <b style="color:#00ff87">£{bk / 10:.1f}m</b></div>')
    over = club_over(team_, new)
    if over:
        html(f'<div class="card" style="padding:12px 14px;border-color:#e90052;margin:8px 0"><div class="v" style="color:#e90052;font-size:1.2rem">'
             + ', '.join(f'{n_} players from {c_}' for c_, n_ in over.items()) + '</div><div class="l" style="font-size:.88rem">FPL allows at most 3 players from one club, '
             'so these transfers cannot be confirmed. Go back and change one, or add a transfer that sells one of them.</div></div>')
    if hits:
        html(f'<div class="card" style="padding:12px 14px;border-color:#e90052;margin:8px 0"><div class="v" style="color:#e90052;font-size:1.2rem">−{fe.HIT * hits} points</div>'
             f'<div class="l" style="font-size:.88rem">You have {ft_left} free transfer{"s" if ft_left != 1 else ""} left, so {hits} of these cost{"s" if hits == 1 else ""} a 4-point hit. '
             f'It comes off your GW{gw} score everywhere in the app.</div></div>')
    else:
        html(f'<div class="note" style="margin:4px 0 8px 0">Uses {free_used} of your {ft_left} free transfer{"s" if ft_left != 1 else ""}. No hit.</div>')
    html('<div class="note" style="margin:6px 0 10px 0">⚠️ <b style="color:#fff">This cannot be undone here</b>, just like a confirmed transfer in FPL. '
         'Not sure yet? Try it in the <b>Planner</b> first: you can change your mind there as often as you like.</div>')
    a, b = st.columns(2)
    if a.button('Yes, confirm', type='primary', use_container_width=True, key=f'{key}_yes', disabled=bk < 0 or bool(over)):
        set_made(made_now + list(new))
        if after: after()
        st.rerun()
    return b


@st.dialog('Add the transfers you have made', width='large')
def made_dialog(team_, made_now, gw):
    """Add one or more transfers: tap who you sold, pick who you bought, repeat; then review them all and confirm. All in one pop-up."""
    S = st.session_state
    sq = team_['squad'].set_index('id'); base = [int(i) for i in sq.index]
    bk = [tuple(int(x) for x in p_) for p_ in S.setdefault('md_basket', [])]          # (sold, bought) pairs added so far
    now_ids = list(base)
    for o_, i_b in bk: now_ids[now_ids.index(o_)] = i_b                                # your squad after them
    clear = lambda: [S.pop(k, None) for k in ('md_out', 'md_n', 'md_basket', 'md_review')]
    card = lambda i, kind: (f'<div class="mv {kind}" style="padding:5px 10px"><img src="{ui.shirt(P.team_code[i], P.pos[i] == 1)}"><div><div class="bigname">{ui.esc(P.name[i])}</div>'
                            f'<div class="m">{P.team[i]} · {POS[P.pos[i]]} · £{P.price[i]:.1f}m</div></div></div>')
    # the transfers added so far, each with Remove
    if bk and not S.get('md_review'):
        bnow = bank_after(team_, bk)
        html(f'<div class="note" style="margin-bottom:4px"><b style="color:#fff">Your transfers so far ({len(bk)})</b> · bank after them '
             + (f'<b style="color:#00ff87">£{bnow / 10:.1f}m</b>' if bnow >= 0 else
                f'<b style="color:#e90052">−£{-bnow / 10:.1f}m</b>: over budget. Add or change a transfer to free up £{-bnow / 10:.1f}m before you confirm.') + '</div>')
        for c_, n_ in club_over(team_, bk).items():
            html(f'<div class="note" style="margin-bottom:4px;color:#e90052"><b>{n_} players from {c_}</b>: FPL allows 3. Sell one of them before you confirm.</div>')
        for k, (a_, b_) in enumerate(bk):
            r1, r2 = st.columns([6, 1.2], vertical_alignment='center')
            r1.markdown(f'<div class="move" style="margin:2px 0">{card(a_, "out")}<div class="arrow">→</div>{card(b_, "in")}</div>', unsafe_allow_html=True)
            if r2.button('Remove', key=f'md_rm_{k}', use_container_width=True):
                S['md_basket'] = [list(p_) for j, p_ in enumerate(bk) if j != k]; S['md_out'] = None; st.rerun(scope='fragment')
    # last step: review all of them and confirm
    if S.get('md_review') and bk:
        b = confirm_block(bk, made_now, team_, gw, after=clear, key='mdcf')
        if b.button('Back', use_container_width=True, key='md_back'):
            S['md_review'] = False; st.rerun(scope='fragment')
        return
    o = S.get('md_out')
    if o is not None and o not in now_ids: o = S['md_out'] = None
    # step 1: who did you sell? tap him on your team's pitch (it already shows the transfers added above)
    if o is None:
        html('<div class="note" style="margin:8px 0 6px 0"><b style="color:#fff">' + ('Add another transfer:' if bk else 'Step 1.') +
             '</b> tap the name of the player you sold.' + (' Or, if that is all of them, review and confirm.' if bk else '') + '</div>')
        if bk and st.button(f'Review and confirm {len(bk)} transfer{"s" if len(bk) > 1 else ""}', type='primary', use_container_width=True, key='md_review_btn'):
            S['md_review'] = True; st.rerun(scope='fragment')
        xi_, cap_, vice_, bench_ = lineup(tuple(sorted(now_ids)), gw, BOOST)
        hit_ = click_pitch(xi_, bench_, cap_, vice_, gw, tag=f'md{len(bk)}', order=now_ids, tip='I sold {name}', ins=[b_ for _, b_ in bk])
        if hit_ is not None:
            S['md_out'] = int(hit_); st.rerun(scope='fragment')
        return
    # step 2: who did you buy? (selling a player you just bought here replaces that transfer)
    orig = next((a_ for a_, b_ in bk if b_ == o), o); rest = [p_ for p_ in bk if p_[1] != o]
    money = bank_after(team_, rest) + int(sq.sell[orig])
    top = st.columns([5, 1.2])
    top[0].markdown(f'<div class="mv out" style="padding:6px 10px"><img src="{ui.shirt(P.team_code[orig], P.pos[orig] == 1)}"><div><div class="bigname">Sold: {ui.esc(P.name[orig])}</div>'
                    f'<div class="m">{P.team[orig]} · {POS[P.pos[orig]]} · sells for £{sq.sell[orig] / 10:.1f}m · money available £{money / 10:.1f}m</div></div></div>', unsafe_allow_html=True)
    if top[1].button('Change', key='md_change_out', use_container_width=True):
        S['md_out'] = None; st.rerun(scope='fragment')
    html('<div class="note" style="margin:8px 0 4px 0"><b style="color:#fff">Step 2.</b> Pick the player you bought.</div>')
    cand = P[(P.pos == P.pos[orig]) & ~P.index.isin(now_ids) & (P.index != orig)].copy()
    f1, f2, f3 = st.columns([2, 1.6, 1.6])
    teams = f1.multiselect('Teams', sorted(cand.team.unique()), placeholder='All teams', key='md_tm')
    mx = f2.slider('Max price (£m)', 3.5, 16.0, 16.0, 0.5, key='md_mx')
    q = f3.text_input('Search', placeholder='Name', key='md_q')
    if teams: cand = cand[cand.team.isin(teams)]
    cand = cand[cand.price <= mx + 1e-9]
    if q: cand = cand[cand.name.str.contains(q, case=False)]
    cand = cand.sort_values(['selected', 'price'], ascending=False)        # most-owned first: the players people usually buy
    n = S.get('md_n', 12)
    # how many you would have from each club after this transfer (the player you sold no longer counts)
    club_n = pd.Series([P.team[x] for x in now_ids if x != o and x in P.index]).value_counts()
    for i, r_ in cand.head(n).iterrows():
        a, b = st.columns([6, 1.3])
        short = round(r_.price * 10) - money
        n_club = int(club_n.get(r_.team, 0)) + 1
        # any player can be picked: only the total of all your transfers has to fit the budget (checked when you confirm)
        a.markdown(f'<div class="mv in" style="padding:5px 10px"><img src="{ui.shirt(r_.team_code, r_.pos == 1)}"><div><div class="bigname">{ui.esc(r_["name"])} '
                   f'<span class="note">{r_.team}</span></div><div class="m">£{r_.price:.1f}m · owned {r_.selected:.0f}%'
                   + (f' · <span style="color:#ffab1b">£{short / 10:.1f}m more than you have now: other transfers must free it up</span>' if short > 0 else '')
                   + (f' · <span style="color:#ffab1b">your {n_club}th from {r_.team}: FPL allows 3, so sell one of them too</span>' if n_club > 3 else '')
                   + '</div></div></div>', unsafe_allow_html=True)
        if b.button('Transfer', key=f'md_i_{i}', type='primary', use_container_width=True):
            S['md_basket'] = [list(p_) for p_ in rest] + [[int(orig), int(i)]]; S['md_out'] = None; S['md_n'] = 12; st.rerun(scope='fragment')
    if cand.empty: st.info('No players match.')
    if len(cand) > n and st.button('Show more', key='md_more'): S['md_n'] = n + 12; st.rerun(scope='fragment')


def apply_made(team_, made_):
    """Your squad after the transfers you already made: new players in, bank and free transfers updated. Returns (team, the transfers that still apply)."""
    orig = [int(i) for i in team_['squad'].id]                     # your GW10 squad, before any confirmed GW11 transfers
    base = dict(orig_ids=orig, orig_bank=team_['bank'], orig_ft=team_['free_transfers'], orig_squad=team_['squad'].copy())
    if not made_: return dict(team_, made=[], made_hits=0, **base), []
    sq = team_['squad'].copy(); bank = team_['bank']; ok = []
    for o, i in made_:
        if o not in set(sq.id) or i in set(sq.id) or i not in P.index or P.pos[o] != P.pos[i]: continue
        row = sq.index[sq.id == o][0]; now_ = int(round(P.price[i] * 10))
        bank += int(sq.loc[row, 'sell']) - now_
        sq.loc[row, ['id', 'buy', 'now', 'sell', 'pos']] = [i, now_, now_, now_, P.pos[i]]
        ok.append((o, i))
    sq['id'] = sq.id.astype(int)
    ft = team_['free_transfers']; hits = max(len(ok) - ft, 0)
    return dict(team_, squad=sq.reset_index(drop=True), bank=bank, free_transfers=max(ft - len(ok), 0), made=ok, made_hits=hits, **base), ok


def remember(key, qp):
    """Streamlit forgets an input's value when it is not on screen (another tab). Keep a copy, and put it in the page address so a refresh keeps it too."""
    store = f'_{key}'
    if store not in st.session_state: st.session_state[store] = st.query_params.get(qp, '') or str(st.session_state.get(key, '') or '')
    if str(st.session_state.get(key, '') or '').strip() == '' and st.session_state[store]:
        st.session_state[key] = st.session_state[store]          # the box was wiped while you were on another tab: put the value back


def parse_team_id(text):
    """The team ID from whatever was typed: the number itself, or any FPL link containing /entry/<id>/."""
    import re
    t = str(text or '').strip()
    m = re.search(r'/entry/(\d+)', t) or re.fullmatch(r'#?(\d+)', t) or re.search(r'(\d{3,})', t)
    return m.group(1) if m else t


def set_team_id(v):
    """Used by 'find your team in a league': fill the team ID box and remember it."""
    st.session_state['eid'] = str(v); remember_save('eid', 'team')


@st.cache_data(ttl=1800, show_spinner='Searching the league...')
def league_entries(lid_, pages=20):
    """Every team in a classic league (first `pages` x 50), for finding your own team by name."""
    out = []
    for p_ in range(1, pages + 1):
        d_ = fe.get(f'leagues-classic/{int(lid_)}/standings/?page_standings={p_}')
        out += [(r['entry'], r['entry_name'], r['player_name']) for r in d_['standings']['results']]
        if not d_['standings'].get('has_next'): break
    return d_['league']['name'], out


def remember_save(key, qp):
    if key == 'eid': st.session_state['eid'] = parse_team_id(st.session_state.get('eid', ''))   # a pasted FPL link becomes the ID
    v = str(st.session_state.get(key, '')).strip(); st.session_state[f'_{key}'] = v
    if v.isdigit(): st.query_params[qp] = v
    elif qp in st.query_params: del st.query_params[qp]


def ls_read():
    """Read what this browser saved last time (team ID, league ID, confirmed transfers for the current gameweek).
    The browser answers on the next rerun, so the first run of a visit sees nothing; the app reruns by itself when the answer arrives."""
    S = st.session_state
    if js_eval is None or S.get('_ls_ok'): return
    raw = S.get('_ls_raw')
    if raw is None: return                                   # not answered yet
    try: data = json.loads(raw) if raw else {}
    except Exception: data = {}
    S['_ls_ok'] = True; S['_ls'] = data
    if not str(S.get('eid', '') or '').strip() and data.get('eid'): S['eid'] = S['_eid'] = data['eid']
    if not str(S.get('_lid', '') or '').strip() and data.get('lid'): S['_lid'] = data['lid']
    by = S.setdefault('made_by_team', {})
    for tid_, ent in (data.get('made') or {}).items():
        if ent.get('gw') == G0 and tid_ not in by:           # only this gameweek's: once the next gameweek starts they are dropped
            by[tid_] = [tuple(p_) for p_ in ent.get('pairs', [])]
    S.pop('_made_for', None)                                 # reload your confirmed transfers with what the browser remembered


def ls_write():
    """Save your team ID, league ID and confirmed transfers (this gameweek only) in this browser. Only written when something changed."""
    S = st.session_state
    if js_eval is None or not S.get('_ls_ok'): return
    made_all = {t: dict(gw=G0, pairs=[list(p_) for p_ in v]) for t, v in S.get('made_by_team', {}).items() if v}
    data = dict(eid=str(S.get('_eid', '') or ''), lid=str(S.get('_lid', '') or ''), made=made_all)
    blob = json.dumps(data, sort_keys=True)
    if blob == S.get('_ls_written'): return
    js_eval(js_expressions=f"localStorage.setItem('{LS_KEY}', {json.dumps(blob)})", key=f'ls_set_{abs(hash(blob))}')
    S['_ls_written'] = blob


ls_read()
remember('eid', 'team')
if '_lid' not in st.session_state: st.session_state['_lid'] = st.query_params.get('league', '')


def save_lid():
    v = str(st.session_state.get('lid_box', '')).strip(); st.session_state['_lid'] = v
    if v.isdigit(): st.query_params['league'] = v
    elif 'league' in st.query_params: del st.query_params['league']

with st.sidebar:
    if BOOST: st.caption(f'Form boost on: {BOOST:.0%}')
    html('<div style="font-family:Outfit;font-size:1.5rem;font-weight:800;margin:4px 0 2px 0">FPL <span style="color:#00ff87">Predictor</span></div>'
         f'<div class="note">Gameweek {G0} deadline in <b style="color:#00ff87">{countdown}</b></div>')
    st.write('')
    team = None
    eid = st.text_input('Your FPL team ID', placeholder='ID or paste your FPL Points link', key='eid',
                        help='On the FPL website open Points: the number after /entry/ in the address is your ID. You can paste the whole link here.',
                        on_change=remember_save, args=('eid', 'team'))
    if eid.strip().isdigit():
        try: team = load_team(int(eid), boot)
        except Exception as e:
            if FPL_UPDATING:             # FPL takes its own site offline for about 30 minutes after each deadline
                st.warning(f'FPL is updating after the GW{LIVE_GW} deadline (it usually takes about 30 minutes). Your team will load as soon as FPL is back.')
            elif isinstance(e, requests.HTTPError): st.error('No team found with that ID.')
            else: st.error(f'Could not load the team ({e}).')
    elif eid.strip(): st.error('That does not look like a team ID. Paste your FPL Points link or type the number.')
    if team is None:
        # FPL has no public search by team name, but a league's table can be searched: find yourself in any league you are in
        with st.expander("Don't know your ID? Find your team in a league"):
            st.caption('Open Leagues on the FPL website, pick any league you are in: the number after /leagues/ in the address is its ID.')
            f_lid = st.text_input('League ID', value=str(st.session_state.get('_lid', '') or ''), key='find_lid', placeholder='e.g. 314')
            f_q = st.text_input('Your team or manager name', key='find_q', placeholder='e.g. Solio Moose')
            if f_lid.strip().isdigit() and len(f_q.strip()) >= 2:
                try:
                    lname, ents = league_entries(f_lid.strip())
                    q_ = f_q.strip().lower(); hits_ = [e_ for e_ in ents if q_ in e_[1].lower() or q_ in e_[2].lower()][:8]
                    st.caption(f'{lname} · {len(ents)} teams searched' + (' (the first 1,000)' if len(ents) >= 1000 else ''))
                    for e_, tn_, mn_ in hits_:
                        st.button(f'{tn_} · {mn_}', key=f'find_{e_}', use_container_width=True, on_click=set_team_id, args=(e_,))
                    if not hits_: st.caption('No team with that name in this league.')
                except Exception: st.error('Could not load that league. Check the ID (classic leagues only).')
    if team is not None:
        # ---- transfers already made in the FPL app for the coming gameweek (FPL keeps them private until the deadline, so you tell us)
        tid = str(st.session_state.get('eid', 'manual'))
        if st.session_state.get('_made_for') != tid:                     # a different team: load ITS confirmed transfers (never another team's)
            st.session_state['_made_for'] = tid
            st.session_state['made'] = load_made(tid)
            if st.session_state['made']: st.session_state.setdefault('made_by_team', {})[tid] = list(st.session_state['made'])
            if st.session_state['made']: st.query_params['made'] = f'{tid}~' + ','.join(f'{o}-{i}' for o, i in st.session_state['made'])
            elif 'made' in st.query_params: del st.query_params['made']
        made = st.session_state.setdefault('made', [])
        team, made_ok = apply_made(team, made)
        if len(made_ok) < len(made): st.session_state['made'] = made = made_ok
        with st.expander(f'Transfers already made for GW{G0}' + (f' ({len(made)})' if made else ''), expanded=bool(made)):
            st.caption('Made a transfer in the FPL app? FPL keeps it private until the deadline, so add it here and every tab will use your new team.')
            for o_, i_ in made:      # confirmed transfers are locked, like in FPL: no undo
                st.markdown(f'<div class="note">🔒 <b style="color:#e90052">{ui.esc(P.name[o_])}</b> → <b style="color:#00ff87">{ui.esc(P.name[i_])}</b></div>', unsafe_allow_html=True)
            if team.get('made_hits'):
                st.markdown(f'<div class="note" style="color:#e90052"><b>−{fe.HIT * team["made_hits"]} hit</b> taken off your GW{G0} score.</div>', unsafe_allow_html=True)
            if team['free_transfers'] == 0:
                st.markdown('<div class="note" style="color:#e90052">No free transfers left: another one costs −4.</div>', unsafe_allow_html=True)
            if st.button('➕ Add transfers I\'ve made', use_container_width=True, key='md_open'):
                for k in ('md_out', 'md_in', 'md_n', 'md_basket', 'md_review'): st.session_state.pop(k, None)
                made_dialog(team, made, G0)
        with st.expander('Bank and free transfers'):
            sig = f'{tid}_{"_".join(f"{o}-{i}" for o, i in made)}'      # a new key whenever your transfers change, so the boxes refresh
            team['bank'] = int(round(st.number_input('Bank (£m)', min(0.0, team['bank'] / 10), 100.0, team['bank'] / 10, 0.1, key=f'bank_{sig}') * 10))
            if team['bank'] < 0: st.error(f'Your confirmed transfers leave you £{-team["bank"] / 10:.1f}m over budget, which FPL would not allow.')
            team['free_transfers'] = st.number_input('Free transfers', 0, 5, int(team['free_transfers']), key=f'ft_{sig}')
            if team.get('made_hits'): st.caption(f'Your transfers include {team["made_hits"]} hit (−{fe.HIT * team["made_hits"]} pts in GW{G0}).')
            st.caption('Worked out from your public transfer history. Check against the FPL site.')
    html(ui.SB_CREDIT)

HIT0 = fe.HIT * team.get('made_hits', 0) if team is not None else 0     # points hit from your confirmed GW11 transfers
ls_write()

# ------------------------------------------------------------------ header and navigation
def _ago(t):
    m = (datetime.now(timezone.utc) - datetime.fromisoformat(t)).total_seconds() / 60
    return f'{m:.0f} min ago' if m < 90 else (f'{m / 60:.0f} h ago' if m < 48 * 60 else f'{m / 1440:.0f} days ago')
if NEWS_AT:                       # team news and prices come live from FPL; the model itself is retrained after each gameweek
    updated_pill = (f'<span class="pill" title="Injury news, chance of playing and prices are fetched from FPL every 15 minutes and applied to every prediction. '
                    f'The model was last updated {_ago(meta["updated_at"]) if meta.get("updated_at") else "recently"}.">'
                    f'Team news &amp; prices live · checked {NEWS_AT.astimezone(UK):%H:%M}</span>')
else:
    updated_pill = (f'<span class="pill">Updated {_ago(meta["updated_at"])}' + (' · late team news' if meta.get('update_type', '').startswith('late') else '') + '</span>') if meta.get('updated_at') else ''
if LIVE_WINDOW: updated_pill += f'<span class="pill" style="border-color:#00ff87;color:#00ff87">● GW{LIVE_GW} {"live now" if LIVE_ON else "in progress"} · predictions update after its results</span>'
sub = (f"{team['manager']} · {team['points']} pts · rank {team['rank']:,}" if team and team.get('rank') else 'Projected points, best XI, captain and transfers for the coming gameweek')
html(f'<div class="hero"><div class="hero-title">{ui.esc(team["name"]) if team else "Your FPL team,"} <span>{"GW" + str(G0) if team else "projected."}</span></div>'
     f'<div class="hero-sub">{ui.esc(sub)}</div>'
     f'<span class="pill g">GW{G0} · {deadline.astimezone(UK):%a %d %b %H:%M} UK time</span><span class="pill">Deadline in {countdown}</span>{updated_pill}</div>')
PAGES = (['Live'] if LIVE_GW else []) + ['My team', 'Transfers', 'Planner', 'Captain', 'Chips', 'League', 'Compare', 'Players', 'Teams', 'Review', 'How it works']
page = st.segmented_control('Page', PAGES, default='Live' if LIVE_WINDOW else 'My team', label_visibility='collapsed', key='nav') or ('Live' if LIVE_WINDOW else 'My team')


def done():
    """End the page: the credit footer, then stop. Used instead of st.stop() so the footer shows on every page."""
    html(ui.CREDIT); st.stop()


def need_team():
    html('<div class="card" style="padding:22px"><div class="v" style="font-size:1.2rem">Add your team to start</div>'
         '<div class="l" style="font-size:.9rem;margin-top:6px">Enter your FPL team ID in the sidebar. '
         'Meanwhile, here are the top projected players for Gameweek %d.</div></div>' % G0)
    ids = P[P.status != 'u'].nlargest(10, f'xp_{G0}').index
    html(ui.player_table(P, ids, ALL_GWS, f'xp_{G0}'))


def move_card(i, kind, gws):
    r = P.loc[i]; pts = sum(r[f'xp_{g}'] for g in gws)
    return (f'<div class="mv {kind}"><img src="{ui.shirt(r.team_code, r.pos == 1)}"><div><div class="bigname">{ui.esc(r["name"])}</div>'
            f'<div class="m">{r.team} · {POS[r.pos]} · £{r.price:.1f}m · {pts:.1f} pts over {len(gws)} GW{"s" if len(gws) > 1 else ""}</div></div></div>')


@st.cache_data(show_spinner=False)
def run_plan(squad_json, bank, ft, allow_hits, chip, roll, banned, keep, gws, boost=0.0):
    from io import StringIO
    return fe.optimise_plan(P, list(gws), pd.read_json(StringIO(squad_json)), bank, ft, allow_hits=allow_hits, chip=chip, roll_value=roll,
                            banned=list(banned), forced=list(keep))


def delta_bar(pb, pa, g, hit=0):
    pa = pa - hit; d = pa - pb; col = ui.GREEN if d > 0.05 else (ui.PINK if d < -0.05 else '#ffffff')
    return (f'<div class="card" style="display:flex;align-items:center;gap:18px;padding:10px 16px;margin-bottom:10px"><div><div class="l">GW{ALL_GWS[0] - 1} team, GW{g} points</div><div class="v">{pb:.1f}</div></div>'
            f'<div style="font-size:1.6rem;color:#00ff87">→</div><div><div class="l">with the changes{f" (after −{hit} hit)" if hit else ""}</div><div class="v">{pa:.1f}</div></div>'
            f'<div style="margin-left:auto;text-align:right"><div class="l">difference</div><div class="v" style="color:{col}">{d:+.1f} pts</div></div></div>')


def compare_pitches(before, after, g, outs=(), ins=(), t_before='Now', t_after='After the changes', hit=0):
    """Two pitches side by side, now and after the changes, with the points difference between them."""
    xp = P[f'xp_{g}']
    lb, la = lineup(tuple(sorted(before)), g, BOOST), lineup(tuple(sorted(after)), g, BOOST)
    pb = xp[lb[0]].sum() + xp[lb[1]]; pa = xp[la[0]].sum() + xp[la[1]]; d = pa - pb - hit
    html(delta_bar(pb, pa, g, hit))
    c1, c3 = st.columns(2)
    with c1:
        html(ui.pitch(P, lb[0], lb[3], lb[1], lb[2], {i: xp[i] * (2 if i == lb[1] else 1) for i in before}, {i: P.loc[i, f'opp_{g}'] for i in before}, g,
                      compact=True, mark_out=set(outs), title=f'{t_before} <small>{pb:.1f} pts</small>'))
    with c3:
        html(ui.pitch(P, la[0], la[3], la[1], la[2], {i: xp[i] * (2 if i == la[1] else 1) for i in after}, {i: P.loc[i, f'opp_{g}'] for i in after}, g,
                      compact=True, mark_in=set(ins), title=f'{t_after} <small>{pa:.1f} pts</small>'))
    return d


@st.dialog('Change a player', width='large')
def replace_dialog(o, g, squad_now, money, rest, moves, lu):
    """Pop-up opened by clicking a name on the Planner pitch: transfer him (any player in his position), or change the lineup."""
    xi, cap, vice, bench = lu
    colx = lambda i: sum(P.loc[i, f'xp_{x}'] for x in rest)
    span = f'GW{rest[0]}' + (f'-{rest[-1]}' if len(rest) > 1 else '')
    html(f'<div class="move" style="grid-template-columns:1fr">{move_card(o, "out", rest)}</div>')
    t1, t2 = st.tabs(['🔁 Transfer him', '🧢 Lineup and captain'])
    with t1:
        html(f'<div class="note">Selling in GW{g} · money available <b style="color:#00ff87">£{money / 10:.1f}m</b> · points shown over {span}</div>')
        cand = P[(P.pos == P.pos[o]) & (P.status != 'u') & ~P.index.isin(squad_now)].copy(); cand['v'] = [colx(i) for i in cand.index]
        f1, f2, f3 = st.columns([2.2, 1.6, 1.4])
        teams = f1.multiselect('Teams', sorted(cand.team.unique()), placeholder='All teams', key=f'dlg_tm_{o}_{g}')
        mx = f2.slider('Max price (£m)', 3.5, 16.0, 16.0, 0.5, key=f'dlg_mx_{o}_{g}')
        sort = f3.selectbox('Sort by', ['Projected points', 'Points per £m', 'Price', 'Ownership', 'Form'], key=f'dlg_sort_{o}_{g}')
        q = st.text_input('Search', placeholder='Search a name', key=f'dlg_q_{o}_{g}', label_visibility='collapsed')
        c_a, c_b = st.columns(2)
        fit = c_a.toggle('Hide injured and doubtful', True, key=f'dlg_fit_{o}_{g}')
        aff = c_b.toggle('Only players I can afford', False, key=f'dlg_aff_{o}_{g}')
        if teams: cand = cand[cand.team.isin(teams)]
        cand = cand[cand.price <= mx + 1e-9]
        if fit: cand = cand[cand.chance.isna() | (cand.chance >= 75)]
        if aff: cand = cand[(cand.price * 10).round() <= money]
        if q: cand = cand[cand.name.str.contains(q, case=False)]
        key_ = {'Projected points': cand.v, 'Points per £m': cand.v / cand.price, 'Price': cand.price, 'Ownership': cand.selected, 'Form': cand.form}[sort]
        cand = cand.loc[key_.sort_values(ascending=False).index]
        nshow = st.session_state.get(f'dlg_n_{o}_{g}', 15)
        html(f'<div class="note" style="margin:4px 0">{len(cand)} {fe.POS_NAME[P.pos[o]]}s · showing {min(nshow, len(cand))}</div>')
        base = colx(o)
        for i, r_ in cand.head(nshow).iterrows():
            a, b = st.columns([6, 1.3])
            gain = r_.v - base; short = round(r_.price * 10) - money
            flag = '' if pd.isna(r_.chance) or r_.chance >= 100 else f' · <b style="color:#ffab1b">{int(r_.chance)}% fit</b>'
            budget = f' · <b style="color:#e90052">over budget by £{short / 10:.1f}m</b>' if short > 0 else ''
            a.markdown(f'<div class="mv {"out" if short > 0 else "in"}" style="padding:6px 10px"><img src="{ui.shirt(r_.team_code, r_.pos == 1)}"><div><div class="bigname">{ui.esc(r_["name"])} '
                       f'<span class="note">{r_.team} · {ui.esc(r_[f"opp_{rest[0]}"])}</span></div>'
                       f'<div class="m">£{r_.price:.1f}m · {r_.v:.1f} pts · <b style="color:{ui.GREEN if gain > 0 else ui.PINK}">{gain:+.1f}</b> vs {ui.esc(P.name[o])}'
                       f' · owned {r_.selected:.0f}% · form {r_.form:.1f}{flag}{budget}</div></div></div>', unsafe_allow_html=True)
            if b.button('Transfer', key=f'dlg_pick_{o}_{g}_{i}', type='primary' if short <= 0 else 'secondary', use_container_width=True,
                        help='Over budget: you can still plan it, and the plan will show how much you are short' if short > 0 else None):
                st.session_state['pl_moves'] = moves + [dict(gw=g, out=int(o), inn=int(i))]
                st.rerun()
        if cand.empty: st.info('No players match these filters.')
        if len(cand) > nshow and st.button('Show 15 more', key=f'dlg_more_{o}_{g}', use_container_width=True):
            st.session_state[f'dlg_n_{o}_{g}'] = nshow + 15; st.rerun(scope='fragment')
    with t2:
        def save(xi_, cap_, vice_):
            bench_ = [i for i in squad_now if i not in xi_]
            st.session_state.setdefault('pl_lineup', {})[g] = dict(bench=bench_, cap=cap_, vice=vice_ if vice_ != cap_ else next(i for i in xi_ if i != cap_))
            st.rerun()
        xpg = P[f'xp_{g}']
        legal = lambda xi_: all(fe.XI_MIN[p] <= sum(P.pos[i] == p for i in xi_) <= fe.XI_MAX[p] for p in (1, 2, 3, 4))
        if o in xi:
            html(f'<div class="note"><b style="color:#fff">{ui.esc(P.name[o])}</b> starts in GW{g}. Bring a bench player on instead:</div>')
            opts = [b for b in bench if legal([i for i in xi if i != o] + [b])]
            for b in opts:
                x1, x2 = st.columns([6, 1.3])
                x1.markdown(f'<div class="mv in" style="padding:6px 10px"><img src="{ui.shirt(P.team_code[b], P.pos[b] == 1)}"><div><div class="bigname">{ui.esc(P.name[b])}</div>'
                            f'<div class="m">{fe.POS_NAME[P.pos[b]]} · {xpg[b]:.1f} pts in GW{g} · <b style="color:{ui.GREEN if xpg[b] > xpg[o] else ui.PINK}">{xpg[b] - xpg[o]:+.1f}</b></div></div></div>', unsafe_allow_html=True)
                if x2.button('Bring on', key=f'dlg_sub_{o}_{b}_{g}', use_container_width=True):
                    save([i for i in xi if i != o] + [b], b if cap == o else cap, b if vice == o else vice)
            if not opts: st.info('No bench player can replace him without breaking the formation rules (1 GK, 3+ DEF, 2+ MID, 1+ FWD).')
            k1, k2 = st.columns(2)
            if o != cap and k1.button('👑 Make captain', key=f'dlg_cap_{o}_{g}', type='primary', use_container_width=True):
                save(xi, o, cap if vice == o else vice)
            if o != vice and o != cap and k2.button('Make vice-captain', key=f'dlg_vice_{o}_{g}', use_container_width=True):
                save(xi, cap, o)
            if o == cap: html('<div class="note">He is your captain (points doubled).</div>')
        else:
            html(f'<div class="note"><b style="color:#fff">{ui.esc(P.name[o])}</b> is on the bench in GW{g}. Put him in the team instead of:</div>')
            opts = [s_ for s_ in xi if legal([i for i in xi if i != s_] + [o])]
            for s_ in sorted(opts, key=lambda i: xpg[i]):
                x1, x2 = st.columns([6, 1.3])
                x1.markdown(f'<div class="mv out" style="padding:6px 10px"><img src="{ui.shirt(P.team_code[s_], P.pos[s_] == 1)}"><div><div class="bigname">{ui.esc(P.name[s_])}</div>'
                            f'<div class="m">{fe.POS_NAME[P.pos[s_]]} · {xpg[s_]:.1f} pts in GW{g} · swapping gives <b style="color:{ui.GREEN if xpg[o] > xpg[s_] else ui.PINK}">{xpg[o] - xpg[s_]:+.1f}</b></div></div></div>', unsafe_allow_html=True)
                if x2.button('Swap', key=f'dlg_sw_{o}_{s_}_{g}', use_container_width=True):
                    save([i for i in xi if i != s_] + [o], o if cap == s_ else cap, o if vice == s_ else vice)
            if not opts: st.info('He cannot come on without breaking the formation rules.')
    if st.button('Close', key=f'dlg_close_{o}_{g}'):
        st.rerun()


def pairs(res):
    inn = list(res['inn']); out = []
    for o in sorted(res['out'], key=lambda x: P.pos[x]):
        j = next(i for i in inn if P.pos[i] == P.pos[o]); inn.remove(j); out.append((o, j))
    return out


# ================================================================== LIVE: the gameweek being played (or just finished), from FPL's live data
if page == 'Live':
    import fpl_live as fl
    g = LIVE_GW
    team_of = {e['id']: e['team'] for e in boot['elements']}; pos_of = {e['id']: e['element_type'] for e in boot['elements']}
    sn = {t['id']: t['short_name'] for t in boot['teams']}

    @st.cache_data(ttl=300, show_spinner=False)
    def _live_picks(eid_, g_):
        return fe.get(f'entry/{int(eid_)}/event/{g_}/picks/')      # a failure is not cached, so the picks load as soon as FPL has them

    def live_picks(eid_, g_):
        try: return _live_picks(eid_, g_)
        except Exception: return None

    @st.cache_data(ttl=300, show_spinner=False)
    def live_entry(eid_):
        try: return fe.get(f'entry/{int(eid_)}/')
        except Exception: return None

    @st.cache_data(ttl=3600, show_spinner=False)
    def predicted_for(g_):
        """What the app predicted for each player that week: the pre-deadline snapshot, or the back-test for early weeks."""
        f_ = ROOT / 'data' / 'snapshots' / f'gw{g_}.parquet'
        if f_.exists():
            s_ = pd.read_parquet(f_); return s_['xp'].to_dict() if 'xp' in s_ else {}
        try:
            bt = pd.read_csv(ROOT / 'models' / 'test_predictions.csv'); bt = bt[(bt.season == '2026-27') & (bt.gw == g_)]
            return bt.groupby('element').blend.sum().to_dict()
        except Exception: return {}

    @st.fragment(run_every=60 if (LIVE_ON or FPL_UPDATING) else None)
    def live_view():
        try: live, fx = live_data(g)
        except Exception:
            html(f'<div class="card" style="padding:18px"><div class="v" style="font-size:1.1rem">' +
                 (f'FPL is updating after the GW{g} deadline' if FPL_UPDATING else 'FPL\'s live data is not reachable right now') +
                 '</div><div class="l" style="font-size:.9rem;margin-top:6px">' + ('It usually takes about 30 minutes. ' if FPL_UPDATING else '') +
                 'This page tries again every minute.</div></div>')
            return
        pl = fl.players(live, fx, team_of)
        states = [fl.fixture_state(f) for f in fx]
        n_live, n_done, n_todo = states.count('live'), states.count('finished'), states.count('not started')
        html(f'<div class="note" style="margin:-4px 0 10px 0"><b style="color:#fff">GW{g} {"live" if n_live else ("final" if not n_todo else "in progress")}</b> · '
             f'{n_live} match{"es" if n_live != 1 else ""} in play · {n_done} finished · {n_todo} still to play'
             + (' · updates every minute' if n_live else '') + f' · checked {NOW.astimezone(UK) if os.environ.get("FPL_NOW") else datetime.now(UK):%H:%M} UK time</div>')
        pred = predicted_for(g)
        eid_ = str(st.session_state.get('eid', '') or '').strip()
        pk = live_picks(eid_, g) if eid_.isdigit() else None
        if pk:
            hits = (pk.get('entry_history') or {}).get('event_transfers_cost', 0) or 0
            t = fl.team_score(pk['picks'], pk.get('active_chip'), pl, pos_of, hits=hits)
            rows = {r['id']: r for r in t['rows']}
            orig = {p['element']: p for p in pk['picks']}
            cm = 3 if pk.get('active_chip') == '3xc' else 2
            pred_total = sum(pred.get(i, 0) * (cm if orig[i]['is_captain'] else (1 if orig[i]['position'] <= 11 or pk.get('active_chip') == 'bboost' else 0)) for i in orig) - hits
            en = live_entry(eid_) or {}
            prov = sum(pl.get(r['id'], {}).get('bonus_prov', 0) * r['mult'] for r in t['rows'])
            chip_ = {'bboost': 'Bench Boost', '3xc': 'Triple Captain', 'freehit': 'Free Hit', 'wildcard': 'Wildcard', 'manager': 'Assistant Manager'}.get(pk.get('active_chip'))
            sec(f'Your GW{g} score', 'live, after hits' + (f' · {chip_}' if chip_ else ''))
            html(ui.cards([(f'{t["total"]:.0f}', 'points so far' + (f' (incl. {prov} provisional bonus)' if prov else '') + (f' · after −{hits} hit' if hits else ''), 'g'),
                           (f'{t["playing"]} · {t["left"]}', 'players playing now · still to play', 'c'),
                           (f'{pred_total:.0f}', 'what the model predicted for your team', ''),
                           (f'{en.get("summary_event_rank"):,}' if en.get('summary_event_rank') else '-', 'your GW rank on FPL (FPL updates it every so often)', '')]))
            # the pitch: the players who count now (after any automatic subs), captain multiplied
            counting = [r['id'] for r in t['rows'] if r['mult'] > 0]; rest_ = [r['id'] for r in t['rows'] if r['mult'] == 0]
            def stat_txt(i):
                s_ = pl.get(i, {}); st_ = s_.get('status', 'no match')
                if st_ == 'not started':
                    ko = [f_['kickoff_time'] for f_ in fx if f_['id'] in s_.get('fixtures', [])]
                    return (datetime.fromisoformat(ko[0].replace('Z', '+00:00')).astimezone(UK).strftime('%a %H:%M') if ko and ko[0] else 'to play')
                if st_ == 'live': return f"playing · {s_.get('mins', 0)}'"
                if st_ == 'finished': return f"done · {s_.get('mins', 0)}'"
                return 'no match'
            val = {i: rows[i]['pts'] * max(rows[i]['mult'], 1) for i in rows}
            P_live = P.assign(chance=100)
            html(ui.pitch(P_live, [i for i in counting if i in P.index], [i for i in rest_ if i in P.index], t['cap_now'], next((r['id'] for r in t['rows'] if r['vice']), None),
                          val, {i: stat_txt(i) for i in rows}, mark_in=[b_ for _, b_ in t['subs']], mark_out=[s_ for s_, _ in t['subs']], whole=True))
            notes = []
            for s_, b_ in t['subs']: notes.append(f'<b style="color:#fff">{ui.esc(P.name[b_])}</b> comes on for <b style="color:#fff">{ui.esc(P.name[s_])}</b> (did not play)')
            if t['cap_now'] != t['cap'] and t['cap'] in P.index: notes.append(f'<b style="color:#fff">{ui.esc(P.name[t["cap_now"]])}</b> takes the armband: {ui.esc(P.name[t["cap"]])} did not play')
            html('<div class="note" style="margin-top:4px">' + (' · '.join(notes) + '. ' if notes else '') +
                 'Green: came on as an automatic sub · the captain\'s points are multiplied. Bonus is provisional (from the live BPS) until FPL adds it, about an hour after each match.</div>')
            # every player: how he got his points, against what the model predicted
            body = ''
            for r in sorted(t['rows'], key=lambda r: (r['mult'] == 0, -r['pts'] * max(r['mult'], 1))):
                i = r['id']
                if i not in P.index: continue
                s_ = pl.get(i, {}); how = ' · '.join(f'{n_} {v:+d}' if v < 0 else f'{n_} {v}' for n_, v in s_.get('lines', [])) or '-'
                role = 'Captain' if r['cap_now'] else ('Vice' if r['vice'] else ('Bench' if r['mult'] == 0 else 'Starter'))
                body += (f'<tr style="{"opacity:.55" if r["mult"] == 0 else ""}"><td><div class="who"><img src="{ui.shirt(P.team_code[i], P.pos[i] == 1)}"><div><b>{ui.esc(P.name[i])}</b>'
                         f'<div class="t">{P.team[i]} · {role}{" ×" + str(r["mult"]) if r["mult"] > 1 else ""}</div></div></div></td><td>{stat_txt(i)}</td>'
                         f'<td style="font-size:.8rem">{ui.esc(how)}</td><td>{pred.get(i, 0):.1f}</td><td><b style="color:{ui.GREEN}">{r["pts"] * max(r["mult"], 1):.0f}</b></td></tr>')
            html('<table class="fpl"><tr><th>Player</th><th>Status</th><th>How he scored</th><th>Predicted</th><th>Points</th></tr>' + body + '</table>')
        elif eid_.isdigit():
            html(f'<div class="note">' + (f'FPL is updating after the GW{g} deadline (it usually takes about 30 minutes). Your live score shows as soon as FPL is back.'
                                          if FPL_UPDATING else f'Your GW{g} picks are not available from FPL yet.') + '</div>')
        else:
            html('<div class="note">Add your team ID in the sidebar to see your live score.</div>')

        # ---- the matches
        sec(f'GW{g} matches', 'live scores · bonus points leaders (provisional until FPL adds the bonus)')
        el_name = {e['id']: e['web_name'] for e in boot['elements']}
        cards_ = ''
        for f_, st_ in sorted(zip(fx, states), key=lambda x: ({'live': 0, 'finished': 2, 'not started': 1}[x[1]], x[0]['kickoff_time'] or '')):
            ko = datetime.fromisoformat(f_['kickoff_time'].replace('Z', '+00:00')).astimezone(UK) if f_.get('kickoff_time') else None
            score = f"{f_['team_h_score']} - {f_['team_a_score']}" if f_.get('started') else (f'{ko:%a %H:%M}' if ko else 'TBC')
            tag = f"<span style='color:#00ff87'>● {f_.get('minutes', 0)}'</span>" if st_ == 'live' else ('FT' if st_ == 'finished' else 'to play')
            bps = fl._stat(f_, 'bps', 'h') | fl._stat(f_, 'bps', 'a')
            off = fl._stat(f_, 'bonus', 'h') | fl._stat(f_, 'bonus', 'a'); bon = off or fl.bonus_from_bps(bps)
            lead = ' · '.join(f'{ui.esc(el_name.get(p_, "?"))} {b_}' for p_, b_ in sorted(bon.items(), key=lambda x: -x[1])) if f_.get('started') else ''
            cards_ += (f'<div class="card" style="padding:10px 12px"><div class="l" style="margin:0">{tag}</div><div class="v" style="font-size:1.15rem;margin:2px 0">'
                       f'{sn.get(f_["team_h"], "?")} <span style="color:#00ff87">{score}</span> {sn.get(f_["team_a"], "?")}</div>'
                       + (f'<div class="l" style="font-size:.75rem">Bonus{"" if off else " (provisional)"}: {lead}</div>' if lead else '') + '</div>')
        html(f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:8px">{cards_}</div>')

        # ---- your mini-league, live
        lid_ = str(st.session_state.get('_lid', '') or '').strip()
        if lid_.isdigit():
            sec('Your mini-league, live', 'the top 20: their live GW score and where they would be now')
            try:
                lg = live_league(lid_, g)
                rows_l = []
                for m in lg['managers']:
                    pk_ = m['picks']
                    if not pk_: continue
                    hits_ = (pk_.get('entry_history') or {}).get('event_transfers_cost', 0) or 0
                    t_ = fl.team_score(pk_['picks'], pk_.get('active_chip'), pl, pos_of, hits=hits_)
                    cap_ = t_['cap_now']
                    rows_l.append(dict(entry=m['entry'], team=m['entry_name'], manager=m['player_name'], rank=m['rank'], before=m['total'] - m['event_total'],
                                       live=t_['total'], left=t_['left'], cap=P.name.get(cap_, '?') if cap_ else '-', chip=pk_.get('active_chip')))
                Lr = pd.DataFrame(rows_l)
                if len(Lr):
                    Lr['now'] = Lr.before + Lr.live; Lr = Lr.sort_values('now', ascending=False).reset_index(drop=True); Lr['live_rank'] = Lr.index + 1
                    me_ = int(eid_) if eid_.isdigit() else None
                    body = ''.join(
                        f'<tr style="{"background:rgba(0,255,135,.08)" if r.entry == me_ else ""}"><td><b>{r.live_rank}</b> '
                        f'<span style="color:{ui.GREEN if r.live_rank < r["rank"] else (ui.PINK if r.live_rank > r["rank"] else "#b9a7c2")}">'
                        f'{"▲" if r.live_rank < r["rank"] else ("▼" if r.live_rank > r["rank"] else "·")}</span></td>'
                        f'<td><b>{ui.esc(r.team)}</b><div class="t" style="color:#b9a7c2;font-size:.75rem">{ui.esc(r.manager)}</div></td>'
                        f'<td><b style="color:{ui.GREEN}">{r.live:.0f}</b></td><td>{r.now:.0f}</td><td>{ui.esc(r.cap)}</td><td>{r.left}</td>'
                        f'<td>{ {"bboost": "BB", "3xc": "TC", "freehit": "FH", "wildcard": "WC"}.get(r.chip, "") }</td></tr>' for _, r in Lr.iterrows())
                    html('<table class="fpl"><tr><th>Live rank</th><th>Team</th><th>GW live</th><th>Total</th><th>Captain</th><th>Still to play</th><th>Chip</th></tr>' + body + '</table>')
                    html(f'<div class="note" style="margin-top:4px">{lg["name"]}. ▲▼ compare with the league table before this gameweek.</div>')
            except Exception as e_:
                st.caption(f'Could not load the league live ({e_}).')
        else:
            html('<div class="note" style="margin-top:6px">Add your mini-league ID on the League tab to follow it live here.</div>')

    live_view()


# ================================================================== MY TEAM
elif page == 'My team':
    if team is None: need_team(); done()
    ids = tuple(int(i) for i in team['squad'].id)
    gw_lab = st.segmented_control('Gameweek', [f'GW{g}' for g in ALL_GWS[:6]], default=f'GW{G0}', label_visibility='collapsed') or f'GW{G0}'
    gw = int(gw_lab[2:]); xp = P[f'xp_{gw}']
    xi, cap, vice, bench = lineup(ids, gw, BOOST)
    total = xp[xi].sum() + xp[cap] - (HIT0 if gw == G0 else 0); bpts = xp[bench].sum()
    value = (team['squad'].now.sum() + team['bank']) / 10
    html(ui.cards([(f'{total:.0f}', f'projected GW{gw} points' + (f' (after −{HIT0} hit)' if HIT0 and gw == G0 else ''), 'g'), (ui.esc(P.name[cap]), f'captain · {2 * xp[cap]:.1f} pts', 'c'),
                   (f'{bpts:.1f}', 'bench points', ''), (f'£{value:.1f}m', 'team value', ''),
                   (f'£{team["bank"] / 10:.1f}m', 'in the bank', ''), (team['free_transfers'], 'free transfers', '')]))
    c1, c2 = st.columns([3, 2], gap='large')
    with c1:
        sec(f'Gameweek {gw} line-up', 'best formation from your 15 · captain ×2')
        val = {i: xp[i] * (2 if i == cap else 1) for i in ids}; fix = {i: P.loc[i, f'opp_{gw}'] for i in ids}
        html(ui.pitch(P, xi, bench, cap, vice, val, fix, gw))
        flagged = [i for i in ids if pd.notna(P.chance[i]) and P.chance[i] < 100]
        if flagged: html('<div class="note" style="margin-top:8px">⚠ ' + ' · '.join(f'<b>{ui.esc(P.name[i])}</b> {int(P.chance[i])}%: {ui.esc(P.news[i])}' for i in flagged) + '</div>')
    with c2:
        sec('Next 6 gameweeks', 'projected team points, no transfers')
        gs = ALL_GWS[:6]; tots = []
        for g in gs:
            x2, c2_, _, _ = lineup(ids, g, BOOST); tots.append(P.loc[x2, f'xp_{g}'].sum() + P.loc[c2_, f'xp_{g}'] - (HIT0 if g == G0 else 0))
        fig = go.Figure(go.Bar(x=[f'GW{g}' for g in gs], y=tots, marker=dict(color=[ui.GREEN if g == gw else '#7a3f86' for g in gs]),
                               text=[f'{t:.0f}' for t in tots], textposition='outside', hovertemplate='%{x}: %{y:.1f} pts<extra></extra>'))
        html_fig = ui.plot_style(fig, 250); html_fig.update_yaxes(range=[0, max(tots) * 1.2], showticklabels=False)
        st.plotly_chart(html_fig, use_container_width=True, config={'displayModeBar': False})
        sec('Your fixtures', 'official difficulty: green easy, red hard')
        order = sorted(ids, key=lambda i: (P.pos[i], -xp[i]))
        html(ui.ticker([(ui.esc(P.name[i]), [(P.loc[i, f'opp_{g}'].split(',')[0].replace(' (H)', ' H').replace(' (A)', ' A') + ('+' if ',' in P.loc[i, f'opp_{g}'] else ''), P.loc[i, f'fdr_{g}']) for g in gs]) for i in order], gs))
    sec('Squad projections')
    hz = st.segmented_control('Horizon', ['Next GW', 'Next 3'], default='Next GW', key='hz_team', label_visibility='collapsed') or 'Next GW'
    col = f'h{HORIZONS[hz]}'
    order = sorted(ids, key=lambda i: -P.loc[i, col])
    html(ui.player_table(P, order, ALL_GWS, col, extra=('Role', lambda i: 'Captain' if i == cap else ('Vice' if i == vice else ('Bench' if i in bench else 'Starter')))))
    tc = xp[cap]
    html(f'<div class="note" style="margin-top:8px">Chips left this half: <b>{", ".join(team["chips_left"]) or "none"}</b>. In GW{gw} a Triple Captain would add about {tc:.1f} points '
         f'and a Bench Boost about {bpts:.1f}. Both are usually worth more in a double gameweek.</div>')

# ================================================================== TRANSFERS
elif page == 'Transfers':
    if team is None: need_team(); done()
    ids = [int(i) for i in team['squad'].id]
    c1, c2, c3 = st.columns([1.3, 1.6, 1.1])
    plan = c1.segmented_control('Plan over', ['1 GW', '2 GWs', '3 GWs'], default='3 GWs', key='plan') or '3 GWs'
    chip = c2.segmented_control('Transfer chip', ['None'] + [x for x in team['chips_left'] if x in ('Wildcard', 'Free Hit')], default='None', key='chip') or 'None'
    c3.write(''); allow_hits = c3.toggle('Allow −4 hits', False, key='hits', help='Off: only your free transfers are used. On: up to one extra transfer a week if it gains more than 4 points.')
    html(f'<div class="note" style="margin:-4px 0 8px 0">You have <b style="color:#00ff87">{team["free_transfers"]} free transfer{"s" if team["free_transfers"] != 1 else ""}</b> this week'
         + (f' ({len(team["pending"])} already made)' if team.get('pending') else '') + '. Unused ones roll over (up to 5). '
         'Bench Boost, Triple Captain and the best week for each chip are on the <b>Chips</b> tab.</div>')
    with st.expander('More options'):
        a_, b_, c_ = st.columns(3)
        banned = a_.multiselect('Never buy', [i for i in P.index if i not in ids], format_func=LAB.get)
        keep = b_.multiselect('Always keep', ids, format_func=LAB.get)
        roll = c_.slider('A saved free transfer is worth (pts)', 0.0, 3.0, 1.0, 0.5, help='Stops the plan using a transfer for a tiny gain.')
    gws = ALL_GWS[:1] if chip == 'Free Hit' else ALL_GWS[:int(plan[0])]
    sj = team['squad'][['id', 'sell', 'now']].to_json()
    with st.spinner('Planning your transfers...'):
        try:
            if chip == 'Free Hit':
                fh = run_opt(sj, team['bank'], team['free_transfers'], 15, 'Free Hit', roll, tuple(banned), tuple(keep), tuple(gws), BOOST)
                R = dict(weeks=[dict(gw=gws[0], out=fh['out'], inn=fh['inn'], hits=0, ft_before=team['free_transfers'], squad=fh['squad'], bank=fh['bank_after'])], bank_after=fh['bank_after'])
            else:
                R = run_plan(sj, team['bank'], team['free_transfers'], allow_hits, None if chip == 'None' else chip, roll, tuple(banned), tuple(keep), tuple(gws), BOOST)
        except ValueError as e:
            st.error(str(e)); done()
    pts = lambda sq_, g: (lambda lu: P.loc[lu[0], f'xp_{g}'].sum() + P.loc[lu[1], f'xp_{g}'])(lineup(tuple(sorted(sq_)), g, BOOST))
    W = R['weeks']; per_gain = [pts(w['squad'], w['gw']) - pts(ids, w['gw']) for w in W]
    hits_tot = sum(w['hits'] for w in W); net = sum(per_gain) - fe.HIT * hits_tot
    n_moves = sum(len(w['inn']) for w in W)
    colr = ui.GREEN if net > 0 else (ui.PINK if net < 0 else '#ffffff')
    html(ui.cards([(f'<span style="color:{colr}">{net:+.1f}</span>', f'points gained over {"GW" + str(gws[0]) if len(gws) == 1 else f"GW{gws[0]}-{gws[-1]}"} vs no transfers (after hits)', ''),
                   (str(len(W[0]['inn'])), f'transfer{"s" if len(W[0]["inn"]) != 1 else ""} this week', 'g'), (f'−{fe.HIT * hits_tot}' if hits_tot else '0', 'hit points', ''),
                   (f'£{W[0]["bank"] / 10:.1f}m', 'bank after this week', 'c')]))
    if chip == 'Free Hit':
        html('<div class="note">Free Hit: this team plays GW%d only. Your current squad comes back next week, and your free transfers are kept.</div>' % gws[0])
    else:
        html('<div class="note" style="margin:-2px 0 6px 0">Why the answer can change with the horizon: a transfer keeps paying off every week you keep the new player, '
             'while saving the transfer is worth about %.1f point. Over 1 week a small gain may not beat saving it (so: hold); over 3 weeks the same move adds up and wins (so: do it now).</div>' % roll)

    # ---- this week: two pitches, now vs after the moves
    w0 = W[0]; g = w0['gw']
    orig = team.get('orig_ids', ids); conf = team.get('made', [])
    sec(f'This week · GW{g}', f'left: your GW{g - 1} team · right: with ' + ('your confirmed and ' if conf else '') + 'the suggested moves')
    compare_pitches(orig, w0['squad'], g, [i for i in orig if i not in w0['squad']], [i for i in w0['squad'] if i not in orig],
                    t_before=f'Your GW{g - 1} team', t_after='With ' + ('confirmed + suggested' if conf else 'the suggested') + ' changes', hit=HIT0 + fe.HIT * w0['hits'])
    if conf:
        html('<div class="note" style="margin:2px 0 6px 0">✅ Already made in the FPL app: ' + ', '.join(f'<b>{ui.esc(P.name[o])}</b> → <b style="color:#00ff87">{ui.esc(P.name[i])}</b>' for o, i in conf)
             + '. The suggestions below come on top of these.</div>')
    if w0['inn']:
        for o, i in pairs(dict(out=w0['out'], inn=w0['inn'])):
            html(f'<div class="move">{move_card(o, "out", gws)}<div class="arrow">→</div>{move_card(i, "in", gws)}</div>')
    else:
        html('<div class="card" style="padding:14px"><div class="v g" style="font-size:1.1rem">Hold this week</div><div class="l" style="font-size:.88rem">No move gains enough now. '
             f'Saving the transfer gives you {min(team["free_transfers"] + 1, 5)} next week.</div></div>')

    # ---- the following weeks
    if len(W) > 1:
        sec('The plan for the next weeks', 'what to do each week · points vs keeping your current team')
        rows_ = ''
        for w, gain in zip(W, per_gain):
            moves = ''.join(f'<span class="mv"><img src="{ui.shirt(P.team_code[o], P.pos[o] == 1)}"><span><b>{ui.esc(P.name[o])}</b> → '
                            f'<b style="color:#00ff87">{ui.esc(P.name[i])}</b></span></span>' for o, i in pairs(dict(out=w['out'], inn=w['inn'])))
            if not moves: moves = f'<span style="color:#b9a7c2">Hold · roll the transfer</span>'
            used_ = len(w['inn']); ftb = w['ft_before']
            ft_txt = 'Wildcard: all free' if (chip == 'Wildcard' and w is W[0]) else f'{ftb} free · uses {min(used_, ftb)}' + (f' + {w["hits"]} hit' if w['hits'] else '')
            rows_ += (f'<div class="week"><div><div class="gw">GW{w["gw"]}</div><div class="ft">{ft_txt}</div></div><div>{moves}</div>'
                      f'<div class="pts" style="color:{ui.GREEN if gain - fe.HIT * w["hits"] > 0 else "#b9a7c2"}">{gain - fe.HIT * w["hits"]:+.1f}</div></div>')
        html(rows_)
        html('<div class="note" style="margin-top:6px">Each week\'s number is the points that week compared with keeping your current team, after any hit. '
             'Later weeks are plans, not instructions: the app re-plans every day with new results and team news, so check back before each deadline.</div>')

# ================================================================== COMPARE
elif page == 'Compare':
    G2 = ALL_GWS[:6]
    # players are compared within one position: the stats that earn points differ by position (saves for keepers, goals for forwards...)
    mine_ = [int(i) for i in team['squad'].id] if team else []
    top_ = max(mine_, key=lambda i: P.loc[i, f'xp_{G0}']) if mine_ else int(P[P.status != 'u'][f'xp_{G0}'].idxmax())
    c0, c2 = st.columns([1, 2.4], gap='medium')   # position above the list, gameweek above the comparison
    c1 = st.container()
    def keep_pos():                           # clicking the selected position again would clear it: keep the last one instead
        if st.session_state.get('cmp_pos') is None: st.session_state['cmp_pos'] = st.session_state.get('_cmp_pos_last') or 'FWD'
        st.session_state['_cmp_pos_last'] = st.session_state['cmp_pos']
    pos_lab = c0.segmented_control('Position', ['GK', 'DEF', 'MID', 'FWD'], default=POS[P.pos[top_]], key='cmp_pos', on_change=keep_pos) or POS[P.pos[top_]]
    st.session_state['_cmp_pos_last'] = pos_lab
    pnum = {v: k for k, v in POS.items()}[pos_lab]
    pool = P[(P.status != 'u') & (P.pos == pnum)]
    mine_p = sorted([i for i in mine_ if P.pos[i] == pnum], key=lambda i: -P.loc[i, f'xp_{G0}'])
    default = mine_p[:2] if len(mine_p) >= 2 else list(pool.nlargest(2, f'xp_{G0}').index)
    gsel = c2.segmented_control('Gameweek', [f'GW{g}' for g in G2[:3]], default=f'GW{G0}', key='cmp_gw') or f'GW{G0}'
    g = int(gsel[2:])
    # every player in the position, drawn like the Players tab, with a tick box in front of each row (tick 2 to 4).
    # The picks are kept per position, so changing a filter or switching position never loses them.
    sel_key = f'cmp_{pos_lab}'
    if sel_key not in st.session_state: st.session_state[sel_key] = default
    sel = [int(i) for i in st.session_state[sel_key] if i in pool.index]
    # left: the filters and a scrollable list of names with a tick box each. Right: the comparison, so it shows next to the list.
    left_, right_ = st.columns([1, 2.4], gap='medium')
    with left_:
        tms_ = st.multiselect('Teams', sorted(pool.team.unique()), placeholder='All teams', key=f'cmp_tm_{pos_lab}')
        plo, phi = float(np.floor(pool.price.min() * 2) / 2), float(np.ceil(pool.price.max() * 2) / 2)
        pr_ = st.slider('Price (£m)', plo, phi, (plo, phi), 0.5, key=f'cmp_pr_{pos_lab}')
        shown_ = pool[pool.price.between(*pr_) & (pool.team.isin(tms_) if tms_ else True)].sort_values(f'xp_{g}', ascending=False)

        def toggle(i, k):
            s_ = [int(x) for x in st.session_state.get(sel_key, [])]
            if i in s_: s_.remove(i)
            elif len(s_) < 4: s_.append(i)
            st.session_state[sel_key] = s_

        full_ = len(sel) >= 4
        html(f'<div class="note" style="margin:2px 0 6px 0"><b style="color:#fff">{len(shown_)} {pos_lab}</b> · best projected first · tick 2 to 4'
             + (' · <span style="color:#ebff00">4 ticked: untick one to swap</span>' if full_ else '') + '</div>')
    cmp_ = sel
    hidden_ = [i for i in cmp_ if i not in shown_.index[:80]]
    with right_:
        k_ = st.columns([3, 1], vertical_alignment='center')
        k_[0].markdown('<div class="note">' + (f'<b style="color:#fff">Comparing {len(cmp_)} of 4:</b> ' + ', '.join(
            ui.esc(P.name[i]) + (' <span style="color:#b9a7c2">(not in the list)</span>' if i in hidden_ else '') for i in cmp_)
            if cmp_ else 'No players ticked yet.') + '</div>', unsafe_allow_html=True)
        if k_[1].button('Clear picks', key=f'cmp_clear_{pos_lab}', use_container_width=True, disabled=not cmp_):
            st.session_state[sel_key] = []; st.rerun()
        if len(cmp_) < 2:
            st.info('Tick at least two players in the list to compare them.' if cmp_ else 'Tick two to four players in the list to compare them.')
        else:
            R = P.loc[cmp_]
            # header cards with shirts
            heads = ''.join(f'<td style="text-align:center;padding:10px"><img src="{ui.shirt(r.team_code, r.pos == 1)}" style="width:70px"><div class="bigname" style="font-family:Outfit;font-weight:800;font-size:1.15rem">'
                            f'{ui.esc(r["name"])}</div><div class="note">{r.team} · {POS[r.pos]} · £{r.price:.1f}m</div></td>' for _, r in R.iterrows())
            pct = lambda v: f'{100 * v:.0f}%'
            n0 = lambda v: f'{v:.0f}'; n1 = lambda v: f'{v:.1f}'; n2 = lambda v: f'{v:.2f}'
            S = season_stats().reindex(R.index)
            only = lambda vals, *ps: vals.where(R.pos.isin(ps))          # a stat that only matters for some positions: '-' for the others
            GK, DF, MD, FW = 1, 2, 3, 4
            rows = [  # (label, values, format, higher is better?) or ('#', section title)
                ('#', f'Projection · GW{g}'),
                (f'Projected points GW{g}', R[f'xp_{g}'], n1, True), ('Projected next 3 GWs', R.h3, n1, True),
                ('Chance to score', only(R[f'p_goal_{g}'], DF, MD, FW), pct, True), ('Chance of an assist', only(R[f'p_assist_{g}'], DF, MD, FW), pct, True),
                ('Clean sheet chance', only(R[f'p_cs_{g}'], GK, DF, MD), pct, True),
                ('Defensive contribution chance', only(R[f'p_dc_{g}'], DF, MD, FW), pct, True),
                ('Bonus chance', R[f'p_bonus_{g}'], pct, True), ('Expected bonus points', R[f'xb_{g}'], n2, True),
                ('Plays 60+ mins', R[f'p_60_{g}'], pct, True),
                ('Opponent', R[f'opp_{g}'], str, None), ('Fixture difficulty', R[f'fdr_{g}'], lambda v: f'{v:.0f} / 5', False),
                ('#', 'Season so far'),
                ('Points', R.total_points, n0, True), ('Points per game played', S.ppg, n1, True), ('Minutes', S.minutes, n0, True), ('Starts', S.starts, n0, True),
                ('Goals', only(S.goals, DF, MD, FW), n0, True), ('Expected goals (xG)', only(S.xg, DF, MD, FW), n1, True),
                ('Assists', only(S.assists, DF, MD, FW), n0, True), ('Expected assists (xA)', only(S.xa, DF, MD, FW), n1, True),
                ('Clean sheets', only(S.cs, GK, DF, MD), n0, True), ('Clean sheet points', only(S.cs_pts, GK, DF, MD), n0, True),
                ('Goals conceded', only(S.gc, GK, DF), n0, False), ('Points lost to goals conceded', only(-S.gc_pts, GK, DF), n0, True),
                ('Saves', only(S.saves, GK), n0, True), ('Saves per 90', only(S.saves90, GK), n1, True), ('Save points', only(S.save_pts, GK), n0, True),
                ('Penalties saved', only(S.pen_saved, GK), n0, True),
                ('Defensive contribution points', only(S.dc_pts, DF, MD, FW), n0, True),
                ('Bonus points', S.bonus, n0, True), ('Yellow / red cards', S.cards, str, None),
                ('#', 'Form and value'),
                ('FPL form (pts / match, 30 days)', R.form, n1, True),
                ('Hit the defensive threshold (last 10)', only(R.dc_hit_r10, DF, MD, FW), pct, True), ('Defensive actions per 90 (last 10)', only(R.dc_p90_r10, DF, MD, FW), n1, True),
                ('Got bonus (last 10 games played)', R.bonus_rate_r10, pct, True),
                ('Points per £m (next 3)', R.h3 / R.price, n2, True), ('Owned by', R.selected / 100, pct, None),
                ('vs this opponent (pts vs his usual)', R.h2h_adj, lambda v: f'{v:+.1f}', True), ('Home / away effect (pts vs usual)', R.venue_adj, lambda v: f'{v:+.1f}', True)]
            body = ''
            for lab, vals, *rest in rows:
                if lab == '#':
                    body += f'<tr><td colspan="{len(R) + 1}" style="padding-top:14px"><div class="ptitle" style="margin:0">{vals}</div></td></tr>'; continue
                fmt, hib = rest
                if vals.isna().all(): continue                            # nobody selected plays a position this stat applies to
                v = vals.values
                best = None
                if hib is not None and pd.notna(v).sum() > 1:
                    vv = pd.Series(v, dtype=float); top_ = vv.max() if hib else vv.min()
                    best = (vv.idxmax() if hib else vv.idxmin()) if (vv.round(2) == round(top_, 2)).sum() == 1 else None   # no highlight on a tie
                cells = ''.join(f'<td class="{"best" if k == best else ""}" style="text-align:center">{"-" if (not isinstance(x, str) and pd.isna(x)) else fmt(x)}</td>' for k, x in enumerate(v))
                body += f'<tr><td style="color:#b9a7c2">{lab}</td>{cells}</tr>'
            html(f'<table class="fpl cmp"><tr><td></td>{heads}</tr>{body}</table>')
            html('<div class="note" style="margin-top:6px">Green marks the best in each row. "vs this opponent" and "home / away" compare his past points in that situation with his usual level, '
                 'pulled towards zero when there are only a few games. Testing found both add little to the projection, so treat them as a talking point.</div>')
            sec('Projected points per gameweek')
            fig = go.Figure()
            for i, colr in zip(cmp_, [ui.GREEN, ui.CYAN, ui.PINK, '#ebff00']):
                fig.add_trace(go.Bar(x=[f'GW{x}' for x in G2], y=[P.loc[i, f'xp_{x}'] for x in G2], name=P.name[i], marker_color=colr,
                                     customdata=[P.loc[i, f'opp_{x}'] for x in G2], hovertemplate='%{x} %{customdata}: %{y:.1f} pts<extra>' + P.name[i] + '</extra>'))
            fig = ui.plot_style(fig, 320); fig.update_layout(barmode='group', bargap=0.25)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    with left_:                                 # the scrollable player list, under the filters
        with st.container(height=640, key=f'cmplist_{pos_lab}', border=False):
            for i in shown_.index[:80]:
                i = int(i); ck = f'cmpck_{pos_lab}_{i}'; st.session_state[ck] = i in sel
                rc_ = st.columns([1, 9], vertical_alignment='center')
                rc_[0].checkbox(f'Compare {P.name[i]}', key=ck, on_change=toggle, args=(i, ck), disabled=full_ and i not in sel, label_visibility='collapsed')
                rc_[1].markdown(ui.name_rows(P, [i], picked=sel), unsafe_allow_html=True)
        if len(shown_) > 80: html(f'<div class="note" style="margin-top:4px">The top 80 of {len(shown_)} by projected points. Use the team and price filters to find others.</div>')

# ================================================================== PLANNER: a sandbox for your own transfers
elif page == 'Planner':
    if team is None: need_team(); done()
    # the Planner replays everything from your GW10 squad: your confirmed GW11 transfers are locked moves, your plans come on top
    sq0 = team['orig_squad'].set_index('id'); ids = [int(i) for i in team['orig_ids']]
    PG = ALL_GWS[:3]
    LOCKED = [dict(gw=PG[0], out=int(o), inn=int(i), locked=True) for o, i in team.get('made', [])]
    tkey = (team['name'], tuple(sorted(ids)))
    if st.session_state.get('pl_team') != tkey:                 # a new team starts with an empty plan
        st.session_state['pl_team'] = tkey; st.session_state['pl_moves'] = []; st.session_state['pl_lineup'] = {}
    moves = [m for m in st.session_state.setdefault('pl_moves', []) if not m.get('locked')]
    allm = LOCKED + moves
    LU = st.session_state.setdefault('pl_lineup', {})          # {gameweek: {'bench': [4 ids], 'cap': id, 'vice': id}} chosen by you
    sellv = lambda i: int(sq0.sell[i]) if i in sq0.index else int(round(P.price[i] * 10))   # bought in the plan: sells for what it cost

    def replay(mv):
        """Apply the moves week by week. Returns the squad, bank, free transfers and hits for each week, any rule problems, and the moves that still make sense."""
        squad, bank, ft = list(ids), team['orig_bank'], team['orig_ft']
        out, ok, problems = {}, [], []
        for g in PG:
            wk = [m for m in mv if m['gw'] == g]; n = 0
            for m in wk:
                if m['out'] not in squad or m['inn'] in squad: continue           # no longer possible after an undo
                squad[squad.index(m['out'])] = m['inn']; bank += sellv(m['out']) - int(round(P.price[m['inn']] * 10)); n += 1; ok.append(m)
            used = min(n, ft); hits = n - used
            clubs = P.loc[squad].team.value_counts()
            if bank < 0: problems.append(f'GW{g}: over budget by £{-bank / 10:.1f}m.')
            if clubs.max() > 3: problems.append(f'GW{g}: {clubs.max()} players from {clubs.idxmax()} (the limit is 3).')
            out[g] = dict(squad=list(squad), bank=bank, ft=ft, n=n, hits=hits)
            ft = min(ft - used + 1, 5)
        return out, ok, problems

    def follow(i, g_):
        """A player id chosen for the lineup, updated for transfers made up to week g_ (if he was sold, his replacement takes his place)."""
        for m in allm:
            if m['gw'] <= g_ and m['out'] == i: i = m['inn']
        return i

    def eff_lineup(g_, squad_):
        """Your chosen XI, bench and captain for week g_, or the best XI if you have not chosen one (or it is no longer legal)."""
        best = lineup(tuple(sorted(squad_)), g_, BOOST)
        ov = LU.get(g_)
        if not ov: return best, False
        bench = [follow(b, g_) for b in ov['bench']]
        if len(set(bench)) != 4 or not set(bench) <= set(squad_): return best, False
        xi = [i for i in squad_ if i not in bench]
        cnt = P.loc[xi].pos.value_counts()
        if any(not (fe.XI_MIN[p] <= cnt.get(p, 0) <= fe.XI_MAX[p]) for p in (1, 2, 3, 4)): return best, False
        cap, vice = follow(ov['cap'], g_), follow(ov['vice'], g_)
        byxp = sorted(xi, key=lambda i: -P.loc[i, f'xp_{g_}'])
        if cap not in xi: cap = byxp[0]
        if vice not in xi or vice == cap: vice = next(i for i in byxp if i != cap)
        bench = sorted(bench, key=lambda i: (P.pos[i] != 1, bench.index(i)))      # goalkeeper first, then your order
        return (xi, cap, vice, bench), True

    W, ok_moves, problems = replay(allm)
    ok_user = [m for m in ok_moves if not m.get('locked')]
    if len(ok_user) < len(moves):
        st.session_state['pl_moves'] = moves = ok_user; allm = LOCKED + moves
        st.toast('Removed a planned move that no longer works.')

    # ---- controls
    c1, c2, c3 = st.columns([1.6, 1, 1])
    gsel = c1.segmented_control('Make changes for', [f'GW{g}' for g in PG], default=f'GW{PG[0]}', key='pl_gw') or f'GW{PG[0]}'
    g = int(gsel[2:])
    c2.write(''); c2.write('')
    if c2.button('Start from the suggestion', use_container_width=True, help='Load the Transfers tab plan, then change it as you like'):
        R = run_plan(team['squad'][['id', 'sell', 'now']].to_json(), team['bank'], team['free_transfers'], False, None, 1.0, (), (), tuple(PG), BOOST)
        st.session_state['pl_moves'] = [dict(gw=w['gw'], out=o, inn=i) for w in R['weeks'] for o, i in pairs(dict(out=w['out'], inn=w['inn']))]
        st.rerun()
    c3.write(''); c3.write('')
    if c3.button('Reset', use_container_width=True, disabled=not (moves or LU), help='Clears your plans. Confirmed transfers stay.'):
        st.session_state['pl_moves'] = []; st.session_state['pl_lineup'] = {}; st.rerun()

    # squad going into the chosen week (before that week's own moves are shown as changes)
    before_g = W[PG[PG.index(g) - 1]]['squad'] if PG.index(g) > 0 else ids
    after_g = W[g]['squad']
    rest = PG[PG.index(g):]; colx = lambda i: sum(P.loc[i, f'xp_{x}'] for x in rest)
    dd = st.expander('Or pick from lists (handy on a phone)')
    a_, b_, c_ = dd.columns([1.5, 2.2, 0.8])
    out_opts = sorted(after_g, key=lambda i: (P.pos[i], colx(i)))
    o = a_.selectbox(f'Sell (GW{g})', out_opts, index=None, placeholder='Choose a player to sell', key=f'pl_out_{g}',
                     format_func=lambda i: f'{P.name[i]} ({P.team[i]}, {fe.POS_NAME[P.pos[i]]}) · sells £{sellv(i) / 10:.1f}m')
    i_ = None
    if o is not None:
        money = W[g]['bank'] + sellv(o)
        cand = P[(P.pos == P.pos[o]) & (P.status != 'u') & ~P.index.isin(after_g)].copy(); cand['v'] = [colx(i) for i in cand.index]
        cand = cand.sort_values('v', ascending=False).head(150)
        lab = {i: f'{r["name"]} ({r.team}, £{r.price:.1f}m) · {r.v:.1f} pts over GW{rest[0]}' + (f'-{rest[-1]}' if len(rest) > 1 else '')
               + ('  · over budget' if round(r.price * 10) > money else '') for i, r in cand.iterrows()}
        i_ = b_.selectbox(f'Buy · money available £{money / 10:.1f}m', cand.index.tolist(), index=None, placeholder='Choose a replacement',
                          key=f'pl_in_{g}_{o}', format_func=lambda i: lab[i])
    else:
        b_.selectbox('Buy', ['Pick who to sell first'], disabled=True, key='pl_in_blank')
    c_.write(''); c_.write('')
    if c_.button('Add', type='primary', use_container_width=True, disabled=i_ is None):
        st.session_state['pl_moves'] = moves + [dict(gw=g, out=int(o), inn=int(i_))]; st.rerun()

    # ---- your moves, each with an undo
    if LOCKED or moves:
        sec('Your changes', ('🔒 confirmed ones are locked · ' if LOCKED else '') + 'click ✕ to undo a planned one')
        for m in LOCKED:
            html(f'<div class="move" style="margin:2px 0"><div style="display:flex;align-items:center;gap:10px"><b style="font-family:Outfit;color:#00ff87">GW{m["gw"]}</b>'
                 f'{move_card(m["out"], "out", PG)}</div><div class="arrow">→</div><div style="display:flex;align-items:center;gap:10px">{move_card(m["inn"], "in", PG)}'
                 f'<span title="Confirmed in the FPL app" style="font-size:1.3rem">🔒</span></div></div>')
        for k, m in enumerate(moves):
            x1, x2 = st.columns([12, 1])
            x1.markdown(f'<div class="move" style="margin:2px 0"><div style="display:flex;align-items:center;gap:10px"><b style="font-family:Outfit;color:#00ff87">GW{m["gw"]}</b>'
                        f'{move_card(m["out"], "out", PG[PG.index(m["gw"]):])}</div><div class="arrow">→</div>{move_card(m["inn"], "in", PG[PG.index(m["gw"]):])}</div>', unsafe_allow_html=True)
            x2.write('')
            if x2.button('✕', key=f'undo_{k}', help='Undo this change'):
                st.session_state['pl_moves'] = moves[:k] + moves[k + 1:]; st.rerun()
    else:
        html('<div class="note" style="margin:6px 0">No changes yet. Pick a player to sell and a replacement, or start from the suggestion. '
             'Changes in GW11 carry into GW12 and GW13, just like real transfers.</div>')

    # ---- totals
    pts = lambda sq_, gw_: (lambda lu: P.loc[lu[0], f'xp_{gw_}'].sum() + P.loc[lu[1], f'xp_{gw_}'])(lineup(tuple(sorted(sq_)), gw_, BOOST))
    pts_eff = lambda gw_: (lambda lu: P.loc[lu[0], f'xp_{gw_}'].sum() + P.loc[lu[1], f'xp_{gw_}'])(eff_lineup(gw_, W[gw_]['squad'])[0])
    gains = {x: pts_eff(x) - pts(ids, x) - fe.HIT * W[x]['hits'] for x in PG}   # vs your GW10 squad, confirmed transfers and their hits included
    net = sum(gains.values()); hits_tot = sum(W[x]['hits'] for x in PG)
    colr = ui.GREEN if net > 0.05 else (ui.PINK if net < -0.05 else '#ffffff')
    html(ui.cards([(f'<span style="color:{colr}">{net:+.1f}</span>', f'points over GW{PG[0]}-{PG[-1]} vs no transfers (after hits)', ''),
                   (str(len(moves)), 'changes in your plan', 'g'), (f'−{fe.HIT * hits_tot}' if hits_tot else '0', 'hit points', ''),
                   (f'£{W[PG[-1]]["bank"] / 10:.1f}m', f'bank after GW{PG[-1]}', 'c' if W[PG[-1]]['bank'] >= 0 else '')]))
    for p_ in problems: st.error(p_)

    # ---- the chosen week: now vs with your changes. Click a shirt on the right to replace him.
    sec(f'GW{g}: your team now vs with your changes')
    html('<div class="hint">👆 Click a player\'s name on the right-hand pitch to transfer him, move him to or from the bench, or make him captain</div>')
    xp = P[f'xp_{g}']
    orig = team.get('orig_ids', ids); conf = team.get('made', [])
    lb = lineup(tuple(sorted(orig)), g, BOOST); la, chosen = eff_lineup(g, after_g)
    pb = xp[lb[0]].sum() + xp[lb[1]]; pa = xp[la[0]].sum() + xp[la[1]]
    outs = [i for i in orig if i not in after_g]; ins = [i for i in after_g if i not in orig]
    best_after = lineup(tuple(sorted(after_g)), g, BOOST)
    moved = [i for i in la[0] if i not in best_after[0]] + [i for i in la[3] if i not in best_after[3]] if chosen else []
    html(delta_bar(pb, pa, g, fe.HIT * W[g]['hits']))
    p1, p3 = st.columns(2)
    with p1:
        html(f'<div class="ptitle">Your GW{ALL_GWS[0] - 1} team <small>{pb:.1f} pts in GW{g} · best XI</small></div>')
        click_pitch(lb[0], lb[3], lb[1], lb[2], g, outs=set(outs), clickable=False, tag='now', order=list(orig))
    with p3:
        html(f'<div class="ptitle">With your changes <small>{pa:.1f} pts · ' + ('your lineup' if chosen else 'best XI') + (' · includes confirmed' if conf else '') + '</small></div>')
        clicked = click_pitch(la[0], la[3], la[1], la[2], g, ins=set(ins), moved=set(moved), tag='new', order=list(after_g))
    html('<div class="note" style="margin-top:4px">Red: going out · green: coming in · cyan: moved by your lineup change. '
         + ('Your lineup applies to this gameweek only; other weeks use their best XI unless you set them too.' if chosen else 'Until you change it, each week uses its best XI and captain.') + '</div>')
    if chosen and st.button('Use the best XI again for GW%d' % g, key=f'lu_reset_{g}'):
        LU.pop(g, None); st.rerun()
    if clicked is not None:
        replace_dialog(clicked, g, after_g, W[g]['bank'] + sellv(clicked), PG[PG.index(g):], moves, la)

    # ---- week by week
    sec('Week by week', 'points that week vs keeping your current team, after hits')
    rows_ = ''
    for x in PG:
        mv = [m for m in allm if m['gw'] == x]
        txt = ''.join(f'<span class="mv"><img src="{ui.shirt(P.team_code[m["out"]], P.pos[m["out"]] == 1)}"><span><b>{ui.esc(P.name[m["out"]])}</b> → '
                      f'<b style="color:#00ff87">{ui.esc(P.name[m["inn"]])}</b>{" 🔒" if m.get("locked") else ""}</span></span>' for m in mv) or '<span style="color:#b9a7c2">No changes · the transfer rolls over</span>'
        w = W[x]; ft_txt = f'{w["ft"]} free · uses {min(w["n"], w["ft"])}' + (f' + <b style="color:#e90052">{w["hits"]} hit (−{fe.HIT * w["hits"]})</b>' if w['hits'] else '')
        rows_ += (f'<div class="week"><div><div class="gw">GW{x}</div><div class="ft">{ft_txt}</div></div><div>{txt}</div>'
                  f'<div class="pts" style="color:{ui.GREEN if gains[x] > 0.05 else (ui.PINK if gains[x] < -0.05 else "#b9a7c2")}">{gains[x]:+.1f}</div></div>')
    html(rows_)
    if moves and not problems:
        verdict = 'Worth it.' if net > 1 else ('Marginal: close to a coin flip.' if net > -1 else 'Not worth it.')
        html(f'<div class="note" style="margin-top:6px"><b style="color:{colr};font-size:1rem">{verdict}</b> Compare with the plan on the Transfers tab to see if the optimiser finds better. '
             'Wildcard and Free Hit plans are on the Transfers tab.</div>')

# ================================================================== LEAGUE: beat your rivals
elif page == 'League':
    c1, c2 = st.columns([2, 1])
    # the box starts from the saved league ID every time it is drawn, so it stays filled across tabs, refreshes and visits
    lid = c1.text_input('Mini-league ID', value=st.session_state.get('_lid', ''), key='lid_box', placeholder='e.g. 123456', on_change=save_lid,
                        help='Open the league on the FPL website. The number after /leagues/ in the address is its ID (not the invite code).')
    if lid.strip(): st.session_state['_lid'] = lid.strip()        # keep it for when you come back from another tab
    if lid.strip().isdigit() and st.query_params.get('league') != lid.strip(): st.query_params['league'] = lid.strip()
    mx = c2.select_slider('Managers to analyse', [10, 20, 30, 50], 30, key='lmax', help='The top of the table. More managers take longer to load.')
    if not lid.strip().isdigit():
        html('<div class="card" style="padding:18px"><div class="v" style="font-size:1.15rem">Enter your mini-league ID</div><div class="l" style="font-size:.9rem;margin-top:6px">'
             'See the table, how many rivals own each player, the players that threaten you, the differentials that gain on them, rivals\' chips, and a transfer plan '
             'aimed at climbing your league. Add your own team ID in the sidebar too, so the app knows which manager is you.</div></div>'); done()

    @st.cache_data(ttl=1800, show_spinner='Loading the league and every rival\'s team...')
    def get_league(lid_, mx_, cur_):
        return fe.load_league(int(lid_), boot, mx_, cur=cur_)
    try: L = get_league(lid.strip(), mx, LIVE_GW)
    except requests.HTTPError: st.error('No classic league found with that ID. Head-to-head leagues are not supported yet.'); done()
    except Exception as e: st.error(f'Could not load the league ({e}).'); done()
    me_id = int(st.session_state['eid']) if team is not None and str(st.session_state.get('eid', '')).isdigit() else None
    M = pd.DataFrame(L['managers'])
    me = M[M.entry == me_id].iloc[0] if me_id is not None and (M.entry == me_id).any() else None
    own, capshare, n_riv = fe.league_ownership(L, exclude=me_id)
    gws_left = len(ALL_GWS)
    G3 = ALL_GWS[:3]
    proj = lambda sq_, g: (lambda lu: P.loc[lu[0], f'xp_{g}'].sum() + P.loc[lu[1], f'xp_{g}'])(lineup(tuple(sorted(int(i) for i in sq_ if i in P.index)), g, BOOST))
    M['next_gw'] = [proj(sq_, G0) for sq_ in M.squad]
    M['next3'] = [sum(proj(sq_, g) for g in G3) for sq_ in M.squad]
    if me_id is not None and team is not None and (M.entry == me_id).any():     # your row uses your current team (confirmed transfers, minus any hit)
        my_now = [int(i) for i in team['squad'].id]; k_ = M.index[M.entry == me_id][0]
        M.at[k_, 'next_gw'] = proj(my_now, G0) - HIT0; M.at[k_, 'next3'] = sum(proj(my_now, g) for g in G3) - HIT0
    my_sq = [int(i) for i in team['squad'].id] if team is not None else (list(me.squad) if me is not None else [])

    # ---- headline
    leader = M.iloc[0]
    if me is not None:
        above = M[M['rank'] < me['rank']]
        gap_lead = leader.total - me.total; gap_up = (above.iloc[-1].total - me.total) if len(above) else 0
        my_next3 = sum(proj(my_sq, g) for g in G3) - HIT0
        html(ui.cards([(f'{int(me["rank"])} / {L["n_total"] if not L["has_more"] else str(L["n_total"]) + "+"}', f'your rank in {ui.esc(L["name"])}', 'g'),
                       ('Leader' if gap_lead == 0 else f'{gap_lead} pts', 'behind the leader' if gap_lead else f'{int(me.total - M.iloc[1].total) if len(M) > 1 else 0} pts clear of 2nd', ''),
                       (f'{gap_up} pts' if gap_up else '-', 'to the manager above you', ''), (str(gws_left), 'gameweeks left (including this one)', 'c'),
                       (f'{my_next3 - M[M.entry != me_id].next3.mean():+.1f}', 'your projected GW%d-%d points vs the league average' % (G3[0], G3[-1]), '')]))
        per_week = gap_lead / max(gws_left, 1)
        if gap_lead == 0:
            mood = 'safe'; adv = f'You lead. Protect it: own the players most of your rivals own (they cannot gain on you through them) and captain the popular choice unless the numbers clearly say otherwise.'
        elif per_week <= 1.5:
            mood = 'balanced'; adv = f'You are {gap_lead} points behind with {gws_left} gameweeks left, about {per_week:.1f} a week. Very catchable: mostly maximise points, with a differential or two.'
        else:
            mood = 'chase'; adv = f'You are {gap_lead} points behind, about {per_week:.1f} a week. You need players your rivals do not have: their points move you up, template points do not.'
        html(f'<div class="card" style="padding:14px 16px;margin-top:4px"><div class="l" style="font-size:.95rem;color:#fff"><b style="color:#00ff87">Strategy · </b>{adv}</div></div>')
    else:
        mood = 'balanced'
        html(f'<div class="note">Showing <b>{ui.esc(L["name"])}</b>. Enter your own team ID in the sidebar (and make sure you are in this league\'s top {mx}) to see your gaps, threats and a plan.</div>')

    # ---- table
    ICON = {'Wildcard': '🃏', 'Free Hit': '🎯', 'Bench Boost': '🪑', 'Triple Captain': '👑'}
    sec('The table', f'top {len(M)} · projections use each manager\'s latest team (rivals\' new transfers stay hidden until the deadline)')
    body = ''
    for m in M.itertuples():
        mv = (m.last_rank or m.rank) - m.rank
        arrow = f'<span style="color:#00ff87">▲{mv}</span>' if mv > 0 else (f'<span style="color:#e90052">▼{-mv}</span>' if mv < 0 else '<span style="color:#b9a7c2">•</span>')
        gap = '' if me is None else (f'{m.total - me.total:+d}' if m.entry != me_id else '<b style="color:#00ff87">you</b>')
        rowst = 'style="outline:1px solid #00ff87"' if m.entry == me_id else ''
        body += (f'<tr {rowst}><td><b>{m.rank}</b> {arrow}</td><td><b>{ui.esc(m.team)}</b><div class="note">{ui.esc(m.manager)}</div></td><td><b>{m.total}</b></td><td>{gap}</td>'
                 f'<td>{m.gw_points}</td><td>{ui.esc(P.name.get(m.captain, "-"))}</td><td>{"".join(ICON.get(c, "") for c in m.chips_left) or "-"}</td>'
                 f'<td><b style="color:#04f5ff">{m.next_gw:.0f}</b></td><td>{m.next3:.0f}</td></tr>')
    html(f'<table class="fpl"><tr><th>Rank</th><th>Team</th><th>Total</th><th>Gap to you</th><th>GW{L["gw"]}</th><th>GW{L["gw"]} captain</th>'
         f'<th>Chips left</th><th>Proj. GW{G0}</th><th>Proj. GW{G3[0]}-{G3[-1]}</th></tr>{body}</table>')
    html('<div class="note" style="margin-top:4px">🃏 Wildcard · 🎯 Free Hit · 🪑 Bench Boost · 👑 Triple Captain still available this half. A rival holding Bench Boost or Triple Captain can jump in a double gameweek.</div>')

    # ---- head to head: your team vs one rival
    sec('Head to head', 'your team next to any rival\'s, and where the points will differ')
    rivals = M[M.entry != me_id]
    if len(rivals) and (my_sq or me is not None):
        default_r = (M[M['rank'] < me['rank']].iloc[-1].entry if me is not None and (M['rank'] < me['rank']).any() else rivals.iloc[0].entry)
        opts_r = rivals.entry.tolist()
        rlab = {m.entry: f'{m.rank}. {m.team} ({m.manager}) · {m.total} pts' for m in rivals.itertuples()}
        rid = st.selectbox('Compare with', opts_r, index=opts_r.index(default_r), format_func=rlab.get, key='h2h_rival')
        R_ = M[M.entry == rid].iloc[0]
        their = [int(i) for i in R_.squad if i in P.index]; mine_ = [int(i) for i in my_sq if i in P.index]
        shared = set(their) & set(mine_); only_me = [i for i in mine_ if i not in shared]; only_them = [i for i in their if i not in shared]
        hz_h = st.segmented_control('Projected over', ['Next GW', 'Next 3'], default='Next GW', key='h2h_hz') or 'Next GW'
        gws_h = ALL_GWS[:1] if hz_h == 'Next GW' else G3
        tot_me = sum(proj(mine_, g_) for g_ in gws_h) - HIT0; tot_th = sum(proj(their, g_) for g_ in gws_h)
        ptsx = lambda i: sum(P.loc[i, f'xp_{g_}'] for g_ in gws_h)
        gap_now = (me.total - R_.total) if me is not None else None
        html(ui.cards([(f'{len(shared)} / 15', 'players you both own', ''),
                       (f'{tot_me:.1f} vs {tot_th:.1f}', f'projected points {"GW%d" % G0 if len(gws_h) == 1 else "GW%d-%d" % (G3[0], G3[-1])} (best XI each)', 'c'),
                       (f'<span style="color:{ui.GREEN if tot_me >= tot_th else ui.PINK}">{tot_me - tot_th:+.1f}</span>', 'expected swing in your favour', ''),
                       (f'{gap_now:+d}' if gap_now is not None else '-', 'current gap (you minus them)', '')]))
        g_ = G0; xpg = P[f'xp_{g_}']
        lm, lt = lineup(tuple(sorted(mine_)), g_, BOOST), lineup(tuple(sorted(their)), g_, BOOST)
        h1, h2 = st.columns(2)
        with h1:
            html(f'<div class="ptitle">You <small>{xpg[lm[0]].sum() + xpg[lm[1]] - HIT0:.1f} pts in GW{g_}' + (f' (after −{HIT0} hit)' if HIT0 else '') + '</small></div>')
            click_pitch(lm[0], lm[3], lm[1], lm[2], g_, ins=set(only_me), clickable=False, tag='h2hme')
        with h2:
            html(f'<div class="ptitle">{ui.esc(R_.team)} <small>{xpg[lt[0]].sum() + xpg[lt[1]]:.1f} pts in GW{g_} · their GW{L["gw"]} squad</small></div>')
            click_pitch(lt[0], lt[3], lt[1], lt[2], g_, diffs=set(only_them), clickable=False, tag='h2hthem')
        html('<div class="note" style="margin-top:4px">Green: only you own him · pink: only they own him · white: you both own him (his points cancel out between you). '
             'Both teams use their best XI and captain for GW%d. Their transfers for GW%d stay hidden until the deadline.</div>' % (g_, G0))
    else:
        html('<div class="note">Add your team ID in the sidebar to compare your team with a rival.</div>')

    # ---- ownership: threats and differentials
    sec('League analysis', 'who your rivals own, and where you can gain or lose ground')
    O = P[P.status != 'u'].copy(); O['lown'] = own.reindex(O.index).fillna(0); O['lcap'] = capshare.reindex(O.index).fillna(0)
    ex = lambda lab, f: (lab, f)
    tA, tB = st.columns(2)
    with tA:
        html('<div class="ptitle" style="white-space:nowrap">Threats <small>popular with rivals, not in your team</small></div>')
        T_ = O[~O.index.isin(my_sq) & (O.lown >= 0.25)].copy(); T_['risk'] = T_.h3 * T_.lown
        T_ = T_.sort_values('risk', ascending=False).head(8)
        html(''.join(f'<div class="mv out" style="margin:5px 0"><img src="{ui.shirt(r.team_code, r.pos == 1)}"><div><div class="bigname">{ui.esc(r["name"])} '
                     f'<span class="note">{r.team}</span></div><div class="m">owned by <b style="color:#fff">{100 * r.lown:.0f}%</b> of rivals · {r.h3:.1f} pts over GW{G3[0]}-{G3[-1]} · '
                     f'£{r.price:.1f}m</div></div></div>' for _, r in T_.iterrows()) or '<div class="note">No big threats: you own the popular players.</div>')
    with tB:
        html('<div class="ptitle" style="white-space:nowrap">League differentials <small>few rivals own them</small></div>')
        D_ = O[~O.index.isin(my_sq) & (O.lown <= 0.15) & (O.chance.isna() | (O.chance >= 75))].copy(); D_['edge'] = D_.h3 * (1 - D_.lown)
        D_ = D_.sort_values('edge', ascending=False).head(8)
        html(''.join(f'<div class="mv in" style="margin:5px 0"><img src="{ui.shirt(r.team_code, r.pos == 1)}"><div><div class="bigname">{ui.esc(r["name"])} '
                     f'<span class="note">{r.team}</span></div><div class="m">owned by <b style="color:#fff">{100 * r.lown:.0f}%</b> of rivals · {r.h3:.1f} pts over GW{G3[0]}-{G3[-1]} · '
                     f'£{r.price:.1f}m</div></div></div>' for _, r in D_.iterrows()))
    if my_sq:
        sec('Your team vs the league', 'how many rivals own each of your players')
        mine = O.loc[[i for i in my_sq if i in O.index]].sort_values('lown', ascending=False)
        html('<table class="fpl"><tr><th>Player</th><th>Owned by rivals</th><th>What it means</th><th>Proj. GW%d-%d</th></tr>' % (G3[0], G3[-1]) +
             ''.join(f'<tr><td><div class="who"><img src="{ui.shirt(r.team_code, r.pos == 1)}"><b>{ui.esc(r["name"])}</b></div></td>'
                     f'<td><div class="xpbar" style="min-width:120px"><i style="width:{max(3, 100 * r.lown):.0f}%"></i><span>{100 * r.lown:.0f}%</span></div></td>'
                     f'<td class="note">{"Shield: his points keep you level" if r.lown >= 0.5 else ("Differential: his points gain you ground" if r.lown <= 0.2 else "Partly shared")}</td>'
                     f'<td>{r.h3:.1f}</td></tr>' for _, r in mine.iterrows()) + '</table>')
    cs = capshare.sort_values(ascending=False).head(3)
    if len(cs):
        html('<div class="note" style="margin-top:8px"><b style="color:#fff">Last week\'s captains:</b> ' + ' · '.join(f'{ui.esc(P.name.get(i, "?"))} {100 * v:.0f}%' for i, v in cs.items())
             + '. Captaining the popular choice keeps you level with them; a different captain is a bigger swing either way.</div>')

    # ---- league-mode transfer plan
    if team is not None:
        sec('Transfers to climb your league', f'GW{G3[0]}-{G3[-1]} plan')
        style = st.segmented_control('Approach', ['Safe', 'Balanced', 'Chase'], default={'safe': 'Safe', 'balanced': 'Balanced', 'chase': 'Chase'}[mood], key='lstyle',
                                     help='Safe: just maximise points. Chase: value points by how few rivals own the player. Balanced: in between.') or 'Balanced'
        lam = {'Safe': 0.0, 'Balanced': 0.5, 'Chase': 1.0}[style]

        @st.cache_data(show_spinner='Planning...')
        def league_plan(squad_json, bank, ft, own_items, lam_, gws_, boost):
            from io import StringIO
            o = pd.Series(dict(own_items), dtype=float) * lam_
            return fe.optimise_plan(fe.league_P(P, o, list(gws_)), list(gws_), pd.read_json(StringIO(squad_json)), bank, ft)
        R = league_plan(team['squad'][['id', 'sell', 'now']].to_json(), team['bank'], team['free_transfers'], tuple(own.items()), lam, tuple(G3), BOOST)
        rows_ = ''
        for w in R['weeks']:
            raw = proj(w['squad'], w['gw']) - proj(my_sq, w['gw'])
            moves = ''.join(f'<span class="mv"><img src="{ui.shirt(P.team_code[o], P.pos[o] == 1)}"><span><b>{ui.esc(P.name[o])}</b> '
                            f'<span class="note">({100 * own.get(o, 0):.0f}% own)</span> → <b style="color:#00ff87">{ui.esc(P.name[i])}</b> '
                            f'<span class="note">({100 * own.get(i, 0):.0f}% own)</span></span></span>' for o, i in pairs(dict(out=w['out'], inn=w['inn']))) or '<span style="color:#b9a7c2">Hold · roll the transfer</span>'
            rows_ += (f'<div class="week"><div><div class="gw">GW{w["gw"]}</div><div class="ft">{w["ft_before"]} free</div></div><div>{moves}</div>'
                      f'<div class="pts" style="color:{ui.GREEN if raw > 0.05 else "#b9a7c2"}">{raw:+.1f}</div></div>')
        html(rows_)
        html('<div class="note" style="margin-top:6px">Numbers are projected points that week vs keeping your team. In Chase, the plan accepts fewer raw points for players your rivals do not own, '
             'because those points actually move you up the table. "% own" is the share of your rivals who own him.</div>')

# ================================================================== CHIPS
elif page == 'Chips':
    if team is None: need_team(); done()
    used = team.get('chips_used', [])
    ALLC = ['Wildcard', 'Free Hit', 'Bench Boost', 'Triple Captain']
    ICON = {'Wildcard': '🃏', 'Free Hit': '🎯', 'Bench Boost': '🪑', 'Triple Captain': '👑'}
    used_in = lambda c, h: [g for n, g in used if n == c and fe.HALVES[h][0] <= g <= fe.HALVES[h][1]]
    half_now = 1 if G0 <= 19 else 2
    sj = team['squad'][['id', 'sell', 'now']].to_json()

    @st.cache_data(show_spinner=False)
    def chip_table(squad_json, bank, ft, gws, wc_starts, boost):
        from io import StringIO
        return fe.chip_values(P, pd.read_json(StringIO(squad_json)), bank, ft, list(gws), wc_starts=wc_starts)

    for h in (1, 2):
        lo, hi = fe.HALVES[h]
        if h < half_now: continue
        gws = [g for g in ALL_GWS if lo <= g <= hi]
        left = [c for c in ALLC if not used_in(c, h)]
        title = f'Gameweeks {lo} to {hi}' + (' (now)' if h == half_now else ' (chips refresh)')
        sec(title, 'first set must be played before the GW19 deadline' if h == 1 else 'a full new set of 4 chips')
        cards_ = []
        for c in ALLC:
            u = used_in(c, h)
            cards_.append((f'{ICON[c]} {c}', f'<b style="color:#e90052">Used in GW{u[0]}</b>' if u else '<b style="color:#00ff87">Available</b>', ''))
        html(ui.cards(cards_))
        if not left or not gws:
            html('<div class="note">No chips left in this half.</div>'); continue
        with st.spinner('Working out the best week for each chip...'):
            V = chip_table(sj, team['bank'], team['free_transfers'], tuple(gws), 6 if h == half_now else 0, BOOST)
        fh_ban = set()
        if h == 2 and (19 in used_in('Free Hit', 1)): fh_ban = {20}            # no Free Hit in two weeks running
        plan = fe.plan_chips(V, left, fh_banned=fh_ban)
        why = {'Bench Boost': lambda g: f'your bench is projected {V.bb[g]:.1f} pts',
               'Triple Captain': lambda g: f'{ui.esc(V.tc_name[g])} captained: +{V.tc[g]:.1f} pts',
               'Free Hit': lambda g: f'a one-week squad scores {V.fh[g]:+.1f} pts more than yours',
               'Wildcard': lambda g: f'a rebuilt squad gains {V.wc[g]:+.1f} pts over 4 weeks vs normal transfers'}
        col = {'Bench Boost': 'bb', 'Triple Captain': 'tc', 'Free Hit': 'fh', 'Wildcard': 'wc'}
        plan_cards = []
        for c in ALLC:
            if c in plan:
                vals = V[col[c]].dropna(); edge = vals.max() - vals.median()
                # if the best week is barely better than a typical week, say so: a double gameweek would usually beat all of them
                flat = ' <span style="color:#ffab1b">· no standout week yet: a double gameweek would usually be worth more</span>' if edge < 1.0 and not V.doubles.any() else ''
                plan_cards.append((f'GW{plan[c]}', f'{ICON[c]} <b>{c}</b> · {why[c](plan[c])}{flat}', 'g'))
            elif c in left: plan_cards.append(('Later', f'{ICON[c]} <b>{c}</b> · decide nearer the time (needs fixtures and team news)', ''))
        html('<div class="note" style="margin:6px 0 2px 0"><b style="color:#fff">Suggested plan</b> · one chip per gameweek</div>' + ui.cards(plan_cards))
        # per-gameweek values, best week per chip highlighted
        show = [c for c in left]
        head = '<tr><th>Gameweek</th>' + ''.join(f'<th>{ICON[c]} {c}</th>' for c in show) + '<th>Fixtures</th></tr>'
        body = ''
        for g in gws:
            cells = ''
            for c in show:
                v = V.loc[g, col[c]]
                best_ = plan.get(c) == g
                nm_style = 'color:#37003c;font-weight:600' if best_ else 'color:#b9a7c2'
                txt = '-' if pd.isna(v) else (f'+{v:.1f}' + (f' <span style="{nm_style}">({ui.esc(V.tc_name[g])})</span>' if c == 'Triple Captain' else ''))
                cells += f'<td style="{"color:#1b0021;background:#00ff87;font-weight:800" if best_ else ""}">{txt}</td>'
            fx = ('<b style="color:#04f5ff">' + f'{V.doubles[g]} doubles</b> ' if V.doubles[g] else '') + (f'<b style="color:#e90052">{V.blanks[g]} blanks</b>' if V.blanks[g] else '')
            body += f'<tr><td><b>GW{g}</b></td>{cells}<td>{fx or "normal"}</td></tr>'
        html(f'<table class="fpl">{head}{body}</table>')
        if h == half_now:
            fig = go.Figure()
            for c, colr in zip(show, [ui.GREEN, ui.CYAN, ui.PINK, '#ebff00']):
                fig.add_trace(go.Scatter(x=[f'GW{g}' for g in gws], y=V[col[c]].values, name=c, mode='lines+markers', line=dict(color=colr, width=3)))
            fig = ui.plot_style(fig, 290); fig.update_yaxes(title='expected points gained')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    html('<div class="note" style="margin-top:8px"><b>How to read this.</b> Each number is the expected points a chip adds in that gameweek for <i>your</i> squad. '
         'Bench Boost = your bench\'s points. Triple Captain = one extra helping of your captain. Free Hit = the best one-week team you could afford minus your team. '
         'Wildcard = a full rebuild over the next 4 weeks minus what your normal free transfers would do (only shown for the next 6 weeks). '
         'Later weeks assume you keep this squad and use today\'s form, so they are rougher. <b>Double gameweeks change everything:</b> FPL usually announces them a few weeks ahead, '
         'and this plan updates automatically when they appear. The plan follows the rules: one chip per week, no Free Hit in two weeks running, and the first set expires at the GW19 deadline.</div>')

# ================================================================== CAPTAIN
elif page == 'Captain':
    if team is None: need_team(); done()
    ids = tuple(int(i) for i in team['squad'].id)
    xi, cap, vice, bench = lineup(ids, G0, BOOST); xp = P[f'xp_{G0}']
    c = sorted(xi, key=lambda i: -xp[i])[:8]
    sec(f'Captain picks for GW{G0}', 'from your starting XI · points shown doubled')
    fig = go.Figure(go.Bar(y=[P.name[i] for i in c][::-1], x=[2 * xp[i] for i in c][::-1], orientation='h',
                           marker=dict(color=[ui.GREEN if i == cap else (ui.CYAN if i == vice else '#7a3f86') for i in c][::-1]),
                           text=[f'{2 * xp[i]:.1f} pts · {P.loc[i, f"opp_{G0}"]} · owned {P.selected[i]:.0f}%' for i in c][::-1], textposition='auto',
                           hovertemplate='%{y}: %{x:.1f} pts<extra></extra>'))
    fig = ui.plot_style(fig, 380); fig.update_xaxes(title='projected points as captain')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    gap = 2 * (xp[c[0]] - xp[c[1]])
    html(f'<div class="cards">' + ''.join(f'<div class="card"><div class="v {"g" if i == cap else "c"}">{ui.esc(P.name[i])}</div><div class="l">{lab} · {2 * xp[i]:.1f} pts</div></div>'
                                       for i, lab in ((cap, 'Captain'), (vice, 'Vice-captain'))) + '</div>')
    html(f'<div class="note" style="margin-top:8px">The top two are {gap:.1f} points apart as captain. A gap under about 1 point is close to a coin flip. '
         'A popular captain protects your rank; a less-owned one gains more if he hauls.</div>')
    sec('Best captains in the game', f'all players, GW{G0}')
    html(ui.player_table(P, P[P.status != 'u'].nlargest(10, f'xp_{G0}').index, ALL_GWS, f'xp_{G0}'))

    sec('Differential captains', f'GW{G0} · owned by under 20%')
    C_ = P[(P.status != 'u') & (P.selected < 20) & (P.chance.isna() | (P.chance >= 75))].nlargest(5, f'xp_{G0}')
    html(ui.player_table(P, C_.index, ALL_GWS, f'xp_{G0}', extra=('As captain', lambda i: f'<b>{2 * P.loc[i, f"xp_{G0}"]:.1f}</b>')))

# ================================================================== PLAYERS
elif page == 'Players':
    f1, f2, f3 = st.columns([1.3, 1.9, 1])
    pos = f1.pills('Position', ['GK', 'DEF', 'MID', 'FWD'], selection_mode='multi', default=['GK', 'DEF', 'MID', 'FWD'])
    hz = f2.segmented_control('Projected over', ['Next GW', 'Next 2', 'Next 3'], default='Next GW', key='hz_pl') or 'Next GW'
    sort = f3.selectbox('Sort by', ['Projected points', 'Edge vs other managers', 'Points per £m', 'Goal chance', 'Clean sheet chance', 'Defensive contribution chance', 'Bonus chance', 'Price', 'Ownership', 'Form'])
    g1, g2, g3, g4, g5 = st.columns([1.5, 1.4, 1, 0.7, 0.9])
    teams = g1.multiselect('Teams', sorted(P.team.unique()), placeholder='All teams')
    lo, hi = g2.slider('Price (£m)', 3.5, 16.0, (3.5, 16.0), 0.5)
    own = g3.slider('Max owned (%)', 0, 100, 100, 5)
    n = g4.selectbox('Show', [25, 50, 100, 200])
    g5.write(''); g5.write(''); fit = g5.toggle('Fit only', True, help='Hide players with less than a 75% chance of playing')
    q = st.text_input('Search', placeholder='Search player or team', label_visibility='collapsed')
    col = f'h{HORIZONS[hz]}'
    D = P[(P.status != 'u') & P.price.between(lo, hi) & (P.selected <= own) & P.pos.isin([k for k, v in POS.items() if v in (pos or POS.values())])]
    if teams: D = D[D.team.isin(teams)]
    if fit: D = D[D.chance.isna() | (D.chance >= 75)]
    if q: D = D[D.name.str.contains(q, case=False) | D.team.str.contains(q, case=False)]
    key = {'Projected points': D[col], 'Edge vs other managers': D[col] * (1 - D.selected / 100), 'Points per £m': D[col] / D.price, 'Price': D.price, 'Ownership': D.selected, 'Form': D.form,
           'Goal chance': D[f'p_goal_{G0}'], 'Clean sheet chance': D[f'p_cs_{G0}'].where(D.pos != 4, 0), 'Defensive contribution chance': D[f'p_dc_{G0}'],
           'Bonus chance': D[f'p_bonus_{G0}']}[sort]
    D = D.loc[key.sort_values(ascending=False).index]
    sec(f'{len(D)} players', f'projected points over {hz.lower()} · bars scaled to the best shown')
    html(ui.player_table(P, D.index[:n], ALL_GWS, col, max_val=D[col].max() if len(D) else 1, detail=last5_cards()))
    # a click anywhere outside an open 'last 5' card closes it (so only one is open at a time). Page text cannot run scripts,
    # so this tiny invisible component adds one click listener to the page, once.
    import streamlit.components.v1 as components
    components.html("""<script>
      const d = window.parent.document;
      if (!d.__pdClose) { d.__pdClose = true;
        d.addEventListener('click', e => d.querySelectorAll('details.pd[open]').forEach(x => { if (!x.contains(e.target)) x.removeAttribute('open'); }), true); }
    </script>""", height=0)
    html('<div class="note" style="margin-top:6px">Click a player to see his last 5 games and how he got his points (click again to close). '
         'Want two players side by side? Open the <b>Compare</b> tab.</div>')

# ================================================================== TEAMS: league table, recent results and team stats, then the fixture ticker
elif page == 'Teams':
    @st.cache_data(ttl=3600, show_spinner=False)
    def team_season():
        """This season's results per team from the fixture list: the table, the last 5 results and clean sheets."""
        fx = json.load(open(ROOT / 'data' / '2026-27' / 'fixtures.json'))
        sn = {t['id']: t['short_name'] for t in boot['teams']}
        rows = []
        for f in fx:
            if not f['finished'] or f['team_h_score'] is None: continue
            for side, tid, oid, gf, ga in (('H', f['team_h'], f['team_a'], f['team_h_score'], f['team_a_score']), ('A', f['team_a'], f['team_h'], f['team_a_score'], f['team_h_score'])):
                rows.append(dict(team=sn[tid], opp=sn[oid], ha=side, gf=gf, ga=ga, ko=f['kickoff_time'], gw=f['event'],
                                 res='W' if gf > ga else ('D' if gf == ga else 'L')))
        R = pd.DataFrame(rows).sort_values('ko')
        g = R.groupby('team')
        S = pd.DataFrame({'p': g.size(), 'w': g.res.apply(lambda x: (x == 'W').sum()), 'd': g.res.apply(lambda x: (x == 'D').sum()),
                          'l': g.res.apply(lambda x: (x == 'L').sum()), 'gf': g.gf.sum(), 'ga': g.ga.sum(), 'cs': g.ga.apply(lambda x: (x == 0).sum())})
        S['gd'] = S.gf - S.ga; S['pts'] = 3 * S.w + S.d
        S = S.sort_values(['pts', 'gd', 'gf'], ascending=False); S['rank'] = range(1, len(S) + 1)
        S['last5'] = g.apply(lambda d: d.tail(5)[['res', 'opp', 'ha', 'gf', 'ga', 'gw']].to_dict('records')).reindex(S.index)
        S['form5'] = g.apply(lambda d: (d.tail(5).res.map({'W': 3, 'D': 1, 'L': 0})).sum()).reindex(S.index)
        return S

    S = team_season()
    TF = P.groupby('team').agg(code=('team_code', 'first'), xgf=('own_xgf', 'first'), xga=('own_xga', 'first'), cs_next=('p_team_cs', 'first'), opp=(f'opp_{G0}', 'first'))
    S = S.join(TF, how='left')
    sec('Teams', f'this season so far, before GW{G0}')
    sort = st.segmented_control('Sort teams by', ['League table', 'Form (last 5)', 'Attack (xG for)', 'Defence (xG against)', f'GW{G0} clean sheet chance'],
                                default='League table', key='tf_sort') or 'League table'
    key = {'League table': ('rank', True), 'Form (last 5)': ('form5', False), 'Attack (xG for)': ('xgf', False), 'Defence (xG against)': ('xga', True),
           f'GW{G0} clean sheet chance': ('cs_next', False)}[sort]
    S = S.sort_values([key[0], 'rank'], ascending=[key[1], True])
    RC = {'W': ('#00ff87', '#1b0021'), 'D': ('#b9a7c2', '#1b0021'), 'L': ('#e90052', '#ffffff')}
    chip = lambda r: (f'<span title="GW{r["gw"]}: {r["opp"]} ({r["ha"]}) {r["gf"]}-{r["ga"]}" style="display:inline-block;width:22px;height:22px;line-height:22px;margin:1px;'
                      f'border-radius:5px;text-align:center;font-weight:800;font-size:.72rem;background:{RC[r["res"]][0]};color:{RC[r["res"]][1]}">{r["res"]}</span>')
    res_line = lambda r: (f'GW{r["gw"]} · <b style="color:{RC[r["res"]][0]}">{r["res"]}</b> <b>{r["gf"]}-{r["ga"]}</b> '
                          f'{"vs" if r["ha"] == "H" else "at"} {r["opp"]}')
    bar = lambda v, mx, good=True: (f'<div class="xpbar" style="min-width:80px"><i style="width:{max(4, min(100, 100 * v / mx)):.0f}%;'
                                    f'{"" if good else "background:linear-gradient(90deg,#ff1751,#e90052)"}"></i><span>{v:.2f}</span></div>')
    body = ''.join(
        f'<tr><td>{r["rank"]}</td><td style="white-space:nowrap"><img src="{ui.shirt(r.code)}" style="width:26px;vertical-align:middle;margin-right:8px"><b>{tm}</b></td>'
        f'<td>{r.w}-{r.d}-{r.l}</td><td style="white-space:nowrap">{r.gf}-{r.ga} ({r.gd:+d})</td><td><b>{r.pts}</b></td>'
        f'<td style="white-space:nowrap"><details class="res5"><summary>{"".join(chip(x) for x in r.last5)}<span class="res5-hint">▾</span></summary>'
        f'<div class="res5-list">{"<br>".join(res_line(x) for x in reversed(r.last5))}</div></details></td>'
        f'<td>{bar(r.xgf, 2.6)}</td><td>{bar(r.xga, 2.6, False)}</td><td>{r.cs}</td><td style="white-space:nowrap">{ui.esc(r.opp)}</td>'
        f'<td><b style="color:{ui.GREEN}">{100 * r.cs_next:.0f}%</b></td></tr>' for tm, r in S.iterrows())
    html(f'<div style="overflow-x:auto"><table class="fpl" style="min-width:940px"><tr><th>#</th><th>Team</th><th>W-D-L</th><th>Goals (GD)</th><th>Pts</th><th>Last 5</th>'
         f'<th>xG for</th><th>xG against</th><th>Clean sheets</th><th>GW{G0}</th><th>CS chance</th></tr>{body}</table></div>')
    html(f'<div class="note" style="margin-top:6px">{S.p.max()} games played. Last 5 runs oldest to newest: click the results to see the scores (newest first). xG for and against are per game over the last 10 games, '
         f'pulled a little towards the league average. CS chance is the chance of a clean sheet in GW{G0}. '
         'All of these feed the model, so a team in good form or with a tight defence already lifts its players\' projections.</div>')

    # ---- fixture ticker, easiest run first
    sec('Fixture ticker', 'sorted easiest first · official difficulty, 1 easy to 5 hard · H home, A away, + double')
    n = st.segmented_control('Gameweeks', ['Next 4', 'Next 6', 'Next 8'], default='Next 6', key='tk_n', label_visibility='collapsed') or 'Next 6'
    gs = ALL_GWS[:int(n.split()[1])]
    T = P.groupby('team').first()
    rows = []
    for tm, r in T.iterrows():
        cells = [(r[f'opp_{g}'].split(',')[0].replace(' (H)', ' H').replace(' (A)', ' A') + ('+' if ',' in r[f'opp_{g}'] else ''), r[f'fdr_{g}']) for g in gs]
        diff = np.mean([f if f else 5.5 for _, f in cells])
        rows.append((diff, f'<img src="{ui.shirt(r.team_code)}" style="width:22px;vertical-align:middle;margin-right:6px">{tm}', cells))
    rows.sort(key=lambda x: x[0])
    html(ui.ticker([(lab, cells) for _, lab, cells in rows], gs))

# ================================================================== REVIEW: predicted vs what happened
elif page == 'Review':
    @st.cache_data(ttl=3600, show_spinner='Loading past gameweeks...')
    def review_data():
        """For every finished gameweek: each player's predicted points and chances, and what actually happened.
        Weeks with a snapshot use exactly what the app showed before the deadline; GW3-10 use the back-test (a model that never saw those weeks)."""
        H = pd.read_csv(ROOT / 'data' / '2026-27' / 'history.csv', low_memory=False)
        # finished gameweeks whose results are downloaded: between FPL marking a week finished and the next full download it is not shown
        # (otherwise every player would show 0 points)
        done_gws = {e['id'] for e in boot['events'] if e['finished']} & set(H['round'].unique())
        act = H[H['round'].isin(done_gws)].groupby(['round', 'element']).agg(
            actual=('total_points', 'sum'), minutes=('minutes', 'sum'), goals=('goals_scored', 'sum'), assists=('assists', 'sum'),
            bonus=('bonus', 'sum'), cs=('clean_sheets', 'sum'), fixtures=('fixture', 'count')).reset_index().rename(columns={'round': 'gw'})
        out = {}
        anyp = lambda x: 1 - np.prod(1 - x.values)
        try:
            bt = pd.read_csv(ROOT / 'models' / 'test_predictions.csv'); bt = bt[bt.season == '2026-27']
            try: ev = pd.read_csv(ROOT / 'models' / 'event_test_predictions.csv'); bt = bt.merge(ev, on=['season', 'gw', 'element', 'fixture'], how='left')
            except FileNotFoundError: pass
            agg = dict(pred=('blend', 'sum'), form=('last5', 'mean'))
            for c in ('p_goal', 'p_assist', 'p_bonus', 'p_mins60'):
                if c in bt: agg[c] = (c, anyp)
            B = bt.groupby(['gw', 'element']).agg(**agg).reset_index().rename(columns={'p_mins60': 'p_60'})
            for g_, d_ in B.groupby('gw'): out[int(g_)] = ('back-test', d_.drop(columns='gw'))
        except FileNotFoundError: pass
        snap_dir = ROOT / 'data' / 'snapshots'
        for f in sorted(snap_dir.glob('gw*.parquet')) if snap_dir.exists() else []:
            g_ = int(f.stem[2:])
            if g_ not in done_gws: continue
            sn = pd.read_parquet(f).reset_index().rename(columns={'id': 'element', 'xp': 'pred'})
            out[g_] = ('snapshot', sn[['element', 'pred'] + [c for c in ('p_goal', 'p_assist', 'p_bonus', 'p_60', 'p_cs', 'p_dc') if c in sn]])
        res = {}
        for g_, (src, d_) in out.items():
            a_ = act[act.gw == g_].drop(columns='gw')
            m_ = d_.merge(a_, on='element', how='outer').fillna({'actual': 0, 'minutes': 0, 'goals': 0, 'assists': 0, 'bonus': 0, 'cs': 0, 'pred': 0})
            res[g_] = (src, m_[m_.element.isin(P.index)])
        return res

    @st.cache_data(ttl=3600, show_spinner=False)
    def my_picks(eid_, gw_):
        try: return fe.get(f'entry/{int(eid_)}/event/{gw_}/picks/')
        except Exception: return None

    LO_, HI_ = {1: 1, 2: 3, 3: 2, 4: 1}, {1: 1, 2: 5, 3: 5, 4: 3}

    def best_legal(ids, score):
        """Best legal XI from ids by score: the minimum per position first, then the best of the rest up to the maximums."""
        s_ = sorted([i for i in ids if i in P.index], key=lambda i: -score(i))
        xi = [i for p_ in (1, 2, 3, 4) for i in [j for j in s_ if P.pos[j] == p_][:LO_[p_]]]
        cnt = dict(LO_)
        for i in s_:
            if len(xi) == 11: break
            if i not in xi and cnt[P.pos[i]] < HI_[P.pos[i]]: xi.append(i); cnt[P.pos[i]] += 1
        return xi

    def autosub(xi, bench, mins):
        """FPL automatic substitutions: a starter with 0 minutes is replaced by the first bench player who played, keeping a legal formation (GK for GK)."""
        xi, used = list(xi), set()
        for k, i in enumerate(list(xi)):
            if mins(i) > 0: continue
            for b in bench:
                if b in used or mins(b) <= 0: continue
                if (P.pos[b] == 1) != (P.pos[i] == 1): continue
                trial = xi[:k] + [b] + xi[k + 1:]
                cnt = pd.Series([P.pos[j] for j in trial]).value_counts()
                if all(cnt.get(p_, 0) >= LO_[p_] for p_ in (1, 2, 3, 4)):
                    xi = trial; used.add(b); break
        return xi

    def week_teams(pk_, d_):
        """The team you played and the model's pick from the same 15, scored by FPL's rules (captain, chips, automatic subs).
        d_: that week's predictions and results indexed by player id. Returns two dicts with xi, bench, cap, vice, mult, pred, pts."""
        pv_ = lambda i: float(d_.pred.get(i, 0.0)); av = lambda i: float(d_.actual.get(i, 0.0)); mn = lambda i: float(d_.minutes.get(i, 0.0))
        pk_s = sorted([p_ for p_ in pk_['picks'] if p_['element'] in P.index], key=lambda x: x['position'])
        chip = pk_.get('active_chip'); cm = 3 if chip == '3xc' else 2; bb = chip == 'bboost'
        ids = [p_['element'] for p_ in pk_s]
        # yours: FPL's own multipliers already include the captain, the chip and the automatic subs
        y = dict(xi=[p_['element'] for p_ in pk_s if p_['position'] <= 11], bench=[p_['element'] for p_ in pk_s if p_['position'] > 11],
                 cap=next((p_['element'] for p_ in pk_s if p_['is_captain']), None), vice=next((p_['element'] for p_ in pk_s if p_['is_vice_captain']), None),
                 mult={p_['element']: p_['multiplier'] for p_ in pk_s})
        y['pred'] = sum(pv_(i) * (cm if i == y['cap'] else (1 if (i in y['xi'] or bb) else 0)) for i in ids)
        y['pts'] = pk_.get('entry_history', {}).get('points', sum(av(i) * y['mult'][i] for i in ids))
        # the model's pick: best XI by prediction, captain and vice by prediction, bench GK first then by prediction
        mx = best_legal(ids, pv_); mb = sorted([i for i in ids if i not in mx], key=lambda i: (P.pos[i] != 1, -pv_(i)))
        mo = sorted(mx, key=pv_, reverse=True); mcap, mvice = mo[0], mo[1]
        played = ids if bb else autosub(mx, mb, mn)
        cap_eff = mcap if mn(mcap) > 0 else (mvice if mn(mvice) > 0 else None)
        mult = {i: (1 if i in played else 0) for i in ids}
        if cap_eff is not None: mult[cap_eff] = cm
        m = dict(xi=mx, bench=mb, cap=mcap, vice=mvice, mult=mult, pred=sum(pv_(i) for i in (ids if bb else mx)) + (cm - 1) * pv_(mcap),
                 pts=sum(av(i) * mult[i] for i in ids), subs=[i for i in played if i not in mx], cap_eff=cap_eff)
        return y, m

    RV = review_data()
    if not RV:
        html('<div class="note">No finished gameweeks with predictions yet.</div>'); done()

    sec('Pick a gameweek', 'see what the model said and what happened')
    gl = sorted(RV, reverse=True)
    gsel = st.selectbox('Gameweek', gl, format_func=lambda g_: f'Gameweek {g_} · ' + ('what the app showed before the deadline' if RV[g_][0] == 'snapshot' else 'back-test'), key='rv_gw')
    src, D = RV[gsel]
    D = D.copy(); D['name'] = D.element.map(P.name); D['team'] = D.element.map(P.team); D['pos'] = D.element.map(P.pos); D['team_code'] = D.element.map(P.team_code)
    D['diff'] = (D.actual - D.pred).where((D.actual - D.pred).abs() >= 0.05, 0.0)
    html('<div class="note" style="margin:-4px 0 10px 0">' + (
        f'These are the exact predictions the app showed before the GW{gsel} deadline.' if src == 'snapshot' else
        f'Back-test: predictions from a model trained only on earlier seasons, so it had never seen GW{gsel}. From GW11 on, this tab shows exactly what the app showed before each deadline.') + '</div>')

    # ---- how the model did this week
    likely = D[D.pred >= 2.0]                                    # players the model expected to play: ranking among them is what matters
    rank_acc = likely[['pred', 'actual']].corr(method='spearman').iloc[0, 1] if len(likely) > 5 else np.nan
    top10 = D.nlargest(10, 'pred'); best10 = D.nlargest(10, 'actual')
    form_top = D[D.pred >= 2.0].nlargest(10, 'form').actual.mean() if 'form' in D and D.form.notna().any() else None
    cap = D.loc[D.pred.idxmax()]
    rmse = float(np.sqrt(((likely.pred - likely.actual) ** 2).mean())) if len(likely) else np.nan
    html(ui.cards([(f'{rank_acc:.2f}', 'ranking accuracy this week (1 = perfect order)', 'g'),
                   (f'{top10.actual.mean():.1f}', 'avg points of the model\'s top 10' + (f' (top 10 by form: {form_top:.1f})' if form_top is not None else ''), 'c'),
                   (f'{ui.esc(cap["name"])}: {cap.actual:.0f}', f'model\'s top pick scored (predicted {cap.pred:.1f})', ''),
                   (f'{rmse:.1f}', 'typical miss in points, likely starters', '')]))

    # ---- team of the week. With your team: the model's best XI from your 15 vs the team you played. Without: the model's XI vs the best in hindsight.
    def dream_xi(col, pool=None):
        """Best legal XI by col (from pool if given): the minimum per position first (1 GK, 3 DEF, 2 MID, 1 FWD), then the best of the rest up to the maximums."""
        s_ = D.dropna(subset=['pos'])
        if pool is not None: s_ = s_[s_.element.isin(pool)]
        s_ = s_.sort_values(col, ascending=False)
        lo, hi = {1: 1, 2: 3, 3: 2, 4: 1}, {1: 1, 2: 5, 3: 5, 4: 3}
        xi = [i for p_ in (1, 2, 3, 4) for i in s_[s_.pos == p_].element.head(lo[p_])]
        cnt = {p_: lo[p_] for p_ in lo}
        for i, p_ in zip(s_.element, s_.pos):
            if len(xi) == 11: break
            if i not in xi and cnt[p_] < hi[p_]: xi.append(i); cnt[p_] += 1
        return [int(i) for i in xi]
    am = D.set_index('element'); P_past = P.assign(chance=100)    # no injury flags on past weeks
    pv = lambda i: float(am.pred.get(i, 0.0)); av_ = lambda i: float(am.actual.get(i, 0.0))
    eid_ = str(st.session_state.get('eid', '') or '').strip()
    pk = my_picks(eid_, gsel) if team is not None and eid_.isdigit() else None
    picks = [p_ for p_ in (pk or {}).get('picks', []) if p_['element'] in P.index]
    if len(picks) == 15:
        y_, m_ = week_teams(pk, am)
        squad_ = [p_['element'] for p_ in picks]
        # each shirt: predicted and scored points as they counted, so a captain shows x2 (x3 with Triple Captain)
        def shirts(t):
            k_ = lambda i: t['mult'].get(i, 1) if t['mult'].get(i, 1) > 1 else 1
            return {i: pv(i) * k_(i) for i in squad_}, {i: f'scored {av_(i) * k_(i):.0f}' for i in squad_}
        mv_, mf_ = shirts(m_); yv_, yf_ = shirts(y_)
        cm_ = 3 if pk.get('active_chip') == '3xc' else 2
        sec('Team of the week', f'GW{gsel} · the model\'s choice from your squad vs the team you played · captain ×{cm_}')
        tw1, tw2 = st.columns(2)
        with tw1:
            html(ui.pitch(P_past, m_['xi'], m_['bench'], m_['cap'], m_['vice'], mv_, mf_, compact=True, mark_in=[i for i in m_['xi'] if i not in y_['xi']],
                          title=f"The model's pick <small>predicted {m_['pred']:.0f} · scored <b style='color:#00ff87'>{m_['pts']:.0f}</b></small>"))
        with tw2:
            html(ui.pitch(P_past, y_['xi'], y_['bench'], y_['cap'], y_['vice'], yv_, yf_, compact=True, mark_out=[i for i in y_['xi'] if i not in m_['xi']],
                          title=f"Your team <small>predicted {y_['pred']:.0f} · scored <b style='color:#00ff87'>{y_['pts']:.0f}</b></small>"))
        diff_ = [i for i in m_['xi'] if i not in y_['xi']]
        eh = pk.get('entry_history', {})
        chip_ = {'bboost': 'Bench Boost', '3xc': 'Triple Captain', 'freehit': 'Free Hit', 'wildcard': 'Wildcard', 'manager': 'Assistant Manager'}.get(pk.get('active_chip'), pk.get('active_chip'))
        same = not diff_ and m_['cap'] == y_['cap']
        html('<div class="note" style="margin-top:4px">Each shirt shows the predicted points, with what he actually scored underneath (the captain\'s are multiplied). '
             + (f'The model would have started <b style="color:#fff">{", ".join(ui.esc(P.name[i]) for i in diff_)}</b> (green) instead of your red-marked players' if diff_
                else 'The model would have picked the same XI as you')
             + (f' and captained <b style="color:#fff">{ui.esc(P.name[m_["cap"]])}</b>' if m_['cap'] != y_['cap'] else '') + '. '
             + ('' if same else f'Its pick would have scored <b style="color:#fff">{m_["pts"] - y_["pts"]:+.0f}</b> compared with yours. ')
             + (f'Automatic substitutions: {", ".join(ui.esc(P.name[i]) for i in m_["subs"])} would have come on for the model. ' if m_['subs'] and not same else '')
             + (f'You played {chip_} this week. ' if chip_ else '')
             + (f'Scores are before the −{eh["event_transfers_cost"]} hit.' if eh.get('event_transfers_cost') else '') + '</div>')
    else:
        mx, bx = dream_xi('pred'), dream_xi('actual')
        mcap = max(mx, key=pv); bcap = max(bx, key=av_)
        m_pts = sum(av_(i) for i in mx) + av_(mcap); b_pts = sum(av_(i) for i in bx) + av_(bcap)
        sec('Team of the week', f'GW{gsel} · any price, captain doubled')
        tw1, tw2 = st.columns(2)
        with tw1:
            html(ui.pitch(P_past, mx, [], mcap, None, {i: pv(i) for i in mx}, {i: f'scored {av_(i):.0f}' for i in mx}, compact=True,
                          title=f"The model's team <small>predicted {sum(pv(i) for i in mx) + pv(mcap):.0f} · scored <b style='color:#00ff87'>{m_pts:.0f}</b></small>"))
        with tw2:
            html(ui.pitch(P_past, bx, [], bcap, None, {i: av_(i) for i in bx}, {i: f'predicted {pv(i):.1f}' for i in bx}, compact=True,
                          mark_in=[i for i in bx if i in mx], title=f"The best team in hindsight <small>scored <b style='color:#00ff87'>{b_pts:.0f}</b></small>"))
        html(f'<div class="note" style="margin-top:4px">The model\'s team got <b style="color:#fff">{m_pts:.0f}</b> of the <b style="color:#fff">{b_pts:.0f}</b> points the perfect team scored '
             f'({100 * m_pts / max(b_pts, 1):.0f}%). Points under each shirt are predicted on the left and actual on the right. '
             'Enter your team ID to compare the model\'s pick from your squad with the team you played.</div>')
    # ---- surprises
    sec('Hits and misses', f'GW{gsel}')
    card_ = lambda r_, kind: (f'<div class="mv {kind}" style="margin:5px 0"><img src="{ui.shirt(r_.team_code, r_.pos == 1)}"><div><div class="bigname">{ui.esc(r_["name"])} '
                              f'<span class="note">{r_.team}</span></div><div class="m">predicted {r_.pred:.1f} · scored <b style="color:#fff">{r_.actual:.0f}</b> · '
                              f'{r_.goals:.0f} G · {r_.assists:.0f} A · {r_.bonus:.0f} bonus</div></div></div>')
    k1, k2, k3 = st.columns(3)
    with k1:
        html('<div class="ptitle">Called it <small>top 20 predictions that hauled</small></div>')
        x_ = D.nlargest(20, 'pred'); x_ = x_[x_.actual >= 8].nlargest(5, 'actual')
        html(''.join(card_(r_, 'in') for _, r_ in x_.iterrows()) or '<div class="note">None of the top 20 scored 8+ this week.</div>')
    with k2:
        html('<div class="ptitle">Missed <small>big scores the model did not expect</small></div>')
        x_ = D[(D.actual >= 8) & (D.pred < 3.5)].nlargest(5, 'actual')
        html(''.join(card_(r_, 'out') for _, r_ in x_.iterrows()) or '<div class="note">No big surprises.</div>')
    with k3:
        html('<div class="ptitle">Backed, but blanked <small>top 20 that scored 2 or less</small></div>')
        x_ = D.nlargest(20, 'pred'); x_ = x_[x_.actual <= 2].nsmallest(5, 'actual')
        html(''.join(card_(r_, 'out') for _, r_ in x_.iterrows()) or '<div class="note">Every top-20 pick returned.</div>')

    # ---- every player
    sec('Every player', 'predicted vs what happened')
    f1, f2, f3, f4 = st.columns([1.4, 1.6, 1.2, 1.2])
    pos_ = f1.pills('Position', ['GK', 'DEF', 'MID', 'FWD'], selection_mode='multi', default=['GK', 'DEF', 'MID', 'FWD'], key='rv_pos')
    tms = f2.multiselect('Teams', sorted(D.team.dropna().unique()), placeholder='All teams', key='rv_tm')
    sort_ = f3.selectbox('Sort by', ['Predicted', 'Scored', 'Beat prediction most', 'Fell short most'], key='rv_sort')
    played = f4.toggle('Only players who played', True, key='rv_played')
    T_ = D[D.pos.isin([k for k, v in POS.items() if v in (pos_ or POS.values())])]
    if tms: T_ = T_[T_.team.isin(tms)]
    if played: T_ = T_[T_.minutes > 0]
    T_ = {'Predicted': T_.sort_values('pred', ascending=False), 'Scored': T_.sort_values('actual', ascending=False),
          'Beat prediction most': T_.sort_values('diff', ascending=False), 'Fell short most': T_.sort_values('diff')}[sort_].head(10)
    pct = lambda r_, c: f'{100 * r_[c]:.0f}%' if c in r_ and pd.notna(r_[c]) else '-'
    body = ''.join(
        f'<tr><td><div class="who"><img src="{ui.shirt(r_.team_code, r_.pos == 1)}"><div><b>{ui.esc(r_["name"])}</b><span class="pos" style="background:{ui.POS_COL[r_.pos]}">{POS[r_.pos]}</span>'
        f'<div class="t">{r_.team}</div></div></div></td><td>{r_.pred:.1f}</td><td><b>{r_.actual:.0f}</b></td>'
        f'<td><b style="color:{ui.GREEN if r_["diff"] > 0.5 else (ui.PINK if r_["diff"] < -0.5 else "#fff")}">{r_["diff"]:+.1f}</b></td><td>{r_.minutes:.0f}</td>'
        f'<td>{pct(r_, "p_goal")} → <b>{r_.goals:.0f}</b></td><td>{pct(r_, "p_assist")} → <b>{r_.assists:.0f}</b></td><td>{pct(r_, "p_bonus")} → <b>{r_.bonus:.0f}</b></td></tr>'
        for _, r_ in T_.iterrows())
    html('<table class="fpl"><tr><th>Player</th><th>Predicted</th><th>Scored</th><th>Difference</th><th>Minutes</th><th>Goal chance → goals</th>'
         f'<th>Assist chance → assists</th><th>Bonus chance → bonus</th></tr>{body}</table>')

    # ---- your season: what you scored each week vs what the model expected from the team you picked
    if team is not None and eid_.isdigit():
        ys = []
        for g_ in sorted(RV):
            pk_ = my_picks(eid_, g_)
            if not pk_ or len([p_ for p_ in pk_['picks'] if p_['element'] in P.index]) < 15: continue
            y_, m_ = week_teams(pk_, RV[g_][1].set_index('element'))
            hit_ = pk_.get('entry_history', {}).get('event_transfers_cost', 0) or 0    # the same squad, so the model's pick pays the same hits
            ys.append(dict(gw=g_, exp=y_['pred'] - hit_, pts=y_['pts'] - hit_, model=m_['pts'] - hit_, hit=hit_))
        if ys:
            ys = pd.DataFrame(ys); gap = ys.model.sum() - ys.pts.sum()
            sec('Your season', f'GW{ys.gw.min()} to GW{ys.gw.max()} · after hits' + (f' (−{ys.hit.sum():.0f} in total)' if ys.hit.sum() else ''))
            html(ui.cards([(f'{ys.pts.sum():.0f}', 'you scored with the teams you picked', 'g'),
                           (f'{ys.model.sum():.0f}', "the model's pick from your squad each week would have scored", 'c'),
                           (f'{gap:+.0f}', 'the model\'s picks vs yours' + (f' · better in {(ys.model > ys.pts).sum()} of {len(ys)} weeks' if len(ys) else ''), '')]))
            fy = go.Figure()
            fy.add_trace(go.Bar(x=[f'GW{g_}' for g_ in ys.gw], y=ys.pts, name='you scored', marker_color=ui.GREEN))
            fy.add_trace(go.Bar(x=[f'GW{g_}' for g_ in ys.gw], y=ys.model, name="model's pick would have scored", marker_color='#7a3f86'))
            fy.add_trace(go.Scatter(x=[f'GW{g_}' for g_ in ys.gw], y=ys.exp, name='model expected from your team', mode='lines+markers', line=dict(color='#04f5ff', width=3)))
            fy.update_layout(barmode='group')
            fy = ui.plot_style(fy, 300); fy.update_yaxes(title='points')
            st.plotly_chart(fy, use_container_width=True, config={'displayModeBar': False})

# ================================================================== HOW IT WORKS
else:
    sc = pd.read_csv(ROOT / 'models' / 'test_scores.csv'); s = sc[sc.group == 'starters']
    nm = {'last5': 'Average of last 5 games', 'gbm': 'Gradient boosting', 'mlp': 'PyTorch network', 'blend': 'Blend (used here)'}
    html(ui.cards([(f'{s[s.predictor == "blend"]["rank corr (per GW)"].iloc[0]:.2f}', 'ranking accuracy of our model (regular starters)', 'g'),
                   (f'{s[s.predictor == "last5"]["rank corr (per GW)"].iloc[0]:.2f}', 'ranking accuracy of recent form', ''),
                   ('5 seasons', 'of game-by-game history, 2022-23 to now', 'c'), ('< 1 sec', 'to search every legal squad', '')]))
    st.markdown("""
#### What it predicts
For every player and every fixture, their **expected FPL points**. A double gameweek adds both fixtures and a blank gives zero.
Your projected gameweek score is the best legal XI from your 15, with the captain doubled.

#### The model
Two models learn from every player's game-by-game history: a **PyTorch neural network** (an ensemble of 5) and a gradient-boosting model, averaged.
They only see things known before kick-off:
- **The player:** points, minutes, expected goals and assists, bonus, defensive actions and saves over his last 3, 5, 10 and 20 games; per-90 rates; how often he reached the defensive-contribution threshold and got bonus in his last 5 and 10 games; defensive actions and bonus-system points per 90; how often he starts; last season's level; price.
- **His team and the opponent:** results and goal difference over the last 5, goals and expected goals for and against over the last 8 to 10, and a clean-sheet chance from those.
- **The match:** home or away, how he has done against this opponent before, and his home or away record, both compared with his usual level.
- **The season:** how many points each position is scoring league-wide right now, so the model adjusts when the rules change.

Flagged players are scaled by their chance of playing.

#### Chances of a goal, an assist, a clean sheet, defensive contribution and bonus
Separate models give each player's chance of scoring, assisting, getting bonus points, reaching the defensive-contribution threshold
(10 clearances, blocks, interceptions and tackles for a defender; 12 including recoveries for a midfielder or forward) and playing 60+ minutes.
Defensive contributions were only recorded from 2025-26, so that model learns from those two seasons. Expected bonus is the chance of any bonus
times how many (1 to 3) he usually gets when he does. The clean-sheet chance is worked out for the team
(from both sides' expected goals, recent form and the league's current clean-sheet rate), then multiplied by his chance of playing 60 minutes.
In a double gameweek it is the chance of at least one.

#### Transfers
A linear-programming optimiser checks every legal squad you can reach. It follows the FPL rules: 2 GK, 5 DEF, 5 MID, 3 FWD, at most 3 per club,
valid formations, selling prices (you keep half of any rise), free transfers and a 4-point hit for each extra one.

#### How good is it?
Tested on this season from Gameweek 3, which the models never saw:""")
    html('<table class="fpl"><tr><th>Method</th><th>Typical error</th><th>Average error</th><th>Ranking accuracy</th></tr>' +
         ''.join(f'<tr><td><b>{nm[r.predictor]}</b></td><td>{r.RMSE:.2f}</td><td>{r.MAE:.2f}</td><td>{r._4:.3f}</td></tr>' for r in s.itertuples()) + '</table>')
    ev = pd.read_csv(ROOT / 'models' / 'event_scores.csv')
    evn = {'goal': 'Scores', 'assist': 'Assists', 'cs': 'Clean sheets', 'mins60': 'Plays 60+ minutes', 'bonus': 'Gets bonus points',
           'dc': 'Defensive contribution (DEF, MID, FWD; 2025-26 on)'}
    html('<div class="sec" style="font-size:1.05rem">Chances: predicted vs what happened</div><table class="fpl"><tr><th>Event</th><th>Predicted rate</th><th>Actual rate</th><th>Top 200 predicted</th><th>Top 200 actual</th><th>Better than guessing the average</th></tr>' +
         ''.join(f'<tr><td><b>{evn[r.event]}</b></td><td>{100 * r._3:.1f}%</td><td>{100 * r._2:.1f}%</td><td>{100 * r._7:.1f}%</td><td>{100 * r._8:.1f}%</td><td>{r._4:.0f}%</td></tr>' for r in ev.itertuples()) + '</table>')
    html('<div class="note" style="margin-top:10px">Regular starters only. FPL points are very random: one goal changes a score by 4 to 10 points, so a typical error of about 3 is normal. '
         'The model ranks players better than recent form, but nobody can tell you who will score this weekend. Projections further ahead hold each player\'s form at today\'s level, so they are rougher.</div>')

html(ui.CREDIT)
