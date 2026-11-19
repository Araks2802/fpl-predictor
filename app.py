"""FPL Points Predictor: your projected points for the coming gameweek, the best XI and captain, and transfer suggestions.

Run locally:  streamlit run app.py
The app only reads data/predictions.parquet, data/meta.json and data/2026-27/bootstrap.json (made by make_predictions.py).
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, plotly.graph_objects as go, requests, streamlit as st
import fpl_engine as fe, ui

ROOT = Path(__file__).parent
st.set_page_config(page_title='FPL Points Predictor', page_icon='⚽', layout='wide', initial_sidebar_state='expanded')
st.markdown(ui.CSS, unsafe_allow_html=True)
POS = ui.POS


@st.cache_data(ttl=3600)
def load():
    P = pd.read_parquet(ROOT / 'data' / 'predictions.parquet'); meta = json.load(open(ROOT / 'data' / 'meta.json'))
    boot = json.load(open(ROOT / 'data' / '2026-27' / 'bootstrap.json'))
    gws = sorted(int(c[3:]) for c in P.columns if c.startswith('xp_'))
    return P, meta, boot, gws


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


P, meta, boot, ALL_GWS = load(); G0 = ALL_GWS[0]; P = P.copy()
with st.sidebar.expander('Model settings'):
    BOOST = st.slider('Form boost', 0, 50, 0, 5, format='%d%%', key='boost',
                      help='Mixes FPL form (points per match over the last 30 days) into every projection. Testing on unseen gameweeks found 0% predicts best: hot streaks mostly fade.') / 100
if BOOST:
    c = (P.chance.astype(float).fillna(100) / 100).where(P.status != 'u', 0.0)
    for k, g in enumerate(ALL_GWS):
        nfix = P[f'opp_{g}'].map(lambda o: 0 if o == '-' else o.count(',') + 1)
        P[f'xp_{g}'] = P[f'xp_{g}'] + BOOST * (P.form * nfix * (1 - (1 - c) * 0.5 ** k) - P[f'xp_{g}'])
deadline = datetime.fromisoformat(meta['deadline'].replace('Z', '+00:00'))
left = deadline - datetime.now(timezone.utc)
countdown = f'{left.days}d {left.seconds // 3600}h' if left.total_seconds() > 0 else 'deadline passed'
HORIZONS = {'Next GW': 1, 'Next 3': 3, 'Next 6': 6, 'Season': len(ALL_GWS)}   # Season = all remaining gameweeks
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
             f'<div class="l" style="font-size:.88rem">You have £{team_["bank"] / 10:.1f}m in the bank. FPL would not allow this transfer, so it cannot be confirmed. Pick a cheaper player.</div></div>')
    else:
        html(f'<div class="note" style="margin:6px 0">Bank afterwards: <b style="color:#00ff87">£{bk / 10:.1f}m</b></div>')
    if hits:
        html(f'<div class="card" style="padding:12px 14px;border-color:#e90052;margin:8px 0"><div class="v" style="color:#e90052;font-size:1.2rem">−{fe.HIT * hits} points</div>'
             f'<div class="l" style="font-size:.88rem">You have {ft_left} free transfer{"s" if ft_left != 1 else ""} left, so {hits} of these cost{"s" if hits == 1 else ""} a 4-point hit. '
             f'It comes off your GW{gw} score everywhere in the app.</div></div>')
    else:
        html(f'<div class="note" style="margin:4px 0 8px 0">Uses {free_used} of your {ft_left} free transfer{"s" if ft_left != 1 else ""}. No hit.</div>')
    html('<div class="note" style="margin:6px 0 10px 0">⚠️ <b style="color:#fff">This cannot be undone here</b>, just like a confirmed transfer in FPL. '
         'Not sure yet? Try it in the <b>Planner</b> first: you can change your mind there as often as you like.</div>')
    a, b = st.columns(2)
    if a.button('Yes, confirm', type='primary', use_container_width=True, key=f'{key}_yes', disabled=bk < 0):
        set_made(made_now + list(new))
        if after: after()
        st.rerun()
    return b


@st.dialog('Remove a confirmed transfer?')
def remove_made_dialog(k, made_now):
    o, i = made_now[k]
    html(f'<div class="note">Remove <b style="color:#e90052">{ui.esc(P.name[o])}</b> → <b style="color:#00ff87">{ui.esc(P.name[i])}</b>?</div>'
         '<div class="note" style="margin:8px 0">Only do this if you added it <b style="color:#fff">by mistake</b>. If you really made it in the FPL app, keep it: '
         'FPL will not undo it, and removing it here would make the app plan from the wrong team.</div>')
    a, b = st.columns(2)
    if a.button('Yes, remove it', type='primary', use_container_width=True, key='rm_yes'):
        set_made(made_now[:k] + made_now[k + 1:]); st.rerun()
    if b.button('Keep it', use_container_width=True, key='rm_no'): st.rerun()


@st.dialog('Confirm your transfers')
def confirm_dialog(new, made_now, team_, gw, after=None):
    b = confirm_block(new, made_now, team_, gw, after)
    if b.button('Cancel', use_container_width=True, key='cf_no'): st.rerun()


@st.dialog('Add a transfer you have made', width='large')
def made_dialog(team_, made_now, gw):
    """Tap who you sold, then pick who you bought, then confirm. All in one pop-up."""
    S = st.session_state
    sq = team_['squad'].set_index('id'); ids_ = [int(i) for i in sq.index]
    o = S.get('md_out'); i_ = S.get('md_in')
    if o is not None and o not in ids_: o = S['md_out'] = None
    # step 1: who did you sell? tap him on your team's pitch
    if o is None:
        html('<div class="note" style="margin-bottom:6px"><b style="color:#fff">Step 1 of 3.</b> Tap the name of the player you sold.</div>')
        xi_, cap_, vice_, bench_ = lineup(tuple(sorted(ids_)), gw, BOOST)
        hit_ = click_pitch(xi_, bench_, cap_, vice_, gw, tag='md', order=ids_, tip='I sold {name}')
        if hit_ is not None:
            S['md_out'] = int(hit_); S['md_in'] = None; st.rerun(scope='fragment')
        return
    money = team_['bank'] + int(sq.sell[o])
    top = st.columns([5, 1.2])
    top[0].markdown(f'<div class="mv out" style="padding:6px 10px"><img src="{ui.shirt(P.team_code[o], P.pos[o] == 1)}"><div><div class="bigname">Sold: {ui.esc(P.name[o])}</div>'
                    f'<div class="m">{P.team[o]} · {POS[P.pos[o]]} · sells for £{sq.sell[o] / 10:.1f}m · money available £{money / 10:.1f}m</div></div></div>', unsafe_allow_html=True)
    if top[1].button('Change', key='md_change_out', use_container_width=True):
        S['md_out'] = None; S['md_in'] = None; st.rerun(scope='fragment')
    # step 2: who did you buy?
    if i_ is None:
        html('<div class="note" style="margin:8px 0 4px 0"><b style="color:#fff">Step 2 of 3.</b> Pick the player you bought.</div>')
        cand = P[(P.pos == P.pos[o]) & ~P.index.isin(ids_)].copy()
        f1, f2, f3 = st.columns([2, 1.6, 1.6])
        teams = f1.multiselect('Teams', sorted(cand.team.unique()), placeholder='All teams', key='md_tm')
        mx = f2.slider('Max price (£m)', 3.5, 16.0, 16.0, 0.5, key='md_mx')
        q = f3.text_input('Search', placeholder='Name', key='md_q')
        if teams: cand = cand[cand.team.isin(teams)]
        cand = cand[cand.price <= mx + 1e-9]
        if q: cand = cand[cand.name.str.contains(q, case=False)]
        cand = cand.sort_values(['selected', 'price'], ascending=False)        # most-owned first: the players people usually buy
        n = S.get('md_n', 12)
        for i, r_ in cand.head(n).iterrows():
            a, b = st.columns([6, 1.3])
            short = round(r_.price * 10) - money
            a.markdown(f'<div class="mv {"out" if short > 0 else "in"}" style="padding:5px 10px"><img src="{ui.shirt(r_.team_code, r_.pos == 1)}"><div><div class="bigname">{ui.esc(r_["name"])} '
                       f'<span class="note">{r_.team}</span></div><div class="m">£{r_.price:.1f}m · owned {r_.selected:.0f}%'
                       + (f' · <b style="color:#e90052">over budget by £{short / 10:.1f}m</b>' if short > 0 else '') + '</div></div></div>', unsafe_allow_html=True)
            if b.button('Transfer' if short <= 0 else 'Too costly', key=f'md_i_{i}', disabled=short > 0, type='primary' if short <= 0 else 'secondary', use_container_width=True):
                S['md_in'] = int(i); st.rerun(scope='fragment')
        if cand.empty: st.info('No players match.')
        if len(cand) > n and st.button('Show more', key='md_more'): S['md_n'] = n + 12; st.rerun(scope='fragment')
        return
    # step 3: confirm
    html('<div class="note" style="margin:8px 0 4px 0"><b style="color:#fff">Step 3 of 3.</b> Check and confirm.</div>')
    b = confirm_block([(int(o), int(i_))], made_now, team_, gw, after=lambda: [S.pop(k, None) for k in ('md_out', 'md_in', 'md_n')], key='mdcf')
    if b.button('Back', use_container_width=True, key='md_back'):
        S['md_in'] = None; st.rerun(scope='fragment')


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


def remember_save(key, qp):
    v = str(st.session_state.get(key, '')).strip(); st.session_state[f'_{key}'] = v
    if v.isdigit(): st.query_params[qp] = v
    elif qp in st.query_params: del st.query_params[qp]


remember('eid', 'team'); remember('lid', 'league')

with st.sidebar:
    if BOOST: st.caption(f'Form boost on: {BOOST:.0%}')
    html('<div style="font-family:Outfit;font-size:1.5rem;font-weight:800;margin:4px 0 2px 0">FPL <span style="color:#00ff87">Predictor</span></div>'
         f'<div class="note">Gameweek {G0} deadline in <b style="color:#00ff87">{countdown}</b></div>')
    st.write('')
    mode = st.segmented_control('Your team', ['Team ID', 'Pick players'], default='Team ID', key='mode')
    team = None
    if mode != 'Pick players':
        eid = st.text_input('FPL team ID', placeholder='e.g. 1234567', key='eid', help='On the FPL website open Points. The number after /entry/ in the address is your ID.',
                            on_change=remember_save, args=('eid', 'team'))
        if eid.strip().isdigit():
            try: team = load_team(int(eid), boot)
            except requests.HTTPError: st.error('No team found with that ID.')
            except Exception as e: st.error(f'Could not load the team ({e}).')
    else:
        picked = []
        for p, n in fe.SQUAD_NEED.items():
            opts = P[P.pos == p].sort_values('price', ascending=False).index.tolist()
            picked += st.multiselect(f'{POS[p]} · pick {n}', opts, format_func=LAB.get, max_selections=n, key=f'pk{p}')
        bank_in = st.number_input('Bank (£m)', 0.0, 100.0, 0.0, 0.1)
        over = P.loc[picked].team.value_counts(); over = over[over > 3]
        if len(over): st.error(f'At most 3 players per club: you have {over.iloc[0]} from {over.index[0]}.')
        if len(picked) == 15 and not len(over):
            sq = pd.DataFrame({'id': picked}); sq['now'] = (P.loc[picked, 'price'].values * 10).round().astype(int); sq['sell'] = sq['now']; sq['buy'] = sq['now']
            sq['pos'] = P.loc[picked, 'pos'].values
            team = dict(name='My team', manager='', squad=sq, bank=int(round(bank_in * 10)), free_transfers=1,
                        chips_left=['Wildcard', 'Free Hit', 'Bench Boost', 'Triple Captain'], chips_used=[], points=None, rank=None, squad_gw=None)
        else: st.caption(f'{len(picked)} of 15 picked')
    if team is not None:
        # ---- transfers already made in the FPL app for the coming gameweek (FPL keeps them private until the deadline, so you tell us)
        tid = str(st.session_state.get('eid', 'manual'))
        if st.session_state.get('_made_for') != tid:                     # a different team: load ITS confirmed transfers (never another team's)
            st.session_state['_made_for'] = tid
            st.session_state['made'] = load_made(tid)
            if st.session_state['made']: st.query_params['made'] = f'{tid}~' + ','.join(f'{o}-{i}' for o, i in st.session_state['made'])
            elif 'made' in st.query_params: del st.query_params['made']
        made = st.session_state.setdefault('made', [])
        team, made_ok = apply_made(team, made)
        if len(made_ok) < len(made): st.session_state['made'] = made = made_ok
        with st.expander(f'Transfers already made for GW{G0}' + (f' ({len(made)})' if made else ''), expanded=bool(made)):
            st.caption('Made a transfer in the FPL app? FPL keeps it private until the deadline, so add it here and every tab will use your new team.')
            for k, (o_, i_) in enumerate(made):
                a_, b_ = st.columns([5, 1.1])
                a_.markdown(f'<div class="note">🔒 <b style="color:#e90052">{ui.esc(P.name[o_])}</b> → <b style="color:#00ff87">{ui.esc(P.name[i_])}</b></div>', unsafe_allow_html=True)
                if b_.button('✕', key=f'made_rm_{k}', help='Entered by mistake? Remove it'): remove_made_dialog(k, made)
            if team.get('made_hits'):
                st.markdown(f'<div class="note" style="color:#e90052"><b>−{fe.HIT * team["made_hits"]} hit</b> taken off your GW{G0} score.</div>', unsafe_allow_html=True)
            if team['free_transfers'] == 0:
                st.markdown('<div class="note" style="color:#e90052">No free transfers left: another one costs −4.</div>', unsafe_allow_html=True)
            if st.button('➕ Add a transfer I\'ve made', use_container_width=True, key='md_open'):
                for k in ('md_out', 'md_in', 'md_n'): st.session_state.pop(k, None)
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

# ------------------------------------------------------------------ header and navigation
def _ago(t):
    m = (datetime.now(timezone.utc) - datetime.fromisoformat(t)).total_seconds() / 60
    return f'{m:.0f} min ago' if m < 90 else (f'{m / 60:.0f} h ago' if m < 48 * 60 else f'{m / 1440:.0f} days ago')
updated_pill = (f'<span class="pill">Updated {_ago(meta["updated_at"])}' + (' · late team news' if meta.get('update_type', '').startswith('late') else '') + '</span>') if meta.get('updated_at') else ''
sub = (f"{team['manager']} · {team['points']} pts · rank {team['rank']:,}" if team and team.get('rank') else 'Projected points, best XI, captain and transfers for the coming gameweek')
html(f'<div class="hero"><div class="hero-title">{ui.esc(team["name"]) if team else "Your FPL team,"} <span>{"GW" + str(G0) if team else "projected."}</span></div>'
     f'<div class="hero-sub">{ui.esc(sub)}</div>'
     f'<span class="pill g">GW{G0} · {deadline:%a %d %b %H:%M} UTC</span><span class="pill">Deadline in {countdown}</span>{updated_pill}</div>')
page = st.segmented_control('Page', ['My team', 'Transfers', 'Planner', 'Captain', 'Chips', 'League', 'Compare', 'Players', 'Fixtures', 'How it works'], default='My team', label_visibility='collapsed', key='nav') or 'My team'


def done():
    """End the page: the credit footer, then stop. Used instead of st.stop() so the footer shows on every page."""
    html(ui.CREDIT); st.stop()


def need_team():
    html('<div class="card" style="padding:22px"><div class="v" style="font-size:1.2rem">Add your team to start</div>'
         '<div class="l" style="font-size:.9rem;margin-top:6px">Enter your FPL team ID in the sidebar, or pick your 15 players. '
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


# ================================================================== MY TEAM
if page == 'My team':
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
        html(ui.ticker([(ui.esc(P.name[i]), [(P.loc[i, f'opp_{g}'].split(',')[0].replace(' (H)', ' H').replace(' (A)', ' a') + ('+' if ',' in P.loc[i, f'opp_{g}'] else ''), P.loc[i, f'fdr_{g}']) for g in gs]) for i in order], gs))
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
        if chip == 'None' and st.button(f'✅ I\'ve made these in the FPL app…', key='made_from_transfers', type='primary',
                                        help=f'Saves them as your GW{g} team, so every tab uses it'):
            confirm_dialog([(int(o), int(i)) for o, i in pairs(dict(out=w0['out'], inn=w0['inn']))], st.session_state.get('made', []), team, g)
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
    default = (sorted([int(i) for i in team['squad'].id], key=lambda i: -P.loc[i, f'xp_{G0}'])[:2] if team else list(P[P.status != 'u'].nlargest(2, f'xp_{G0}').index))
    c1, c2 = st.columns([3, 1])
    cmp_ = c1.multiselect('Players to compare (up to 4)', P[P.status != 'u'].sort_values(f'h3', ascending=False).index.tolist(), default=default,
                          format_func=LAB.get, max_selections=4, key='cmp')
    gsel = c2.segmented_control('Gameweek', [f'GW{g}' for g in G2[:3]], default=f'GW{G0}', key='cmp_gw') or f'GW{G0}'
    g = int(gsel[2:])
    if not cmp_:
        st.info('Pick two to four players to compare.')
    else:
        R = P.loc[cmp_]
        # header cards with shirts
        heads = ''.join(f'<td style="text-align:center;padding:10px"><img src="{ui.shirt(r.team_code, r.pos == 1)}" style="width:70px"><div class="bigname" style="font-family:Outfit;font-weight:800;font-size:1.15rem">'
                        f'{ui.esc(r["name"])}</div><div class="note">{r.team} · {POS[r.pos]} · £{r.price:.1f}m</div></td>' for _, r in R.iterrows())
        pct = lambda v: f'{100 * v:.0f}%'
        rows = [  # (label, values, format, higher is better?)
            (f'Projected GW{g}', R[f'xp_{g}'], lambda v: f'{v:.1f}', True), ('Projected next 3 GWs', R.h3, lambda v: f'{v:.1f}', True),
            ('Projected rest of season', R[f'h{len(ALL_GWS)}'], lambda v: f'{v:.0f}', True),
            (f'Chance to score (GW{g})', R[f'p_goal_{g}'], pct, True), (f'Chance of an assist (GW{g})', R[f'p_assist_{g}'], pct, True),
            (f'Clean sheet chance (GW{g})', R[f'p_cs_{g}'].where(R.pos != 4), pct, True),
            (f'Defensive contribution chance (GW{g})', R[f'p_dc_{g}'].where(R.pos != 1), pct, True),
            (f'Bonus chance (GW{g})', R[f'p_bonus_{g}'], pct, True), (f'Expected bonus points (GW{g})', R[f'xb_{g}'], lambda v: f'{v:.2f}', True),
            (f'Plays 60+ mins (GW{g})', R[f'p_60_{g}'], pct, True),
            ('Hit the defensive threshold (last 10)', R.dc_hit_r10.where(R.pos != 1), pct, True), ('Defensive actions per 90', R.dc_p90_r10.where(R.pos != 1), lambda v: f'{v:.1f}', True),
            ('Got bonus (last 10 games played)', R.bonus_rate_r10, pct, True),
            (f'Opponent (GW{g})', R[f'opp_{g}'], str, None), ('Fixture difficulty', R[f'fdr_{g}'], lambda v: f'{v:.0f} / 5', False),
            ('Points so far', R.total_points, lambda v: f'{v:.0f}', True), ('FPL form (pts / match, 30 days)', R.form, lambda v: f'{v:.1f}', True),
            ('Points per £m (next 3)', R.h3 / R.price, lambda v: f'{v:.2f}', True), ('Owned by', R.selected / 100, pct, None),
            ('Team form (pts / game, last 5)', R.own_form5, lambda v: f'{v:.2f}', True), ('Team xG for / game', R.own_xgf, lambda v: f'{v:.2f}', True),
            ('Team xG against / game', R.own_xga, lambda v: f'{v:.2f}', False),
            ('vs this opponent (pts vs his usual)', R.h2h_adj, lambda v: f'{v:+.1f}', True), ('Home / away effect (pts vs usual)', R.venue_adj, lambda v: f'{v:+.1f}', True)]
        body = ''
        for lab, vals, fmt, hib in rows:
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

    gw1_moves = [m for m in moves if m['gw'] == PG[0]]
    if gw1_moves and st.button(f'✅ I\'ve made my GW{PG[0]} changes in the FPL app ({len(gw1_moves)})…', key='made_from_planner', type='primary',
                               help=f'Saves them as your GW{PG[0]} team on every tab. Your later-week plans stay.'):
        later = [m for m in moves if m['gw'] != PG[0]]
        confirm_dialog([(int(m['out']), int(m['inn'])) for m in gw1_moves], st.session_state.get('made', []), team, PG[0],
                       after=lambda: st.session_state.__setitem__('pl_moves', later))

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
    lid = c1.text_input('Mini-league ID', key='lid', placeholder='e.g. 123456', on_change=remember_save, args=('lid', 'league'),
                        help='Open the league on the FPL website. The number after /leagues/ in the address is its ID (not the invite code).')
    if lid.strip(): st.session_state['_lid'] = lid.strip()        # keep it for when you come back from another tab
    if lid.strip().isdigit() and st.query_params.get('league') != lid.strip(): st.query_params['league'] = lid.strip()
    mx = c2.select_slider('Managers to analyse', [10, 20, 30, 50], 30, key='lmax', help='The top of the table. More managers take longer to load.')
    if not lid.strip().isdigit():
        html('<div class="card" style="padding:18px"><div class="v" style="font-size:1.15rem">Enter your mini-league ID</div><div class="l" style="font-size:.9rem;margin-top:6px">'
             'See the table, how many rivals own each player, the players that threaten you, the differentials that gain on them, rivals\' chips, and a transfer plan '
             'aimed at climbing your league. Add your own team ID in the sidebar too, so the app knows which manager is you.</div></div>'); done()

    @st.cache_data(ttl=1800, show_spinner='Loading the league and every rival\'s team...')
    def get_league(lid_, mx_):
        return fe.load_league(int(lid_), boot, mx_)
    try: L = get_league(lid.strip(), mx)
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
    hz = f2.segmented_control('Projected over', list(HORIZONS), default='Next GW', key='hz_pl') or 'Next GW'
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
    html(ui.player_table(P, D.index[:n], ALL_GWS, col, max_val=D[col].max() if len(D) else 1))
    html('<div class="note" style="margin-top:6px">Want two players side by side? Open the <b>Compare</b> tab.</div>')

# ================================================================== FIXTURES
elif page == 'Fixtures':
    n = st.segmented_control('Gameweeks', ['Next 4', 'Next 6', 'Next 8'], default='Next 6', label_visibility='collapsed') or 'Next 6'
    gs = ALL_GWS[:int(n.split()[1])]
    T = P.groupby('team').first()
    rows = []
    for tm, r in T.iterrows():
        cells = [(r[f'opp_{g}'].split(',')[0].replace(' (H)', ' H').replace(' (A)', ' a') + ('+' if ',' in r[f'opp_{g}'] else ''), r[f'fdr_{g}']) for g in gs]
        diff = np.mean([f if f else 5.5 for _, f in cells])
        rows.append((diff, f'<img src="{ui.shirt(r.team_code)}" style="width:22px;vertical-align:middle;margin-right:6px">{tm}', cells))
    rows.sort(key=lambda x: x[0])
    sec('Fixture ticker', 'sorted easiest first · official difficulty, 1 easy to 5 hard · H home, a away, + double')
    html(ui.ticker([(lab, cells) for _, lab, cells in rows], gs))
    sec('Team form', 'what the model knows about each club before GW%d' % G0)
    TF = P.groupby('team').agg(code=('team_code', 'first'), form=('own_form5', 'first'), gd=('own_gd5', 'first'), xgf=('own_xgf', 'first'),
                               xga=('own_xga', 'first'), cs=('p_team_cs', 'first'), opp=(f'opp_{G0}', 'first'))
    sort = st.segmented_control('Sort teams by', ['Form', 'Attack (xG for)', 'Defence (xG against)', f'GW{G0} clean sheet chance'], default='Form', key='tf_sort') or 'Form'
    key = {'Form': ('form', False), 'Attack (xG for)': ('xgf', False), 'Defence (xG against)': ('xga', True), f'GW{G0} clean sheet chance': ('cs', False)}[sort]
    TF = TF.sort_values(key[0], ascending=key[1])
    bar = lambda v, mx, good=True: f'<div class="xpbar" style="min-width:90px"><i style="width:{max(4, min(100, 100 * v / mx)):.0f}%;{"" if good else "background:linear-gradient(90deg,#ff1751,#e90052)"}"></i><span>{v:.2f}</span></div>'
    body = ''.join(f'<tr><td><div class="who"><img src="{ui.shirt(r.code)}"><b>{tm}</b></div></td><td>{bar(r.form, 3)}</td><td>{r.gd:+.2f}</td><td>{bar(r.xgf, 2.6)}</td>'
                   f'<td>{bar(r.xga, 2.6, False)}</td><td>{ui.esc(r.opp)}</td><td><b style="color:{ui.GREEN}">{100 * r.cs:.0f}%</b></td></tr>' for tm, r in TF.iterrows())
    html('<table class="fpl"><tr><th>Team</th><th>Points / game (last 5)</th><th>Goal diff / game (last 5)</th><th>xG for / game</th><th>xG against / game</th>'
         f'<th>GW{G0} opponent</th><th>Clean sheet chance</th></tr>{body}</table>')
    html('<div class="note" style="margin-top:6px">Form uses the last 5 results. xG for and against use the last 10 games, pulled a little towards the league average. '
         'All of these feed the model, so a team in hot form or with a tight defence already lifts its players\' projections.</div>')

# ================================================================== HOW IT WORKS
else:
    sc = pd.read_csv(ROOT / 'models' / 'test_scores.csv'); s = sc[sc.group == 'starters']
    nm = {'last5': 'Average of last 5 games', 'gbm': 'Gradient boosting', 'mlp': 'PyTorch network', 'blend': 'Blend (used here)'}
    html(ui.cards([(f'{s[s.predictor == "blend"]["rank corr (per GW)"].iloc[0]:.2f}', 'ranking accuracy of our model (regular starters)', 'g'),
                   (f'{s[s.predictor == "last5"]["rank corr (per GW)"].iloc[0]:.2f}', 'ranking accuracy of recent form', ''),
                   ('5 seasons', 'of game-by-game history, 2022-23 to now', 'c'), ('< 1 sec', 'to search every legal squad', '')]))
    st.markdown("""
#### What it predicts
For every player and every fixture, his **expected FPL points**. A double gameweek adds both fixtures and a blank gives zero.
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
    evn = {'goal': 'Scores', 'assist': 'Assists', 'cs': 'Clean sheet (player-level model)', 'mins60': 'Plays 60+ minutes', 'bonus': 'Gets bonus points',
           'dc': 'Defensive contribution (DEF, MID, FWD; 2025-26 on)'}
    html('<div class="sec" style="font-size:1.05rem">Chances: predicted vs what happened</div><table class="fpl"><tr><th>Event</th><th>Predicted rate</th><th>Actual rate</th><th>Top 200 predicted</th><th>Top 200 actual</th><th>Better than guessing the average</th></tr>' +
         ''.join(f'<tr><td><b>{evn[r.event]}</b></td><td>{100 * r._3:.1f}%</td><td>{100 * r._2:.1f}%</td><td>{100 * r._7:.1f}%</td><td>{100 * r._8:.1f}%</td><td>{r._4:.0f}%</td></tr>' for r in ev.itertuples()) + '</table>')
    html('<div class="note" style="margin-top:6px">Clean sheets are running hot this season (29% of team games, against 21% to 27% in the last four seasons and about 23% expected from the chances teams allow), '
         'so the model expects some return to normal. The app uses the team-based clean-sheet model.</div>')
    html('<div class="note" style="margin-top:10px">Regular starters only. FPL points are very random: one goal changes a score by 4 to 10 points, so a typical error of about 3 is normal. '
         'The model ranks players better than recent form, but nobody can tell you who will score this weekend. Projections further ahead hold each player\'s form at today\'s level, so they are rougher.</div>')

html(ui.CREDIT)
