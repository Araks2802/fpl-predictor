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
with st.sidebar:
    if BOOST: st.caption(f'Form boost on: {BOOST:.0%}')
    html('<div style="font-family:Outfit;font-size:1.5rem;font-weight:800;margin:4px 0 2px 0">FPL <span style="color:#00ff87">Predictor</span></div>'
         f'<div class="note">Gameweek {G0} deadline in <b style="color:#00ff87">{countdown}</b></div>')
    st.write('')
    mode = st.segmented_control('Your team', ['Team ID', 'Pick players'], default='Team ID', key='mode')
    team = None
    if mode != 'Pick players':
        eid = st.text_input('FPL team ID', placeholder='e.g. 1234567', key='eid', help='On the FPL website open Points. The number after /entry/ in the address is your ID.')
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
        with st.expander('Bank and free transfers'):
            team['bank'] = int(round(st.number_input('Bank (£m)', 0.0, 100.0, team['bank'] / 10, 0.1, key='bank') * 10))
            team['free_transfers'] = st.number_input('Free transfers', 0, 5, int(team['free_transfers']), key='ft')
            st.caption('Worked out from your public transfer history. Check against the FPL site.')
    html(ui.SB_CREDIT)

# ------------------------------------------------------------------ header and navigation
def _ago(t):
    m = (datetime.now(timezone.utc) - datetime.fromisoformat(t)).total_seconds() / 60
    return f'{m:.0f} min ago' if m < 90 else (f'{m / 60:.0f} h ago' if m < 48 * 60 else f'{m / 1440:.0f} days ago')
updated_pill = (f'<span class="pill">Updated {_ago(meta["updated_at"])}' + (' · late team news' if meta.get('update_type', '').startswith('late') else '') + '</span>') if meta.get('updated_at') else ''
sub = (f"{team['manager']} · {team['points']} pts · rank {team['rank']:,}" if team and team.get('rank') else 'Projected points, best XI, captain and transfers for the coming gameweek')
html(f'<div class="hero"><div class="hero-title">{ui.esc(team["name"]) if team else "Your FPL team,"} <span>{"GW" + str(G0) if team else "projected."}</span></div>'
     f'<div class="hero-sub">{ui.esc(sub)}</div>'
     f'<span class="pill g">GW{G0} · {deadline:%a %d %b %H:%M} UTC</span><span class="pill">Deadline in {countdown}</span>{updated_pill}</div>')
page = st.segmented_control('Page', ['My team', 'Transfers', 'Planner', 'Captain', 'Chips', 'Differentials', 'Compare', 'Players', 'Fixtures', 'How it works'], default='My team', label_visibility='collapsed', key='nav') or 'My team'


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
    total = xp[xi].sum() + xp[cap]; bpts = xp[bench].sum()
    value = (team['squad'].now.sum() + team['bank']) / 10
    html(ui.cards([(f'{total:.0f}', f'projected GW{gw} points', 'g'), (ui.esc(P.name[cap]), f'captain · {2 * xp[cap]:.1f} pts', 'c'),
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
            x2, c2_, _, _ = lineup(ids, g, BOOST); tots.append(P.loc[x2, f'xp_{g}'].sum() + P.loc[c2_, f'xp_{g}'])
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
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    plan = c1.segmented_control('Plan over', ['1 GW', '3 GWs'], default='3 GWs', key='plan') or '3 GWs'
    chip = c2.segmented_control('Transfer chip', ['None'] + [x for x in team['chips_left'] if x in ('Wildcard', 'Free Hit')], default='None', key='chip') or 'None'
    maxt = c3.number_input('Max transfers', 1, 5, min(max(team['free_transfers'], 1) + 1, 5), disabled=chip != 'None')
    roll = c4.number_input('Saving a transfer is worth', 0.0, 3.0, 1.0, 0.5, help='Points. A transfer must gain at least this much to be used now.')
    html('<div class="note" style="margin:-4px 0 8px 0">Only Wildcard and Free Hit change transfers. For Bench Boost, Triple Captain and the best week to play each chip, open the <b>Chips</b> tab.</div>')
    with st.expander('Never buy / always keep'):
        a, b = st.columns(2)
        banned = a.multiselect('Never buy', [i for i in P.index if i not in ids], format_func=LAB.get)
        keep = b.multiselect('Always keep', ids, format_func=LAB.get)
    gws = ALL_GWS[:int(plan.split()[0])]; ch = None if chip == 'None' else chip
    sj = team['squad'][['id', 'sell', 'now']].to_json()
    base, _ = fe.squad_points(P, ids, gws if ch != 'Free Hit' else gws[:1])
    opts = []
    with st.spinner('Checking every legal squad...'):
        for k in ([maxt] if ch else range(0, maxt + 1)):
            try: res = run_opt(sj, team['bank'], team['free_transfers'], k, ch, roll, tuple(banned), tuple(keep), tuple(gws), BOOST)
            except ValueError: continue
            v, _ = fe.squad_points(P, res['squad'], res['gws'])
            opts.append((k, res, v - base - fe.HIT * res['hits']))
    if not opts: st.error('No legal squad can be reached with these settings. Check the never buy and always keep lists.'); done()
    k, res, gain = opts[-1]
    sec('Suggested transfers', f'planning over GW{res["gws"][0]}' + (f' to GW{res["gws"][-1]}, later weeks count a little less' if len(res['gws']) > 1 else ''))
    if not res['inn']:
        html('<div class="card" style="padding:18px"><div class="v g" style="font-size:1.3rem">Hold. Keep your team.</div><div class="l" style="font-size:.9rem">No transfer gains enough to be worth it'
             + (', and saving gives you an extra free transfer next week.' if team['free_transfers'] < 5 else '.') + '</div></div>')
    else:
        for o, i in pairs(res):
            html(f'<div class="move">{move_card(o, "out", res["gws"])}<div class="arrow">→</div>{move_card(i, "in", res["gws"])}</div>')
        html(f'<div style="margin:10px 0 4px 0"><span class="gain">{gain:+.1f} pts</span> &nbsp; <span class="note">expected gain after {res["hits"]} hit'
             f'{"s" if res["hits"] != 1 else ""} (−{fe.HIT * res["hits"]}) · bank afterwards £{res["bank_after"] / 10:.1f}m</span></div>')
    if not ch and len(opts) > 1:
        sec('Every option')
        rows = ''.join(f'<tr><td><b>{"Hold" if kk == 0 else f"Up to {kk}"}</b></td><td>{", ".join(f"{ui.esc(P.name[o])} → {ui.esc(P.name[i])}" for o, i in pairs(rr)) or "No transfer"}</td>'
                       f'<td>{rr["hits"]}</td><td><b style="color:{ui.GREEN if g > 0 else "#fff"}">{g:+.1f}</b></td></tr>' for kk, rr, g in opts)
        html(f'<table class="fpl"><tr><th>Transfers</th><th>Moves</th><th>Hits</th><th>Gain</th></tr>{rows}</table>')
    g1 = res['gws'][0]; xi, cap, vice, bench = fe.best_xi(res['squad'], P, g1); xp = P[f'xp_{g1}']
    sec(f'Your GW{g1} team after these moves', f'{xp[xi].sum() + xp[cap]:.0f} projected points')
    html(ui.pitch(P, xi, bench, cap, vice, {i: xp[i] * (2 if i == cap else 1) for i in res['squad']}, {i: P.loc[i, f'opp_{g1}'] for i in res['squad']}, g1))
    html('<div class="note" style="margin-top:8px">Projections are averages. A move worth one or two points is close to a coin flip in a single week, so weigh injury news, rotation and price changes too.</div>')

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

# ================================================================== PLANNER: try your own transfers
elif page == 'Planner':
    if team is None: need_team(); done()
    sq = team['squad'].set_index('id'); ids = [int(i) for i in sq.index]
    c1, c2 = st.columns([2, 1.3])
    hz = c2.segmented_control('Judge over', ['Next GW', 'Next 3'], default='Next 3', key='hz_plan') or 'Next 3'
    gws = ALL_GWS[:HORIZONS[hz]]
    sell = c1.multiselect('Players to sell', ids, format_func=lambda i: f'{P.name[i]} ({P.team[i]}, sells for £{sq.sell[i] / 10:.1f}m)', key='sell',
                          placeholder='Pick the players you want to move on')
    col = f'h{HORIZONS[hz]}'
    buys, problems = {}, []
    if sell:
        sec('Replacements', f'same position · sorted by projected points over {hz.lower()}')
        cols = st.columns(min(len(sell), 3))
        taken = set(ids) - set(sell)
        for n, o in enumerate(sell):
            money = team['bank'] + sum(int(sq.sell[x]) for x in sell) - sum(int(round(P.price[b] * 10)) for x, b in buys.items())
            cand = P[(P.pos == P.pos[o]) & (P.status != 'u') & ~P.index.isin(list(taken) + list(buys.values()))].sort_values(col, ascending=False)
            # money left for this slot: bank + every sale - replacements already chosen
            lab = {i: f'{P.name[i]} ({P.team[i]}, £{P.price[i]:.1f}m) · {P.loc[i, col]:.1f} pts' + ('  · over budget' if round(P.price[i] * 10) > money else '') for i in cand.index}
            b = cols[n % 3].selectbox(f'Replace {P.name[o]}', [None] + cand.index.tolist(), format_func=lambda i: '- choose -' if i is None else lab[i], key=f'buy_{o}')
            if b is not None: buys[o] = b
    if sell and len(buys) == len(sell) and len(set(buys.values()) | (set(ids) - set(sell))) < 15:
        st.error('A replacement is already in your squad. Pick someone else.')
    elif sell and len(buys) == len(sell):
        new = [i for i in ids if i not in sell] + list(buys.values())
        bank_after = team['bank'] + sum(int(sq.sell[o]) for o in sell) - sum(int(round(P.price[b] * 10)) for b in buys.values())
        clubs = P.loc[new].team.value_counts()
        if bank_after < 0: problems.append(f'Over budget by £{-bank_after / 10:.1f}m.')
        if clubs.max() > 3: problems.append(f'Too many players from {clubs.idxmax()} ({clubs.max()}). The limit is 3 per club.')
        hits = max(0, len(sell) - team['free_transfers']); cost = fe.HIT * hits
        before = {g: v for g, v in zip(gws, [lineup(tuple(ids), g, BOOST) for g in gws])}
        after = {g: v for g, v in zip(gws, [lineup(tuple(sorted(new)), g, BOOST) for g in gws])}
        team_pts = lambda lu, g: P.loc[lu[0], f'xp_{g}'].sum() + P.loc[lu[1], f'xp_{g}']
        diffs = [team_pts(after[g], g) - team_pts(before[g], g) for g in gws]
        gross = sum(diffs); net = gross - cost
        colr = ui.GREEN if net > 0 else ui.PINK
        sec('The verdict', f'best XI and captain picked each week, before and after')
        for o, b in buys.items():
            html(f'<div class="move">{move_card(o, "out", gws)}<div class="arrow">→</div>{move_card(b, "in", gws)}</div>')
        html(ui.cards([(f'<span style="color:{colr}">{net:+.1f}</span>', f'net points over {hz.lower()} (after hits)', ''), (f'{gross:+.1f}', 'points gained before hits', ''),
                       (f'−{cost}' if cost else '0', f'hit cost ({len(sell)} transfer{"s" if len(sell) > 1 else ""}, {team["free_transfers"]} free)', ''),
                       (f'£{bank_after / 10:.1f}m', 'bank afterwards', 'c' if bank_after >= 0 else '')]))
        for p_ in problems: st.error(p_)
        if not problems:
            verdict = ('Worth it.' if net > 1 else 'Marginal: close to a coin flip.' if net > -1 else 'Not worth it.')
            html(f'<div class="note" style="margin-top:6px"><b style="color:{colr};font-size:1rem">{verdict}</b> '
                 + ('Remember a saved free transfer has value too.' if hits == 0 and net < 2 else '') + '</div>')
        if len(gws) > 1:
            fig = go.Figure(go.Bar(x=[f'GW{g}' for g in gws], y=diffs, marker=dict(color=[ui.GREEN if d >= 0 else ui.PINK for d in diffs]),
                                   text=[f'{d:+.1f}' for d in diffs], textposition='outside', hovertemplate='%{x}: %{y:+.1f} pts<extra></extra>'))
            fig = ui.plot_style(fig, 260); lo_, hi_ = min(diffs + [0]), max(diffs + [0]); pad = 0.25 * (hi_ - lo_ or 1)
            fig.update_yaxes(title='points gained', range=[lo_ - (pad if lo_ < 0 else 0), hi_ + pad])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        g1 = gws[0]; xi, cap, vice, bench = after[g1]; xp = P[f'xp_{g1}']
        sec(f'Your GW{g1} team after these moves', f'{team_pts(after[g1], g1):.1f} pts vs {team_pts(before[g1], g1):.1f} now')
        html(ui.pitch(P, xi, bench, cap, vice, {i: xp[i] * (2 if i == cap else 1) for i in new}, {i: P.loc[i, f'opp_{g1}'] for i in new}, g1))
    elif sell:
        st.info('Pick a replacement for each player you are selling.')
    else:
        html('<div class="note">Choose players to sell, pick who comes in, and see the net gain or loss in projected points, including hits, '
             'money left and the club limit. Gains count the best XI and captain each week, so a bench player swap counts for little.</div>')

# ================================================================== DIFFERENTIALS
elif page == 'Differentials':
    f1, f2, f3, f4 = st.columns([1.3, 1.6, 1.2, 1])
    pos = f1.pills('Position', ['GK', 'DEF', 'MID', 'FWD'], selection_mode='multi', default=['GK', 'DEF', 'MID', 'FWD'], key='pos_dif')
    hz = f2.segmented_control('Projected over', list(HORIZONS), default='Next 3', key='hz_dif') or 'Next 3'
    own = f3.slider('Owned by under (%)', 1, 30, 10, 1, key='own_dif')
    mx = f4.slider('Max price (£m)', 4.0, 16.0, 16.0, 0.5, key='mx_dif')
    col = f'h{HORIZONS[hz]}'
    D = P[(P.status != 'u') & (P.selected < own) & (P.price <= mx) & (P.chance.isna() | (P.chance >= 75))
          & P.pos.isin([k for k, v in POS.items() if v in (pos or POS.values())])].copy()
    # the edge over rivals: points you gain on the managers who do not own him
    D['edge'] = D[col] * (1 - D.selected / 100)
    D = D.sort_values('edge', ascending=False)
    if len(D):
        top = D.iloc[0]
        html(ui.cards([(ui.esc(top['name']), f'top differential · {top[col]:.1f} pts · owned {top.selected:.1f}%', 'g'),
                       (f'{len(D)}', f'players under {own}% owned', ''), (f'{D.head(10)[col].mean():.1f}', f'avg projection of the top 10 ({hz.lower()})', 'c')]))
    sec('Best differentials', f'owned by fewer than {own}% of managers · ranked by edge over rivals')
    html(ui.player_table(P, D.index[:20], ALL_GWS, col, max_val=D[col].max() if len(D) else 1,
                         extra=('Edge', lambda i: f'<b style="color:{ui.GREEN}">{D.edge[i]:.1f}</b>')))
    html('<div class="note" style="margin-top:8px"><b>Edge</b> = projected points × the share of managers who do not own him. '
         'Points from a player most rivals also own barely move your rank; points from a differential do. Low ownership also means more risk, so check minutes and news.</div>')
    if team is not None:
        mine = [i for i in team['squad'].id if P.selected[i] < own]
        sec('Your differentials', f'players in your squad owned by under {own}%')
        html(ui.player_table(P, sorted(mine, key=lambda i: -P.loc[i, col]), ALL_GWS, col) if mine else '<div class="note">None. Your squad is close to the template.</div>')
    sec('Differential captains', f'GW{G0} · owned by under 20%')
    C_ = P[(P.status != 'u') & (P.selected < 20) & (P.chance.isna() | (P.chance >= 75))].nlargest(5, f'xp_{G0}')
    html(ui.player_table(P, C_.index, ALL_GWS, f'xp_{G0}', extra=('As captain', lambda i: f'<b>{2 * P.loc[i, f"xp_{G0}"]:.1f}</b>')))

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

# ================================================================== PLAYERS
elif page == 'Players':
    f1, f2, f3 = st.columns([1.3, 1.9, 1])
    pos = f1.pills('Position', ['GK', 'DEF', 'MID', 'FWD'], selection_mode='multi', default=['GK', 'DEF', 'MID', 'FWD'])
    hz = f2.segmented_control('Projected over', list(HORIZONS), default='Next GW', key='hz_pl') or 'Next GW'
    sort = f3.selectbox('Sort by', ['Projected points', 'Points per £m', 'Goal chance', 'Clean sheet chance', 'Defensive contribution chance', 'Bonus chance', 'Price', 'Ownership', 'Form'])
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
    key = {'Projected points': D[col], 'Points per £m': D[col] / D.price, 'Price': D.price, 'Ownership': D.selected, 'Form': D.form,
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
