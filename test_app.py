"""Smoke test: every page, with no team, a team ID, a hand-picked team, and the Transfers/Players options."""
import py_compile
py_compile.compile("app.py", doraise=True)   # a syntax error must fail the test run, not pass silently
from streamlit.testing.v1 import AppTest
import pandas as pd
P = pd.read_parquet('data/predictions.parquet'); c0 = [c for c in P.columns if c.startswith('xp_')][0]
pick, used = {}, set()
for p, n in {1: 2, 2: 5, 3: 5, 4: 3}.items():
    pick[p] = []
    for i in P[(P.pos == p) & (P.status == 'a')].sort_values(c0, ascending=False).index:
        if P.team_id[i] not in used: pick[p].append(int(i)); used.add(P.team_id[i])
        if len(pick[p]) == n: break
import json as _j, fpl_engine as _fe
TEAM1 = tuple(sorted(int(i) for i in _fe.load_entry(1, _j.load(open('data/2026-27/bootstrap.json')))['squad'].id))
CASES = []
for page in ['My team', 'Transfers', 'Planner', 'Captain', 'Chips', 'League', 'Compare', 'Players', 'Teams', 'Review', 'How it works']:
    for setup in ('none', 'id'): CASES.append((page, setup, {}))
CASES += [('Transfers', 'id', {'chip': 'Wildcard'}), ('Transfers', 'id', {'chip': 'Free Hit'}), ('Transfers', 'id', {'plan': '1 GW'}), ('Transfers', 'id', {'hits': True}), ('Transfers', 'id', {'chip': 'Wildcard', 'plan': '2 GWs'}),
          ('Players', 'none', {'hz_pl': 'Next 2'}), ('My team', 'id', {'boost': 30}), ('Compare', 'none', {'cmp_pos': 'FWD', 'cmp_FWD': [411, 379], 'cmp_gw': 'GW12'}), ('Compare', 'none', {'cmp_FWD': []}), ('Teams', 'none', {'tf_sort': 'Defence (xG against)'}), ('League', 'id', {'_lid': '5508'}), ('Review', 'id', {'rv_gw': 3}), ('Review', 'id', {'rv_gw': 10}), ('Review', 'id', {'rv_sort': 'Fell short most'}), ('Review', 'none', {'rv_sort': 'Beat prediction most', 'rv_played': False}),
          ('My team', 'id', {'made': [(569, 79)], '_made_for': '1'}), ('Transfers', 'id', {'made': [(569, 79), (8, 112)], '_made_for': '1'}),
          ('Planner', 'id', {'made': [(569, 79)], '_made_for': '1'}), ('My team', 'id', {'made': [(569, 411)], '_made_for': '1'}), ('My team', 'id', {'made': [(569, 379), (8, 4)], '_made_for': '1'}), ('League', 'id', {'_lid': '5508', 'h2h_hz': 'Next 3'}), ('League', 'none', {'_lid': '5508'}), ('League', 'id', {'_lid': '1757', 'lstyle': 'Chase'}), ('League', 'id', {'_lid': '999999999'}),
          ('Planner', 'id', {'pl_team': ('Solio Moose', TEAM1), 'pl_moves': [dict(gw=11, out=569, inn=79)]}),
          ('Planner', 'id', {'pl_team': ('Solio Moose', TEAM1), 'pl_moves': [dict(gw=11, out=569, inn=79), dict(gw=12, out=8, inn=112), dict(gw=12, out=566, inn=4)], 'pl_gw': 'GW12'}),
          ('Planner', 'id', {'pl_team': ('Solio Moose', TEAM1), 'pl_moves': [dict(gw=11, out=569, inn=411)]}),
          ('Planner', 'id', {'pl_team': ('Solio Moose', TEAM1), 'pl_moves': [dict(gw=12, out=79, inn=379)]}),
          ('Planner', 'id', {'pl_team': ('Solio Moose', TEAM1), 'pl_moves': [], 'pl_lineup': {11: dict(bench=[302, 166, 277, 173], cap=154, vice=411)}}),
          ('Planner', 'id', {'pl_team': ('Solio Moose', TEAM1), 'pl_moves': [dict(gw=11, out=166, inn=79)], 'pl_lineup': {11: dict(bench=[302, 166, 277, 173], cap=411, vice=154)}}),
          ('Transfers', 'id', {'plan': '2 GWs'}), ('Compare', 'none', {'cmp_pos': 'GK', 'cmp_GK': [496, 57, 1]}), ('Compare', 'id', {'cmp_pos': 'DEF'}), ('Compare', 'id', {'cmp_pos': 'MID'}),
           ('Players', 'none', {'hz_pl': 'Next 3'}), ('My team', 'id', {'hz_team': 'Next 3'})]
fails = 0
for page, setup, extra in CASES:
    at = AppTest.from_file('app.py', default_timeout=240)
    at.session_state['nav'] = page
    if setup == 'id': at.session_state['eid'] = '1'
    for k, v in extra.items(): at.session_state[k] = v
    at.run()
    ex = list(at.exception)
    if not at.markdown: ex = ex or ['page rendered nothing']   # an empty page means the script never ran
    fails += bool(ex)
    shown = sum('class="mv in"' in m.value for m in at.markdown) + sum('Hold. Keep your team' in m.value for m in at.markdown)
    if page == 'Planner' and 'pl_moves' in extra: print('   errors:', [e.value for e in at.error], '| moves kept:', len(at.session_state['pl_moves']))
    print(f'{page:13s} {setup:7s} {str(extra):28s} {"FAIL " + str(ex[0].value)[:400] if ex else "ok"}' + (f' | transfer cards {shown}' if page == 'Transfers' and setup != 'none' else ''))
print('failures:', fails)
