"""Smoke test: every page, with no team, a team ID, a hand-picked team, and the Transfers/Players options."""
from streamlit.testing.v1 import AppTest
import pandas as pd
P = pd.read_parquet('data/predictions.parquet'); c0 = [c for c in P.columns if c.startswith('xp_')][0]
pick, used = {}, set()
for p, n in {1: 2, 2: 5, 3: 5, 4: 3}.items():
    pick[p] = []
    for i in P[(P.pos == p) & (P.status == 'a')].sort_values(c0, ascending=False).index:
        if P.team_id[i] not in used: pick[p].append(int(i)); used.add(P.team_id[i])
        if len(pick[p]) == n: break
CASES = []
for page in ['My team', 'Transfers', 'Planner', 'Captain', 'Chips', 'Differentials', 'Compare', 'Players', 'Fixtures', 'How it works']:
    for setup in ('none', 'id', 'manual'): CASES.append((page, setup, {}))
CASES += [('Transfers', 'manual', {'chip': 'Wildcard'}), ('Transfers', 'manual', {'chip': 'Free Hit'}), ('Transfers', 'id', {'plan': '1 GW'}),
          ('Players', 'none', {'hz_pl': 'Season'}), ('Differentials', 'id', {'hz_dif': 'Season', 'own_dif': 5}), ('My team', 'id', {'boost': 30}), ('Compare', 'none', {'cmp': [411, 379, 1, 4], 'cmp_gw': 'GW12'}), ('Compare', 'none', {'cmp': []}), ('Fixtures', 'none', {'tf_sort': 'Defence (xG against)'}), ('Compare', 'none', {'cmp': [202, 60, 411]}),
          ('Planner', 'id', {'sell': [569], 'buy_569': 79}), ('Planner', 'id', {'sell': [569, 8], 'buy_569': 79, 'buy_8': 112, 'hz_plan': 'Next GW'}),
          ('Planner', 'id', {'sell': [569, 8, 566], 'buy_569': 379, 'buy_8': 112, 'buy_566': 4}), ('Planner', 'id', {'sell': [569]}), ('Players', 'none', {'hz_pl': 'Next 6'}), ('My team', 'id', {'hz_team': 'Next 3'})]
fails = 0
for page, setup, extra in CASES:
    at = AppTest.from_file('app.py', default_timeout=240)
    at.session_state['nav'] = page
    if setup == 'id': at.session_state['eid'] = '1'
    if setup == 'manual':
        at.session_state['mode'] = 'Pick players'
        for p, ids in pick.items(): at.session_state[f'pk{p}'] = ids
    for k, v in extra.items(): at.session_state[k] = v
    at.run()
    ex = list(at.exception); fails += bool(ex)
    shown = sum('class="mv in"' in m.value for m in at.markdown) + sum('Hold. Keep your team' in m.value for m in at.markdown)
    if page == 'Planner' and 'sell' in extra: print('   verdict cards:', [e.value for e in at.error], [m.value[:90] for m in at.markdown if 'net points' in m.value][:1])
    print(f'{page:13s} {setup:7s} {str(extra):28s} {"FAIL " + str(ex[0].value)[:400] if ex else "ok"}' + (f' | transfer cards {shown}' if page == 'Transfers' and setup != 'none' else ''))
print('failures:', fails)
