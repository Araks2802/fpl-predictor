"""Confirmed transfers must belong to one team: old links, another team's link, or switching teams must never add them to your team."""
from streamlit.testing.v1 import AppTest
cases = [('old-style link, no team', {'team': '1', 'made': '8-112'}, '1'),
         ('saved for team 1, viewing team 1', {'team': '1', 'made': '1~8-112'}, '1'),
         ('saved for team 1, viewing team 71002', {'team': '71002', 'made': '1~8-112'}, '71002')]
for name, qp, eid in cases:
    at = AppTest.from_file('app.py', default_timeout=240)
    for k, v in qp.items(): at.query_params[k] = v
    at.session_state['nav'] = 'My team'; at.run()
    print(f'{name:40s} -> confirmed transfers: {at.session_state["made"]} | team box: {at.session_state["eid"]!r}')
# switching teams in one session: team 1 has a confirmed transfer stored, then you look at team 71002, then back at team 1
for eid in ('71002', '1'):
    at = AppTest.from_file('app.py', default_timeout=240); at.session_state['nav'] = 'My team'; at.session_state['eid'] = eid
    at.session_state['made_by_team'] = {'1': [(8, 112)]}; at.session_state['_made_for'] = 'some other team'; at.run()
    print(f'{"session has team 1 stored, viewing " + eid:40s} -> confirmed transfers: {at.session_state["made"]}')
