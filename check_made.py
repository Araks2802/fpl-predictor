"""Confirmed transfers: Planner counts them in GW11, and a hit shows on every page."""
import re
from streamlit.testing.v1 import AppTest
for made in ([(569, 79)], [(569, 79), (8, 112)]):
    print(f'\n=== confirmed {made}')
    for page in ('Planner', 'My team', 'Transfers', 'League'):
        at = AppTest.from_file('app.py', default_timeout=240); at.session_state['nav'] = page; at.session_state['eid'] = '1'
        at.session_state['made'] = made; at.session_state['_made_for'] = '1'
        if page == 'League': at.session_state['lid'] = '5508'
        at.run()
        txt = ' '.join(m.value for m in at.markdown)
        if page == 'Planner':
            row = re.search(r'<div class="gw">GW11</div><div class="ft">(.*?)</div></div><div>(.*?)</div><div class="pts"[^>]*>([^<]+)<', txt)
            print('  Planner GW11 row:', re.sub('<[^>]+>', '', row.group(1)), '|', re.sub('<[^>]+>', '', row.group(2)), '|', row.group(3))
            print('  Planner bar:', 'after −4 hit' in txt)
        elif page == 'My team':
            print('  My team card:', re.findall(r'<div class="v g">([^<]+)</div><div class="l">(projected GW11 points[^<]*)</div>', txt))
        elif page == 'Transfers':
            print('  Transfers bar shows the hit:', 'after −4 hit' in txt)
        else:
            print('  League head to head:', re.findall(r'You <small>([^<]+)</small>', txt))
