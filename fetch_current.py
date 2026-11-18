"""Download the current season from the official FPL API.

  python fetch_current.py          full: players, fixtures AND every player's match-by-match history (~2 minutes)
  python fetch_current.py --light  light: players and fixtures only (prices, injury news, chance of playing). A few seconds.
                                   Used for the late team-news refresh, when no matches have been played since the last full fetch.
"""
import json, sys, time, requests, pandas as pd
from pathlib import Path
OUT = Path(__file__).parent / 'data' / '2026-27'; OUT.mkdir(parents=True, exist_ok=True)
B = 'https://fantasy.premierleague.com/api/'
s = requests.Session(); s.headers['User-Agent'] = 'Mozilla/5.0 (personal FPL project)'


def get(path, tries=4):
    for k in range(tries):
        try:
            r = s.get(B + path, timeout=30)
            if r.status_code == 200: return r.json()
        except requests.RequestException:
            pass
        time.sleep(3 * (k + 1))
    raise RuntimeError(f'FPL API failed for {path}')


boot = get('bootstrap-static/'); fx = get('fixtures/')
light = '--light' in sys.argv
if not light:
    rows = []
    for e in boot['elements']:
        try: hist = get(f"element-summary/{e['id']}/")['history']
        except RuntimeError: print('fail', e['id']); continue
        for h in hist: h['element'] = e['id']; rows.append(h)
        time.sleep(0.05)
    if len(rows) < 100: raise SystemExit('Too few history rows; the API may be down. Keeping the old files.')
    pd.DataFrame(rows).to_csv(OUT / 'history.csv', index=False); print(len(rows), 'history rows')
# write the small files last, so a failure above never leaves a half-updated set
json.dump(boot, open(OUT / 'bootstrap.json', 'w')); json.dump(fx, open(OUT / 'fixtures.json', 'w'))
print('light' if light else 'full', 'fetch done')
