"""Download the four past seasons from the vaastav/Fantasy-Premier-League archive (skips files already present).
They are only needed for training, so they are not stored in the git repo."""
import requests
from pathlib import Path
D = Path(__file__).parent / 'data'
URL = 'https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{s}/{f}'
for s in ('2022-23', '2023-24', '2024-25', '2025-26'):
    (D / s).mkdir(parents=True, exist_ok=True)
    for f in ('gws/merged_gw.csv', 'players_raw.csv', 'fixtures.csv', 'teams.csv'):
        out = D / s / Path(f).name
        if out.exists() and out.stat().st_size > 1000: continue
        r = requests.get(URL.format(s=s, f=f), timeout=120); r.raise_for_status(); out.write_bytes(r.content); print('downloaded', out)
print('history ready')
