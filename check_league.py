import json, time, pandas as pd, fpl_engine as fe
boot = json.load(open('data/2026-27/bootstrap.json')); P = pd.read_parquet('data/predictions.parquet')
gws = sorted(int(c[3:]) for c in P.columns if c.startswith('xp_'))[:3]
t0 = time.time(); L = fe.load_league(5508, boot); print(L['name'], len(L['managers']), 'managers', f'{time.time() - t0:.1f}s', '| gw', L['gw'])
for m in L['managers'][:5]: print(m['rank'], m['manager'], m['total'], 'cap', P.name.get(m['captain']), '| chips left', m['chips_left'])
own, cap, n = fe.league_ownership(L, exclude=1)
print('most owned in league:', [(P.name[i], round(v, 2)) for i, v in own.sort_values(ascending=False).head(6).items()])
me = fe.load_entry(1, boot)
Q = fe.league_P(P, own, gws)
r = fe.optimise_plan(Q, gws, me['squad'], me['bank'], me['free_transfers'])
print('league-mode plan:', [(w['gw'], [P.name[o] for o in w['out']], [P.name[i] for i in w['inn']]) for w in r['weeks']])
