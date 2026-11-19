import json, time, pandas as pd, fpl_engine as fe
boot = json.load(open('data/2026-27/bootstrap.json')); P = pd.read_parquet('data/predictions.parquet')
gws = sorted(int(c[3:]) for c in P.columns if c.startswith('xp_'))[:3]
t = fe.load_entry(1, boot); print('FT', t['free_transfers'], 'bank', t['bank'])
for hits in (False, True):
    t0 = time.time(); r = fe.optimise_plan(P, gws, t['squad'], t['bank'], t['free_transfers'], allow_hits=hits)
    print(f'allow hits={hits}  ({time.time() - t0:.1f}s)  free transfers left at the end: {r["ft_end"]}')
    for w in r['weeks']:
        print(f"  GW{w['gw']}: FT {w['ft_before']} | " + (', '.join(f'{P.name[o]} -> {P.name[i]}' for o, i in zip(w['out'], w['inn'])) or 'hold') + f" | hits {w['hits']} | bank {w['bank']}")
r = fe.optimise_plan(P, gws, t['squad'], t['bank'], 1, allow_hits=False)
w = r['weeks'][0]; assert len(w['inn']) <= 1, 'with 1 free transfer and no hits, week 1 must have at most 1 transfer'
print('check passed: 1 FT, no hits -> at most 1 transfer this week')
