import json, pandas as pd, fpl_engine as fe
boot = json.load(open('data/2026-27/bootstrap.json')); P = pd.read_parquet('data/predictions.parquet'); gws = json.load(open('data/meta.json'))['gws']
t = fe.load_entry(1, boot)
for chip in ('Wildcard', 'Free Hit'):
    r = fe.optimise(P, gws, t['squad'], t['bank'], t['free_transfers'], chip=chip)
    n = P.loc[r['squad']]
    print(chip, '| gameweeks', r['gws'], '| changes', len(r['inn']), '| hits', r['hits'], '| bank after', r['bank_after'],
          '| positions', n.pos.value_counts().sort_index().to_dict(), '| max per club', n.team.value_counts().max())
