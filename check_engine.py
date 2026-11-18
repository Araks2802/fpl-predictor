import json, time, pandas as pd
import fpl_engine as fe
boot = json.load(open('data/2026-27/bootstrap.json'))
P = pd.read_parquet('data/predictions.parquet'); meta = json.load(open('data/meta.json')); gws = meta['gws']
t = fe.load_entry(1, boot)
print(t['name'], '| bank', t['bank'], '| FT', t['free_transfers'], '| chips', t['chips_left'], '| squad from GW', t['squad_gw'])
sq = t['squad'].merge(P[['name', 'team', 'price', f'xp_{gws[0]}']], left_on='id', right_index=True)
print(sq.to_string())
base, per = fe.squad_points(P, list(t['squad'].id), gws); print('no transfers: weighted', round(base, 1), {k: round(v, 1) for k, v in per.items()})
for k in (1, 2, 3):
    t0 = time.time(); r = fe.optimise(P, gws, t['squad'], t['bank'], t['free_transfers'], max_transfers=k)
    v, per = fe.squad_points(P, r['squad'], gws)
    print(f"max {k}: out {[P.name[i] for i in r['out']]} in {[P.name[i] for i in r['inn']]} hits {r['hits']} bank {r['bank_after']} | weighted {v:.1f} gain {v - base - 4 * r['hits']:+.1f} | {time.time() - t0:.1f}s")
r = fe.optimise(P, gws, t['squad'], t['bank'], t['free_transfers'], chip='Wildcard'); v, _ = fe.squad_points(P, r['squad'], gws); print('wildcard', round(v, 1), len(r['inn']), 'changes, bank', r['bank_after'])
xi, cap, vice, bench = fe.best_xi(list(t['squad'].id), P, gws[0]); print('captain', P.name[cap], 'vice', P.name[vice], 'bench', [P.name[i] for i in bench])
