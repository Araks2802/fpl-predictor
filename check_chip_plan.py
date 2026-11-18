import json, time, pandas as pd, fpl_engine as fe
boot = json.load(open('data/2026-27/bootstrap.json')); P = pd.read_parquet('data/predictions.parquet')
gws = sorted(int(c[3:]) for c in P.columns if c.startswith('xp_'))
t = fe.load_entry(1, boot); print('used', t['chips_used'], '| left this half', t['chips_left'])
t0 = time.time(); V = fe.chip_values(P, t['squad'], t['bank'], t['free_transfers'], [g for g in gws if g <= 19]); print(f'first half: {time.time() - t0:.1f}s')
print(V.round(1).to_string()); print('plan', fe.plan_chips(V, ['Bench Boost', 'Triple Captain', 'Free Hit', 'Wildcard']))
t0 = time.time(); V2 = fe.chip_values(P, t['squad'], t['bank'], t['free_transfers'], [g for g in gws if g >= 20], wc_starts=0); print(f'second half: {time.time() - t0:.1f}s')
print(fe.plan_chips(V2, ['Bench Boost', 'Triple Captain', 'Free Hit']))
