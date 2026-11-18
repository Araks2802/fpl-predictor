"""What is in the data for defensive contributions and bonus?"""
import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import pandas as pd, numpy as np, fpl_data as fd
h = fd.build(horizon=1); x = h[~h.future & (h.minutes > 0)]
for s in fd.SEASONS:
    g = x[x.season == s]
    print(s, 'dc mean %.2f  max %.0f  | share dc>=10 %.3f | bonus>0 %.3f' % (g.defensive_contribution.mean(), g.defensive_contribution.max(), (g.defensive_contribution >= 10).mean(), (g.bonus > 0).mean()))
g = x[x.si >= 3]
thr = np.where(g.pos == 2, 10, 12); hit = g.defensive_contribution >= thr
print('\n2025-26 on: DC threshold reached, by position (players who played):'); print(hit.groupby(g.pos).mean().round(3))
print('60+ mins only:'); print(hit[g.minutes >= 60].groupby(g.pos[g.minutes >= 60]).mean().round(3))
print('\nbonus>0 by position (60+ mins):'); print((g[g.minutes >= 60].bonus > 0).groupby(g[g.minutes >= 60].pos).mean().round(3))
raw = pd.read_csv('data/2026-27/history.csv', nrows=3); print('\nAPI columns:', [c for c in raw.columns if 'def' in c or c in ('tackles', 'recoveries', 'clearances_blocks_interceptions', 'bps', 'bonus')])
