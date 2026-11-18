"""Is the under-prediction for regular starters real? Check it on a whole held-out season half, not just 8 gameweeks.

Train on 2022-23 to 2024-25, then predict all of 2025-26 (38 gameweeks, never seen)."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np, pandas as pd
import fpl_data as fd, train_mlp as tm
from fpl_model import predict

h = fd.build(horizon=1)
hist = h[~h.future & h.total_points.notna() & h.pos.isin([1, 2, 3, 4])]
train = hist[hist.si < 2]; val = hist[hist.si == 2]; test = hist[hist.si == 3].copy()
nets, sc = tm.fit_nets(train, val, verbose=False); gb = tm.fit_gb(train)
test['pred'] = (predict(nets, sc, test) + gb.predict(test[tm.F].fillna(-1))) / 2
test.to_parquet('models/bias_2025_26.parquet')
for lab, m in (('all rows', test.minutes_r3 >= 0), ('starters (60+ mins over last 3)', test.minutes_r3 >= 60)):
    g = test[m]; print(f'\n{lab}: predicted {g.pred.mean():.2f} vs actual {g.total_points.mean():.2f}')
    for p, x in g.groupby('pos'): print(f'   {tm.fd_pos[p]}: predicted {x.pred.mean():.2f} vs actual {x.total_points.mean():.2f}  (n={len(x)})')
g = test[test.minutes_r3 >= 60].copy(); g['bin'] = pd.qcut(g.pred, 5, labels=False)
print('\nstarters by prediction fifth'); print(g.groupby('bin').agg(pred=('pred', 'mean'), actual=('total_points', 'mean'), played60=('minutes', lambda x: (x >= 60).mean())).round(2))
