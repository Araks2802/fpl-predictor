"""Second opinion on feature sets: train on 2022-23 to 2024-25 and score every gameweek of 2025-26 (never seen)."""
import os, sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np, pandas as pd
import fpl_data as fd, train_mlp as tm
from fpl_model import predict

h = fd.build(horizon=1)
hist = h[~h.future & h.total_points.notna() & h.pos.isin([1, 2, 3, 4])]
train = hist[hist.si < 2]; val = hist[hist.si == 2]; test = hist[hist.si == 3].copy()
for name in sys.argv[1:]:
    tm.F = tm.FEATS[name]
    nets, sc = tm.fit_nets(train, val, verbose=False); gb = tm.fit_gb(train)
    test['p'] = (predict(nets, sc, test) + gb.predict(test[tm.F].fillna(-1))) / 2
    st = test[test.minutes_r3 >= 45]
    rc = np.mean([g[['p', 'total_points']].corr(method='spearman').iloc[0, 1] for _, g in st.groupby('gw')])
    top = np.mean([g.nlargest(10, 'p').total_points.mean() for _, g in st.groupby('gw')])
    print(f'{name:5s}  starters RMSE {np.sqrt(((st.p - st.total_points) ** 2).mean()):.3f}  ranking {rc:.3f}  top-10 picks avg {top:.2f}', flush=True)
