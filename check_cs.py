import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np, pandas as pd, fpl_data as fd, train_events as te
h = fd.build(horizon=1); T = te.team_table(h); T = T[~T.future & T.cs.notna()]
print('team clean-sheet rate by season:'); print(T.groupby('season').cs.agg(['mean', 'size']).round(3))
print('mean Poisson clean-sheet chance by season:'); print(T.groupby('season').cs_chance.mean().round(3))
cur = T[T.season == '2026-27']; print('2026-27 by gw:', cur.groupby('gw').cs.mean().round(2).to_dict())
# a simple calibrated alternative: logistic fit of team CS on log(expected goals against) with a season offset
