import fpl_data as fd, pandas as pd
h = fd.build(horizon=3)
print(h.shape, 'future rows:', int(h.future.sum()), '| duplicate player-fixture rows:', int(h.duplicated(['season', 'code', 'fixture']).sum()))
print(h[~h.future].groupby('season').agg(rows=('code', 'size'), xg_mean=('expected_goals', 'mean'), own_xgf=('own_xgf', 'mean'), cs=('cs_chance', 'mean')).round(3))
print(h[fd.EXTRA_FEATURES].isna().mean().round(3).to_string())
f = h[h.future & (h.gw == h[h.future].gw.min())]
print(f.groupby('team_id')[['own_xgf', 'own_xga', 'att_vs_def', 'cs_chance']].first().round(2).sort_values('att_vs_def').tail(5))
