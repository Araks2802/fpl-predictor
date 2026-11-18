"""Would giving more weight to recent form improve predictions? Mix the model with the last-5 average and score on unseen gameweeks."""
import numpy as np, pandas as pd
for name, path, pred in (('2026-27 GW3+', 'models/test_predictions.csv', 'blend'), ('2025-26 full season', 'models/bias_2025_26.parquet', 'pred')):
    t = pd.read_parquet(path) if path.endswith('parquet') else pd.read_csv(path)
    form = t['total_points_r5'] if 'total_points_r5' in t else t['last5']
    t = t.assign(form=form.fillna(0)); st = t[t.minutes_r3 >= 45]
    print(f'\n{name}  (regular starters)')
    for w in (0, 0.1, 0.2, 0.3, 0.5):
        p = (1 - w) * st[pred] + w * st.form
        rc = np.mean([g.assign(p=p[g.index])[['p', 'total_points']].corr(method='spearman').iloc[0, 1] for _, g in st.groupby('gw')])
        print(f'  form weight {w:.1f}: RMSE {np.sqrt(((p - st.total_points) ** 2).mean()):.3f}  ranking {rc:.3f}  mean predicted {p.mean():.2f} vs actual {st.total_points.mean():.2f}')
