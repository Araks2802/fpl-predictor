"""Does the model under-predict? Compare predicted and actual points on the test gameweeks (2026-27 GW3 onwards)."""
import pandas as pd, numpy as np
t = pd.read_csv('models/test_predictions.csv')
print('overall mean  predicted %.2f  actual %.2f' % (t.blend.mean(), t.total_points.mean()))
t['bin'] = pd.qcut(t.blend.rank(method='first'), 10, labels=False)
c = t.groupby('bin').agg(pred=('blend', 'mean'), actual=('total_points', 'mean'), n=('blend', 'size'))
print('\nby prediction decile (9 = highest predictions)'); print(c.round(2).to_string())
top = t.sort_values('blend', ascending=False)
for q in (50, 200, 500):
    s = top.head(q); print(f'top {q} predictions: predicted {s.blend.mean():.2f}  actual {s.total_points.mean():.2f}')
st = t[t.minutes_r3 >= 60]
print('\nregular starters (60+ mins): predicted %.2f  actual %.2f' % (st.blend.mean(), st.total_points.mean()))
for p, g in st.groupby('pos'): print('  pos', p, 'predicted %.2f actual %.2f' % (g.blend.mean(), g.total_points.mean()))
