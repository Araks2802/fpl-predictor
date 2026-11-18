"""Collect the exact numbers and charts used in the project report (notes/report_facts.json and notes/fig/*.png)."""
import os, sys, json
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import fpl_data as fd, train_mlp as tm
from fpl_model import PointsNet

FIG = os.path.join(ROOT, 'notes', 'fig'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})
NAVY, GREEN, PINK, GREY = '#37003c', '#00b86b', '#e90052', '#999999'
facts = {}

h = fd.build(horizon=3)
hist = h[~h.future & h.total_points.notna() & h.pos.isin([1, 2, 3, 4])]
facts['rows_total'] = int(len(h)); facts['rows_hist'] = int(len(hist)); facts['rows_future'] = int(h.future.sum())
facts['rows_by_season'] = hist.groupby('season').size().to_dict()
facts['players_by_season'] = hist.groupby('season').code.nunique().to_dict()
facts['n_features'] = len(fd.FEATURES)
facts['feature_groups'] = {'BASE': fd.BASE_FEATURES, 'EXTRA': fd.EXTRA_FEATURES, 'TEAM_FORM': fd.TEAM_FORM, 'MATCHUP': fd.MATCHUP}
facts['points_dist'] = {'mean': float(hist.total_points.mean()), 'share_0': float((hist.total_points == 0).mean()), 'share_le2': float((hist.total_points <= 2).mean()),
                        'share_ge10': float((hist.total_points >= 10).mean()), 'max': int(hist.total_points.max())}
facts['minutes_zero_share'] = float((hist.minutes == 0).mean())
net = PointsNet(len(fd.FEATURES)); facts['n_params'] = int(sum(p.numel() for p in net.parameters()))
train, val, test = tm.split(h)
facts['split'] = {'train': len(train), 'val': len(val), 'test': len(test)}

# ---- points distribution chart
fig, ax = plt.subplots(figsize=(7, 3)); v = hist.total_points.clip(-3, 20).value_counts().sort_index()
ax.bar(v.index, v.values / v.sum() * 100, color=NAVY); ax.set_xlabel('points in a match'); ax.set_ylabel('% of player-matches'); ax.set_title('Most FPL scores are 0, 1 or 2 points')
plt.tight_layout(); plt.savefig(f'{FIG}/points_dist.png'); plt.close()

# ---- one recorded training run: train and validation loss per epoch
tm.F = fd.FEATURES
sc = tm.Scaler(train[tm.F].mean(), train[tm.F].std().replace(0, 1), tm.F) if hasattr(tm, 'Scaler') else None
from fpl_model import Scaler
sc = Scaler(train[tm.F].mean(), train[tm.F].std().replace(0, 1), tm.F)
Xtr, Ptr, ytr = sc.tensors(train); Xva, Pva, yva = sc.tensors(val)
torch.manual_seed(0); net = PointsNet(len(tm.F)); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.MSELoss()
tr_l, va_l = [], []
for ep in range(30):
    net.train(); perm = torch.randperm(len(Xtr)); tot = 0
    for i in range(0, len(perm), 512):
        idx = perm[i:i + 512]; opt.zero_grad(); loss = lossf(net(Xtr[idx], Ptr[idx]), ytr[idx]); loss.backward(); opt.step(); tot += loss.item() * len(idx)
    net.eval()
    with torch.no_grad(): va_l.append(lossf(net(Xva, Pva), yva).item()); tr_l.append(tot / len(Xtr))
best = int(np.argmin(va_l)); facts['curve'] = {'train': tr_l, 'val': va_l, 'best_epoch': best + 1}
fig, ax = plt.subplots(figsize=(7, 3.2)); ax.plot(range(1, 31), tr_l, color=NAVY, lw=2, label='training loss'); ax.plot(range(1, 31), va_l, color=PINK, lw=2, label='validation loss')
ax.axvline(best + 1, color=GREY, ls='--'); ax.text(best + 1.3, max(va_l), f'best validation: epoch {best + 1}', color=GREY, fontsize=8, va='top')
ax.set_xlabel('epoch'); ax.set_ylabel('MSE'); ax.legend(frameon=False); ax.set_title('Training keeps improving; validation stops: that gap is overfitting')
plt.tight_layout(); plt.savefig(f'{FIG}/training_curve.png'); plt.close()

# ---- calibration by decile on the test gameweeks
t = pd.read_csv('models/test_predictions.csv'); t['bin'] = pd.qcut(t.blend.rank(method='first'), 10, labels=False)
c = t.groupby('bin').agg(pred=('blend', 'mean'), act=('total_points', 'mean'))
facts['calib'] = c.round(3).to_dict(orient='list'); facts['calib_overall'] = [float(t.blend.mean()), float(t.total_points.mean())]
fig, ax = plt.subplots(figsize=(4.6, 4)); ax.plot([0, 4.5], [0, 4.5], color=GREY, ls='--', lw=1); ax.plot(c.pred, c.act, 'o-', color=NAVY, lw=2)
ax.set_xlabel('average predicted points'); ax.set_ylabel('average actual points'); ax.set_title('Calibration: predicted vs actual\n(each dot = 10% of test rows)')
plt.tight_layout(); plt.savefig(f'{FIG}/calibration.png'); plt.close()

# ---- method comparison and other saved results
sc_ = pd.read_csv('models/test_scores.csv'); facts['scores'] = sc_.round(4).to_dict(orient='records')
s = sc_[sc_.group == 'starters'].set_index('predictor')
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
lab = {'last5': 'Last 5 avg', 'gbm': 'Grad. boosting', 'mlp': 'PyTorch', 'blend': 'Blend'}
cols = [GREY, '#7a3f86', PINK, GREEN]
ax[0].bar([lab[k] for k in s.index], s['rank corr (per GW)'], color=cols); ax[0].set_title('Ranking accuracy (higher is better)'); ax[0].set_ylim(0, 0.32)
ax[1].bar([lab[k] for k in s.index], s['RMSE'], color=cols); ax[1].set_title('Typical error, RMSE (lower is better)'); ax[1].set_ylim(2.8, 3.4)
for a in ax: a.tick_params(axis='x', labelsize=8)
plt.tight_layout(); plt.savefig(f'{FIG}/methods.png'); plt.close()
facts['events'] = pd.read_csv('models/event_scores.csv').round(4).to_dict(orient='records')

# ---- experiments recorded during the project
fw = [(0, 0.269, 0.261), (0.1, 0.271, 0.259), (0.2, 0.265, 0.252), (0.3, 0.255, 0.242), (0.5, 0.229, 0.217)]
fig, ax = plt.subplots(figsize=(6, 3)); ax.plot([x[0] * 100 for x in fw], [x[1] for x in fw], 'o-', color=NAVY, lw=2, label='2026-27 GW3+')
ax.plot([x[0] * 100 for x in fw], [x[2] for x in fw], 'o-', color=PINK, lw=2, label='2025-26 full season'); ax.set_xlabel('extra weight on recent form (%)'); ax.set_ylabel('ranking accuracy')
ax.legend(frameon=False); ax.set_title('More weight on form made predictions worse'); plt.tight_layout(); plt.savefig(f'{FIG}/form_weight.png'); plt.close()
P = pd.read_parquet('data/predictions.parquet')
facts['examples'] = P.nlargest(8, 'xp_11')[['name', 'team', 'price', 'xp_11', 'p_goal_11', 'p_assist_11', 'p_cs_11', 'p_60_11', 'selected']].round(3).reset_index().to_dict(orient='records')
json.dump(facts, open(os.path.join(ROOT, 'notes', 'report_facts.json'), 'w'), indent=1, default=str)
print({k: facts[k] for k in ('rows_total', 'rows_hist', 'n_features', 'n_params', 'split', 'points_dist', 'minutes_zero_share')}); print('best epoch', best + 1, [round(x, 3) for x in va_l[:12]])
print(facts['rows_by_season'], facts['players_by_season'])
