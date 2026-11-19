"""Train and test the points model.

Run:  KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python train_mlp.py
(the two settings avoid a macOS crash when PyTorch and scikit-learn load different threading libraries)

We compare three predictors on gameweeks the model never saw (2026-27 GW3 onwards):
  1. last-5 average     the obvious baseline
  2. gradient boosting  a strong non-neural baseline (scikit-learn)
  3. PyTorch MLP        our neural network (ensemble of 5)
and a blend of 2 and 3.
"""
import os, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
import fpl_data as fd
from fpl_model import PointsNet, Scaler, predict

OUT = Path(__file__).parent / 'models'; OUT.mkdir(exist_ok=True)
fd_pos = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
# FEATS picks a feature set so new ideas can be tested against old ones on the same data
FEATS = {'base': fd.BASE_FEATURES, 'prev': fd.BASE_FEATURES + fd.EXTRA_FEATURES, 'team': fd.BASE_FEATURES + fd.EXTRA_FEATURES + fd.TEAM_FORM,
         'nodcb': fd.BASE_FEATURES + fd.EXTRA_FEATURES + fd.TEAM_FORM + fd.MATCHUP, 'all': fd.FEATURES}
F = FEATS[os.environ.get('FEATS', 'all')]


def split(h, final=False):
    hist = h[~h.future & h.total_points.notna() & h.pos.isin([1, 2, 3, 4])]
    if final:   # final model: train on everything played, validate on the last 4 played gameweeks for early stopping
        last = hist[hist.season == '2026-27'].gw.max()
        val = hist[(hist.season == '2026-27') & (hist.gw > last - 4)]
        return hist.drop(val.index), val, None
    train = hist[(hist.si < 3) | ((hist.season == '2025-26') & (hist.gw <= 28))]
    val = hist[(hist.season == '2025-26') & (hist.gw > 28)]
    test = hist[(hist.season == '2026-27') & (hist.gw >= 3)]
    return train, val, test


def fit_nets(train, val, n=5, epochs=40, lr=1e-3, wd=1e-4, bs=512, verbose=True):
    sc = Scaler(train[F].mean(), train[F].std().replace(0, 1), F)
    Xtr, Ptr, ytr = sc.tensors(train); Xva, Pva, yva = sc.tensors(val)
    nets = []
    for seed in range(n):
        torch.manual_seed(seed)
        net = PointsNet(len(F)); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd); lossf = nn.MSELoss()
        best, best_state, bad = 1e9, None, 0
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                opt.zero_grad(); loss = lossf(net(Xtr[idx], Ptr[idx]), ytr[idx]); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): v = lossf(net(Xva, Pva), yva).item()
            if v < best - 1e-4: best, bad, best_state = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 5: break
        net.load_state_dict(best_state); net.eval(); nets.append(net)
        if verbose: print(f'  net {seed}: stopped at epoch {ep + 1}, best validation MSE {best:.4f}')
    return nets, sc


def fit_gb(train):
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=24, l2_regularization=1.0, random_state=0)
    gb.fit(train[F].fillna(-1), train.total_points); gb.feature_list_ = list(F); return gb


def report(df, cols, title):
    rows = []
    for c in cols:
        e = df[c] - df.total_points
        top = [g[g.minutes_r3 >= 45].nlargest(10, c).total_points.mean() for _, g in df.groupby('gw')]
        corr = np.mean([g[[c, 'total_points']].corr(method='spearman').iloc[0, 1] for _, g in df.groupby('gw')])
        rows.append((c, np.sqrt((e ** 2).mean()), e.abs().mean(), corr, np.mean(top)))
    t = pd.DataFrame(rows, columns=['predictor', 'RMSE', 'MAE', 'rank corr (per GW)', 'avg pts of top-10 picks']).round(3)
    print(f'\n{title}'); print(t.to_string(index=False)); return t


if __name__ == '__main__':
    h = fd.build()
    train, val, test = split(h)
    print('TEST RUN  train', len(train), '| val', len(val), '| test', len(test))
    nets, sc = fit_nets(train, val); gb = fit_gb(train)
    test = test.copy()
    test['last5'] = test.total_points_r5.fillna(0)
    test['gbm'] = gb.predict(test[F].fillna(-1))
    test['mlp'] = predict(nets, sc, test)
    test['blend'] = (test.gbm + test.mlp) / 2
    a = report(test, ['last5', 'gbm', 'mlp', 'blend'], 'All players, 2026-27 GW3 onwards')
    b = report(test[test.minutes_r3 >= 45], ['last5', 'gbm', 'mlp', 'blend'], 'Regular starters (45+ mins a game over the last 3)')
    st_ = test[test.minutes_r3 >= 60]; top = test.nlargest(200, 'blend')
    print(f"\nCALIBRATION  starters: predicted {st_.blend.mean():.2f} vs actual {st_.total_points.mean():.2f} | top 200 predictions: {top.blend.mean():.2f} vs {top.total_points.mean():.2f}")
    for p, g in st_.groupby('pos'): print(f'   {fd_pos[p]}: predicted {g.blend.mean():.2f} vs actual {g.total_points.mean():.2f}')
    pd.concat([a.assign(group='all'), b.assign(group='starters')]).to_csv(OUT / 'test_scores.csv', index=False)
    tag = os.environ.get('FEATS', 'all'); a.assign(features=tag).to_csv(OUT / f'test_scores_{tag}.csv', index=False)
    # saved with ids and match stats, for the app's Review tab (predicted vs what happened, for weeks the model never saw).
    # The scores above start at GW3, but the same model never saw GW1-2 either, so the Review tab gets every played week of the season.
    rv = h[~h.future & h.total_points.notna() & h.pos.isin([1, 2, 3, 4]) & (h.season == '2026-27')].copy()
    rv['last5'] = rv.total_points_r5.fillna(0); rv['gbm'] = gb.predict(rv[F].fillna(-1)); rv['mlp'] = predict(nets, sc, rv); rv['blend'] = (rv.gbm + rv.mlp) / 2
    keep_ = ['season', 'gw', 'element', 'code', 'team_id', 'opponent_team', 'was_home', 'fixture', 'web_name', 'pos', 'total_points', 'minutes', 'goals_scored',
             'assists', 'bonus', 'clean_sheets', 'minutes_r3', 'last5', 'gbm', 'mlp', 'blend']
    rv[[c for c in keep_ if c in rv.columns]].to_csv(OUT / 'test_predictions.csv', index=False)

    if os.environ.get('FINAL') == '0': raise SystemExit
    print('\nFINAL MODEL: retrain on everything played so far')
    tr, va, _ = split(h, final=True)
    nets, sc = fit_nets(tr, va); gb = fit_gb(tr)
    torch.save({'states': [n.state_dict() for n in nets], 'mu': sc.mu.to_dict(), 'sd': sc.sd.to_dict(), 'features': F}, OUT / 'mlp.pt')
    joblib.dump(gb, OUT / 'gbm.joblib'); print('saved models/mlp.pt and models/gbm.joblib')
