"""The PyTorch model: a position embedding plus a small feed-forward network that outputs expected points."""
import numpy as np, torch, torch.nn as nn


class PointsNet(nn.Module):
    def __init__(self, n_feat, hidden=128, drop=0.2):
        super().__init__()
        self.pos = nn.Embedding(4, 8)                      # GK, DEF, MID, FWD each get a learned 8-number vector
        self.body = nn.Sequential(
            nn.Linear(n_feat + 8, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden // 2, 1))

    def forward(self, x, pos):
        return self.body(torch.cat([x, self.pos(pos)], dim=1)).squeeze(1)


class Scaler:
    """Standardise features with the TRAINING mean and sd only, so no information leaks from the test period."""
    def __init__(self, mu, sd, features): self.mu, self.sd, self.F = mu, sd, features

    def tensors(self, df):
        x = ((df[self.F] - self.mu) / self.sd).fillna(0).clip(-6, 6).values.astype('float32')
        pos = (df.pos.values - 1).astype('int64')
        y = df.total_points.values.astype('float32') if 'total_points' in df else np.zeros(len(df), 'float32')
        return torch.tensor(x), torch.tensor(pos), torch.tensor(np.nan_to_num(y))


def predict(nets, scaler, df):
    x, p, _ = scaler.tensors(df)
    with torch.no_grad():
        return np.mean([n(x, p).numpy() for n in nets], axis=0)


def load(path):
    ck = torch.load(path, weights_only=False)
    import pandas as pd
    sc = Scaler(pd.Series(ck['mu']), pd.Series(ck['sd']), ck['features'])
    nets = []
    for st in ck['states']:
        n = PointsNet(len(ck['features'])); n.load_state_dict(st); n.eval(); nets.append(n)
    return nets, sc
