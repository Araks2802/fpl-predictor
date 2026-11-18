import torch, torch.nn as nn
w = torch.tensor(3.0, requires_grad=True); loss = (w - 5) ** 2; loss.backward(); print('grad', w.grad)
w = torch.tensor(3.0, requires_grad=True)
for step in range(50):
    loss = (w - 5) ** 2; loss.backward()
    with torch.no_grad(): w -= 0.1 * w.grad
    w.grad.zero_()
print('w after 50 steps', round(w.item(), 4))
class TinyNet(nn.Module):
    def __init__(self, n_in):
        super().__init__(); self.layers = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.layers(x).squeeze(1)
net = TinyNet(3); print('pred shape', net(torch.randn(5, 3)).shape, '| params', sum(p.numel() for p in net.parameters()))
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]); print(x.mean(dim=0))
import fpl_data as fd; print('features', len(fd.FEATURES))
