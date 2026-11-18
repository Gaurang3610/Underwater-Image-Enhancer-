import torch, torch.nn.functional as F

def _gauss(n=11, s=1.5):
    g = torch.exp(-(torch.arange(n) - n // 2) ** 2 / (2 * s * s)); g /= g.sum()
    return (g[:, None] * g[None, :])

def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    w = _gauss().to(x)[None, None].repeat(3, 1, 1, 1); p = 5
    f = lambda t: F.conv2d(t, w, padding=p, groups=3)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sxx + syy + C2))
    return m.mean()

class HybridLoss(torch.nn.Module):
    """L = l1*|J^-J|_1 + l2*(1-SSIM)  (paper Eq. 5, l1=1.0, l2=0.5)"""
    def __init__(self, l1=1.0, l2=0.5): super().__init__(); self.l1, self.l2 = l1, l2
    def forward(self, pred, ref):
        return self.l1 * F.l1_loss(pred, ref) + self.l2 * (1 - ssim(pred, ref))
