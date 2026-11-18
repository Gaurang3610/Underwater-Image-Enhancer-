import torch, torch.nn as nn

def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

class UNet(nn.Module):
    """U-Net encoder-decoder (widths 32/64/128, bottleneck 256).
    residual=True: output = clamp(input + correction), starting as identity.
    residual=False: original sigmoid output (old best.pt)."""
    def __init__(self, w=(32, 64, 128), bott=256, residual=True):
        super().__init__()
        self.residual = residual
        self.enc = nn.ModuleList(); c = 3
        for o in w: self.enc.append(block(c, o)); c = o
        self.pool = nn.MaxPool2d(2)
        self.bott = block(c, bott); c = bott
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        for o in reversed(w):
            self.up.append(nn.ConvTranspose2d(c, o, 2, stride=2))
            self.dec.append(block(o * 2, o)); c = o
        self.out = nn.Conv2d(c, 3, 1)
        if residual:
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        inp, skips = x, []
        for e in self.enc:
            x = e(x); skips.append(x); x = self.pool(x)
        x = self.bott(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], 1))
        o = self.out(x)
        return torch.clamp(inp + o, 0, 1) if self.residual else torch.sigmoid(o)
