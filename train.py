import argparse, time, torch
from torch.utils.data import DataLoader
from model import UNet; from losses import HybridLoss
from dataset import split_pairs, Pairs

p = argparse.ArgumentParser()
p.add_argument("--pair", nargs=2, action="append", required=True, metavar=("RAW", "REF"))
p.add_argument("--epochs", type=int, default=60); p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--limit", type=int, default=None)
p.add_argument("--out", default="best_v2.pt")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
tr, va, te = split_pairs(a.pair, limit=a.limit)
print(f"train {len(tr)} | val {len(va)} | test {len(te)} | device {dev}")
trl = DataLoader(Pairs(tr, True), a.bs, shuffle=True, num_workers=2, drop_last=True)
val = DataLoader(Pairs(va, False), a.bs, num_workers=2)
net, crit = UNet(residual=True).to(dev), HybridLoss()
opt = torch.optim.Adam(net.parameters(), a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-6)
best = 9e9
for ep in range(a.epochs):
    t0 = time.time(); net.train(); tl = 0
    for x, y in trl:
        x, y = x.to(dev), y.to(dev); opt.zero_grad()
        l = crit(net(x), y); l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += l.item()
    sched.step(); net.eval(); v = 0
    with torch.no_grad():
        for x, y in val: v += crit(net(x.to(dev)), y.to(dev)).item() * len(x)
    v /= len(va)
    mark = ""
    if v < best:
        best = v; mark = " *saved"
        torch.save({"state": net.state_dict(), "residual": True}, a.out)
    print(f"epoch {ep+1}/{a.epochs} train {tl/len(trl):.4f} val {v:.4f} ({time.time()-t0:.0f}s){mark}")
print("best val", round(best, 4))
