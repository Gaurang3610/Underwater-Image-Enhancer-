import argparse, cv2, numpy as np, torch
from scipy.stats import wilcoxon
from dataset import split_pairs
from infer import load_model, enhance_rgb
import baselines as B, metrics as M

p = argparse.ArgumentParser()
p.add_argument("--pair", nargs=2, action="append", required=True)
p.add_argument("--ckpt", default="best_v2.pt"); p.add_argument("--limit", type=int, default=None)
p.add_argument("--max_images", type=int, default=None)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
_, _, te = split_pairs(a.pair, limit=a.limit)
te = te[:a.max_images] if a.max_images else te
net = load_model(a.ckpt, dev)
methods = {
    "Raw": lambda im: im, "CLAHE": B.clahe, "DCP": B.dcp,
    "U-Net": lambda im: enhance_rgb(net, im[..., ::-1].copy())[..., ::-1].copy()}
R = {m: {k: [] for k in ("PSNR", "SSIM", "UIQM", "UCIQE")} for m in methods}
for i, (ra, rf) in enumerate(te):
    raw, ref = cv2.imread(ra), cv2.imread(rf)
    ref = cv2.resize(ref, (raw.shape[1], raw.shape[0]))
    for m, fn in methods.items():
        out = fn(raw)
        R[m]["PSNR"].append(M.psnr(out, ref)); R[m]["SSIM"].append(M.ssim(out, ref))
        R[m]["UIQM"].append(M.uiqm(out)); R[m]["UCIQE"].append(M.uciqe(out))
    if (i + 1) % 25 == 0: print(f"{i+1}/{len(te)}")
print(f"\nTest images: {len(te)}")
print(f"{'Method':8}" + "".join(f"{k:>18}" for k in R['Raw']))
for m in R:
    print(f"{m:8}" + "".join(f"{np.mean(v):>10.3f}±{np.std(v):.3f}" for v in R[m].values()))
for base in ("CLAHE", "DCP"):
    for k in ("PSNR", "SSIM"):
        print(f"Wilcoxon U-Net vs {base} {k}: p = {wilcoxon(R['U-Net'][k], R[base][k]).pvalue:.2e}")
