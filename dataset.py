import os, math, random, torch, numpy as np
from PIL import Image
from torch.utils.data import Dataset

def split_pairs(pairs, seed=42, limit=None):
    """pairs: [(raw_dir, ref_dir), ...]. 70/15/15 split per dataset. Returns lists of (raw_path, ref_path)."""
    tr, va, te = [], [], []
    for raw, ref in pairs:
        names = sorted(f for f in os.listdir(raw) if os.path.exists(os.path.join(ref, f)))
        random.Random(seed).shuffle(names)
        if limit: names = names[:limit]
        a, b = int(.7 * len(names)), int(.85 * len(names))
        items = [(os.path.join(raw, n), os.path.join(ref, n)) for n in names]
        tr += items[:a]; va += items[a:b]; te += items[b:]
    return tr, va, te

class Pairs(Dataset):
    """Random 256 crops + flips/rot90 for training; center crop for validation."""
    def __init__(self, items, train=False, size=256):
        self.items, self.train, self.size = items, train, size
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        a, b = self.items[i]
        x, y = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
        if y.size != x.size: y = y.resize(x.size, Image.BICUBIC)
        s = self.size
        if min(x.size) < s:
            k = s / min(x.size); sz = (math.ceil(x.size[0] * k), math.ceil(x.size[1] * k))
            x, y = x.resize(sz, Image.BICUBIC), y.resize(sz, Image.BICUBIC)
        w, h = x.size
        l, t = (random.randint(0, w - s), random.randint(0, h - s)) if self.train else ((w - s) // 2, (h - s) // 2)
        box = (l, t, l + s, t + s)
        f = lambda im: torch.from_numpy(np.asarray(im.crop(box), np.float32) / 255).permute(2, 0, 1)
        x, y = f(x), f(y)
        if self.train:
            if random.random() < .5: x, y = x.flip(-1), y.flip(-1)
            if random.random() < .5: x, y = x.flip(-2), y.flip(-2)
            k = random.randint(0, 3); x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
        return x, y
