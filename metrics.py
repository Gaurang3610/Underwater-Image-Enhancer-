import cv2, numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr, structural_similarity as _ssim

def psnr(a, b): return _psnr(b, a, data_range=255)
def ssim(a, b): return _ssim(b, a, channel_axis=2, data_range=255)

def uciqe(bgr):
    """Yang & Sowmya 2015: c1*std(chroma)+c2*contrast(L)+c3*mean(sat)."""
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0] / 100, lab[..., 1] / 128, lab[..., 2] / 128
    chroma = np.sqrt(a ** 2 + b ** 2)
    sat = chroma / np.sqrt(chroma ** 2 + L ** 2 + 1e-6)
    fl = np.sort(L.ravel()); n = max(1, len(fl) // 100)
    con = fl[-n:].mean() - fl[:n].mean()
    return 0.4680 * chroma.std() + 0.2745 * con + 0.2576 * sat.mean()

def _trim(x, a=.1):
    x = np.sort(x.ravel()); k = int(a * len(x)); return x[k:len(x) - k]

def uiqm(bgr):
    """Panetta 2016 (simplified): 0.0282*UICM + 0.2953*UISM + 3.5753*UIConM."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(bgr)]
    rg, yb = r - g, (r + g) / 2 - b
    t1, t2 = _trim(rg), _trim(yb)
    uicm = -0.0268 * np.hypot(t1.mean(), t2.mean()) + 0.1586 * np.hypot(t1.std(), t2.std())
    sob = lambda c: np.hypot(cv2.Sobel(c, cv2.CV_32F, 1, 0), cv2.Sobel(c, cv2.CV_32F, 0, 1))
    def eme(c, k=8):
        h, w = c.shape; v = []
        for i in range(0, h - k + 1, k):
            for j in range(0, w - k + 1, k):
                p = c[i:i + k, j:j + k]; mx, mn = p.max(), p.min()
                if mn > 0 and mx > 0: v.append(np.log(mx / mn))
        return 2 * np.mean(v) if v else 0
    uism = sum(w * eme(sob(c) * c) for w, c in zip((.299, .587, .114), (r, g, b)))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    uicon = np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()) / 10   # contrast proxy
    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uicon
