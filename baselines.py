import cv2, numpy as np

def clahe(bgr, clip=2.0):
    l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clip, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def dcp(bgr, w=0.95, patch=15, t0=0.1):
    I = bgr.astype(np.float32) / 255
    k = np.ones((patch, patch), np.uint8)
    dark = cv2.erode(I.min(2), k)
    idx = np.argsort(dark.ravel())[-max(1, dark.size // 1000):]
    A = I.reshape(-1, 3)[idx].mean(0)
    t = 1 - w * cv2.erode((I / A).min(2), k)
    J = (I - A) / np.maximum(t, t0)[..., None] + A
    return (np.clip(J, 0, 1) * 255).astype(np.uint8)
