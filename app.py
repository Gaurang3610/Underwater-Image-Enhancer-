import os, cv2, numpy as np, torch, streamlit as st
from infer import load_model, enhance_rgb
import baselines as B, metrics as M

st.set_page_config("Underwater Enhancer", layout="wide")
st.title("AI-Based Underwater Image Enhancer")
st.caption("Maritime security prototype · original and enhanced shown together — output is decision support, not evidence.")

CK = next((p for p in ("best_v2.pt", "best.pt") if os.path.exists(p)), None)
@st.cache_resource
def load(p): return load_model(p) if p else None
net = load(CK)
opts = (["U-Net (AI)"] if net else []) + ["CLAHE", "Dark Channel Prior"]
method = st.sidebar.selectbox("Method", opts)
if net: st.sidebar.success(f"Loaded {CK}")
else: st.sidebar.warning("No best_v2.pt / best.pt found. Showing classical baselines.")

def enhance(bgr):
    if method == "CLAHE": return B.clahe(bgr)
    if method == "Dark Channel Prior": return B.dcp(bgr)
    return enhance_rgb(net, bgr[..., ::-1].copy())[..., ::-1].copy()

up = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png"])
if up:
    bgr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1)
    out = enhance(bgr)
    c1, c2 = st.columns(2)
    c1.subheader("Original"); c1.image(bgr[..., ::-1])
    c2.subheader("Enhanced"); c2.image(out[..., ::-1])
    s = st.columns(4)
    s[0].metric("UIQM (orig → enh)", f"{M.uiqm(bgr):.2f} → {M.uiqm(out):.2f}")
    s[1].metric("UCIQE (orig → enh)", f"{M.uciqe(bgr):.3f} → {M.uciqe(out):.3f}")
    ref = st.file_uploader("Optional reference image (for PSNR/SSIM)", type=["jpg", "png"], key="r")
    if ref:
        r = cv2.imdecode(np.frombuffer(ref.read(), np.uint8), 1); r = cv2.resize(r, (out.shape[1], out.shape[0]))
        s[2].metric("PSNR", f"{M.psnr(out, r):.2f} dB"); s[3].metric("SSIM", f"{M.ssim(out, r):.3f}")
    st.download_button("Download enhanced image", cv2.imencode(".png", out)[1].tobytes(), "enhanced.png", "image/png")
