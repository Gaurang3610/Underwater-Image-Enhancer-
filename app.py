import os, base64, datetime, cv2, numpy as np, streamlit as st
import streamlit.components.v1 as components
from infer import load_model, enhance_rgb
import baselines as B, metrics as M

st.set_page_config("AquaClear", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")
ACC = "#4BA3C3"
st.markdown(f"""<style>
#MainMenu, footer, header {{visibility:hidden}}
.block-container {{max-width:1150px; padding-top:1rem}}
.hero {{text-align:center; margin:3rem 0 .3rem; font-size:2rem; font-weight:700}}
.sub {{text-align:center; opacity:.7; margin-bottom:1.5rem}}
.logo {{font-weight:700; font-size:1.15rem}} .logo:before {{content:"●"; color:{ACC}; margin-right:.45rem}}
[data-testid=stFileUploaderDropzone] {{border:2px dashed {ACC}; border-radius:12px; padding:2.5rem}}
.stButton>button, .stDownloadButton>button {{width:100%; border-radius:8px}}
</style>""", unsafe_allow_html=True)

DARK_CSS = """<style>
.stApp, [data-testid=stAppViewContainer], [data-testid=stMain] {background:#111E2B !important; color:#E6EEF4 !important}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp span,.stApp li,.stApp div[data-testid=stMarkdownContainer],.stApp [data-testid=stCaptionContainer] {color:#E6EEF4 !important}
[data-testid=stVerticalBlockBorderWrapper] {background:#162534 !important; border-color:#27425A !important}
[data-testid=stFileUploaderDropzone] {background:#162534 !important}
[data-baseweb=select]>div {background:#1B2D3D !important; color:#E6EEF4 !important}
.stButton>button:not([kind=primary]), .stDownloadButton>button:not([kind=primary]) {background:#1B2D3D !important; color:#E6EEF4 !important; border-color:#27425A !important}
hr {border-color:#27425A !important}
</style>"""

CK = next((p for p in ("best_v2.pt", "best.pt") if os.path.exists(p)), None)
@st.cache_resource
def load(p): return load_model(p) if p else None
net = load(CK)
OPTS = (["U-Net (AI)"] if net else []) + ["CLAHE", "Dark Channel Prior"]

ss = st.session_state
for k, v in dict(page="upload", n=0, gallery=[], results={}, s=100, col=0, con=0, bri=0,
                 method=OPTS[0]).items():
    ss.setdefault(k, v)
if ss.page in ("processing", "result") and "orig" not in ss: ss.page = "upload"

# ---------- helpers ----------
def enhance(rgb, m):
    bgr = rgb[..., ::-1].copy()
    if m == "CLAHE": return B.clahe(bgr)[..., ::-1].copy()
    if m == "Dark Channel Prior": return B.dcp(bgr)[..., ::-1].copy()
    return enhance_rgb(net, rgb)

def score(rgb, side=512):  # scored on a downscaled copy for speed
    h, w = rgb.shape[:2]; s = min(1, side / max(h, w))
    if s < 1: rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    return M.uiqm(bgr), M.uciqe(bgr)

def run(m):
    if m not in ss.results:
        e = enhance(ss.orig, m); ss.results[m] = (e, *score(e))
    return ss.results[m]

def adjust(o, e, s, col, con, bri):
    o, e = o.astype(np.float32), e.astype(np.float32)
    x = o + (e - o) * (s / 100)
    g = (x @ np.array([.299, .587, .114], np.float32))[..., None]
    x = g + (x - g) * (1 + col / 100)
    x = (x - 128) * (1 + con / 100) + 128 + bri * 1.28
    return np.clip(x, 0, 255).astype(np.uint8)

def jpg(rgb, q=88): return cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
def b64(rgb, side=1100):
    h, w = rgb.shape[:2]; s = min(1, side / max(h, w))
    if s < 1: rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(jpg(rgb)).decode()

SLIDER = """<style>body{margin:0;font-family:sans-serif}
#box{position:relative;margin:0 auto;overflow:hidden;border-radius:12px;touch-action:none;cursor:ew-resize;user-select:none}
#box img{position:absolute;inset:0;width:100%;height:100%;display:block;pointer-events:none}
#top{position:absolute;inset:0;clip-path:inset(0 50% 0 0)}
#bar{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#4BA3C3}
#knob{position:absolute;top:50%;left:1px;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;background:#4BA3C3;color:#fff;font:15px/34px sans-serif;text-align:center}
.tag{position:absolute;top:10px;padding:3px 10px;border-radius:6px;background:rgba(0,0,0,.55);color:#fff;font:12px sans-serif}</style>
<div id=box><img src="__AFTER__"><div id=top><img src="__BEFORE__"></div><div id=bar><div id=knob>&harr;</div></div>
<span class=tag style="left:10px">Before</span><span class=tag style="right:10px">After</span></div>
<script>var box=document.getElementById('box'),tp=document.getElementById('top'),bar=document.getElementById('bar'),R=__R__,d=false;
function fit(){var w=Math.min(document.body.clientWidth,560*R);box.style.width=w+'px';box.style.height=(w/R)+'px'}
function set(x){var r=box.getBoundingClientRect(),p=Math.max(0,Math.min(100,(x-r.left)/r.width*100));tp.style.clipPath='inset(0 '+(100-p)+'% 0 0)';bar.style.left=p+'%'}
box.addEventListener('pointerdown',function(e){d=true;set(e.clientX)});
window.addEventListener('pointermove',function(e){if(d)set(e.clientX)});
window.addEventListener('pointerup',function(){d=false});window.addEventListener('resize',fit);fit();</script>"""

def reset(): ss.s, ss.col, ss.con, ss.bri = 100, 0, 0, 0
def go(p): ss.page = p

# ---------- top bar ----------
c1, c2, c3, c4, c5 = st.columns([4.5, 1, 1, 1.5, 1.2])
if c5.toggle("Dark mode", key="dark"): st.markdown(DARK_CSS, unsafe_allow_html=True)
c1.markdown('<span class="logo">AquaClear</span>', unsafe_allow_html=True)
c2.button("Gallery", on_click=go, args=("gallery",)); c3.button("About", on_click=go, args=("about",))
if ss.page == "result": c4.button("New image", type="primary", on_click=go, args=("upload",))
st.divider()

# ---------- pages ----------
if ss.page == "upload":
    st.markdown('<div class="hero">Enhance your underwater photos</div>'
                '<div class="sub">Restore color, clarity and contrast in seconds</div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        up = st.file_uploader("Drag & drop an image here", type=["jpg", "jpeg", "png"],
                              label_visibility="collapsed", key=f"up{ss.n}")
        st.caption("JPG, PNG · large images are scaled to 2048 px")
        ex = os.path.exists("example.jpg") and st.button("Try an example image")
    data = up.getvalue() if up else (open("example.jpg", "rb").read() if ex else None)
    if data:
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        if bgr is None: st.error("Could not read this image."); st.stop()
        h, w = bgr.shape[:2]; s = min(1, 2048 / max(h, w))
        if s < 1: bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ss.orig = bgr[..., ::-1].copy(); ss.name = up.name if up else "example.jpg"
        ss.results = {}; ss.n += 1; reset(); ss.page = "processing"; st.rerun()

elif ss.page == "processing":
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="hero" style="margin-top:2rem">Enhancing your image…</div>', unsafe_allow_html=True)
        st.image(ss.orig, width=300)
        bar = st.progress(10, text="Preparing image")
        bar.progress(35, text="Running enhancement model")
        e, uq, uc = run(ss.method)
        bar.progress(75, text="Scoring quality")
        ss.base = score(ss.orig)
        ss.gallery.insert(0, (ss.name, jpg(cv2.resize(e, (300, int(300 * e.shape[0] / e.shape[1])))),
                              datetime.datetime.now().strftime("%d %b, %H:%M")))
        bar.progress(100, text="Done")
    ss.page = "result"; st.rerun()

elif ss.page == "result":
    left, right = st.columns([3, 1.1])
    with right:
        with st.container(border=True):
            st.markdown("**Adjustments**")
            st.selectbox("Method", OPTS, key="method")
            st.slider("Strength", 0, 100, key="s")
            st.slider("Color correction", -50, 50, key="col")
            st.slider("Contrast", -50, 50, key="con")
            st.slider("Brightness", -50, 50, key="bri")
            st.button("Reset", on_click=reset)
    with st.spinner("Enhancing…"):
        enh, uq, uc = run(ss.method)
    adj = adjust(ss.orig, enh, ss.s, ss.col, ss.con, ss.bri)
    with right:
        st.download_button("Download", cv2.imencode(".png", adj[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes(),
                           file_name="aquaclear_enhanced.png", mime="image/png", type="primary")
    with left:
        h, w = ss.orig.shape[:2]
        components.html(SLIDER.replace("__AFTER__", b64(adj)).replace("__BEFORE__", b64(ss.orig))
                        .replace("__R__", str(w / h)), height=580)
        st.caption(f"{ss.method} · UIQM {ss.base[0]:.2f} → {uq:.2f} · UCIQE {ss.base[1]:.3f} → {uc:.3f} "
                   "(scores are for the unadjusted enhancement; higher is better)")

elif ss.page == "gallery":
    st.subheader("Past enhancements")
    if not ss.gallery: st.info("Nothing here yet. Enhance an image and it will appear here (this session only).")
    for i in range(0, len(ss.gallery), 4):
        cols = st.columns(4)
        for col, (name, thumb, t) in zip(cols, ss.gallery[i:i + 4]):
            col.image(thumb, caption=f"{name} · {t}")
    st.button("Back", on_click=go, args=("upload",))

else:
    st.subheader("About")
    st.write("AquaClear enhances underwater images with a U-Net trained on paired underwater data, "
             "and compares it with CLAHE and Dark Channel Prior. Quality is reported with UIQM and UCIQE.")
    st.warning("Enhancement can invent plausible detail. Treat the output as decision support, "
               "not evidence, and keep a human operator in the loop.")
    st.button("Back", on_click=go, args=("upload",))
