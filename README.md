# AI-Based Underwater Image Enhancement (Maritime Security)
1. `pip install -r requirements.txt`
2. Get UIEB (or EUVP) and arrange as `raw/` + `ref/` folders with matching filenames.
3. Train (Kaggle GPU): see v2 commands: train.py --pair RAW REF [--pair ...] --epochs 60 -> best_v2.pt
   Evaluate: evaluate.py --pair RAW REF [...] --ckpt best_v2.pt
4. Run prototype: `streamlit run app.py`
Files: model.py (U-Net), losses.py, dataset.py, train.py, baselines.py (CLAHE, DCP), metrics.py (PSNR, SSIM, UIQM, UCIQE), app.py.
Note: UIQM here is a simplified implementation; validate against a reference implementation before reporting numbers.
