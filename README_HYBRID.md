# Hybrid Soupliotis/Anandan + AI Supervised Video Enhancement

This package combines **motion-aware classical CV** with an **AI enhancer (Real-ESRGAN)**.

## 1) Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements_hybrid.txt
```

> GPU strongly recommended. If you only have CPU, add `--device cpu` when running.

## 2) Run
Basic (classical only):
```bash
python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out.mp4
```

Hybrid with AI (Real-ESRGAN) and 2x upscale:
```bash
python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out_ai.mp4 \
  --ai realesrgan --upscale 2 --device cuda --half
```

Notes:
- First AI run will **download model weights** automatically.
- If you see memory warnings, try `--upscale 2`, remove `--half`, or set `--device cpu` (slower).

## 3) Options
- `--mode ecc|orb` — global stabilization method
- `--stab-alpha` — camera path smoothing (0.9–0.98 typical)
- `--clip` — percent clip for auto tone (0.5–2.0 typical)
- `--sharpen`, `--radius` — edge-aware sharpening controls
- `--max-frames N` — quick test on first N frames

## 4) Tips
- For **handheld phone**: `--mode ecc --stab-alpha 0.95 --clip 1.0 --sharpen 0.6`
- For **dashcam**: `--mode orb --stab-alpha 0.9 --clip 0.5 --sharpen 0.5 --radius 1.0`
- For **drone**: `--mode ecc --stab-alpha 0.98 --clip 1.0 --sharpen 0.4 --radius 2.0`
- For **low-light**: increase `--clip` to 2.0 and reduce `--sharpen` to avoid noise emphasis.

## 5) Known limitations
- Real-ESRGAN is frame-wise; it’s robust but not explicitly temporal. Pre-stabilization reduces flicker.
- For best temporal consistency, consider BasicVSR++ (heavier install). This script keeps it simple and portable.

## 6) Credits
- **Classical pipeline** based on the Soupliotis/Anandan patents (Microsoft).
- **AI enhancer** via Real-ESRGAN (Xintao Wang et al.).
