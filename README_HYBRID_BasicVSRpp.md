# Hybrid Soupliotis/Anandan + AI (Real-ESRGAN or BasicVSR++)

This adds a **BasicVSR++** backend via an export/import workflow (so you can use **MMEditing**),
alongside the built-in **Real-ESRGAN** frame-wise enhancer.

## Install (core)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements_hybrid.txt
```

### Optional: Real-ESRGAN (frame-wise AI)
The first run will auto-download weights. GPU recommended (`--device cuda --half`).

### Optional: BasicVSR++ (temporal AI via MMEditing)
BasicVSR++ is provided by **MMEditing**. Install per your CUDA/PyTorch (see MMEditing docs).
Quick sketch (Linux/CUDA 11.x; adjust to your versions):
```bash
pip install -U openmim
mim install "mmengine>=0.10.0"
mim install "mmcv>=2.0.0"
mim install mmdet mmedit
# Download a BasicVSR++ checkpoint, e.g. reds4 model:
# https://download.openmmlab.com/mmediting/v1/restorers/basicvsr/basicvsr_pp_reds4_8x4_600k.pth
```

## Usage

### A) Classical only
```bash
python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out.mp4
```

### B) Hybrid with Real-ESRGAN (2× upscale)
```bash
python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out_ai.mp4 \
  --ai realesrgan --upscale 2 --device cuda --half
```

### C) Hybrid with BasicVSR++ (export → run MMEditing → import)
1) **Export preprocessed frames** (Soupliotis/Anandan stages applied):
```bash
python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out_bvsr.mp4 \
  --ai basicvsrpp --export-frames preai_frames
```
This writes numbered PNGs into `preai_frames/`.

2) **Run BasicVSR++ via MMEditing** (separate step):
Use MMEditing’s CLI (command varies by version). For example:
```bash
# Example CLI (MMEditing >=1.0 may provide mmedit_infer_restorer)
mmedit_infer_restorer --model basicvsrpp \
  --weights /path/to/basicvsrpp_reds4.pth \
  --input preai_frames \
  --output enhanced_frames \
  --window-size 15 --fps 30
```
If your MMEditing version doesn’t have the CLI, use their demo/inference script or notebook equivalent.

3) **Import enhanced frames & encode video**:
```bash
python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out_bvsr.mp4 \
  --ai basicvsrpp --import-frames enhanced_frames --fps 30
```
Alternatively, use the helper:
```bash
python encode_from_frames.py --frames enhanced_frames --output out_bvsr.mp4 --fps 30
```

## Tips
- For **handheld**: `--mode ecc --stab-alpha 0.95 --clip 1.0 --sharpen 0.6`
- For **dashcam**: `--mode orb --stab-alpha 0.9 --clip 0.5 --sharpen 0.5 --radius 1.0`
- For **drone**: `--mode ecc --stab-alpha 0.98 --clip 1.0 --sharpen 0.4 --radius 2.0`
- For **low-light**: increase `--clip` (1.5–2.0) and lower `--sharpen`.

## Notes
- Real-ESRGAN is **frame-wise**; great detail but not explicitly temporal.
- BasicVSR++ is **temporal**; stronger consistency, heavier install.
- Export/import keeps this project lightweight while letting you use the full MMEditing stack when needed.
