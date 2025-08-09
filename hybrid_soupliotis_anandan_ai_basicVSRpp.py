#!/usr/bin/env python3
"""
Hybrid Soupliotis/Anandan + AI Supervised Video Enhancement
===========================================================
Now supports two AI backends:
  - None / Real-ESRGAN (built-in, frame-wise)
  - BasicVSR++ (via MMEditing) using an export/import workflow

Modes:
  1) Preprocess -> (optional) Real-ESRGAN -> write video
  2) Preprocess -> export frames for BasicVSR++ -> (user runs MMEditing) -> import frames -> write video

Examples:
  # Classical only
  python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out.mp4

  # Real-ESRGAN 2x
  python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out_ai.mp4 --ai realesrgan --upscale 2 --device cuda --half

  # BasicVSR++ (export preprocessed frames for MMEditing)
  python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out_bvsr.mp4 --ai basicvsrpp --export-frames preai_frames

  # After you run MMEditing and produce enhanced frames into 'enhanced_frames', import and encode:
  python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out_bvsr.mp4 --ai basicvsrpp --import-frames enhanced_frames --fps 30
"""

import argparse, os, sys
import numpy as np
import cv2

# --- Optional Real-ESRGAN loader ---
def load_realesrgan(model_name="RealESRGAN_x4plus", half=False, device="cuda"):
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        import torch

        if model_name == "RealESRGAN_x4plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth'
        else:
            raise ValueError("Unsupported model_name")

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            half=half and torch.cuda.is_available(),
            tile=0, tile_pad=10, pre_pad=0,
            device=device if torch.cuda.is_available() else 'cpu'
        )
        return upsampler
    except Exception as e:
        print("[AI] Real-ESRGAN not available:", e)
        return None

# --- Classical CV utilities ---
def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def estimate_global_transform(prev_gray, gray, mode='ecc', max_iters=50, eps=1e-4):
    if mode == 'ecc':
        warp_mode = cv2.MOTION_AFFINE
        M = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)
        try:
            _, M = cv2.findTransformECC(prev_gray, gray, M, warp_mode, criteria, None, 5)
            ok = True
        except cv2.error:
            ok = False
            M = np.eye(2, 3, dtype=np.float32)
        return M, ok
    else:
        orb = cv2.ORB_create(800)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return np.eye(2,3, dtype=np.float32), False
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 8:
            return np.eye(2,3, dtype=np.float32), False
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            M = np.eye(2,3, dtype=np.float32); ok=False
        else:
            ok=True
        return M.astype(np.float32), ok

def smooth_transform(M, M_prev_smooth, alpha=0.9):
    if M_prev_smooth is None: return M
    return alpha * M_prev_smooth + (1.0 - alpha) * M

def warp_affine(frame, M, shape):
    h, w = shape[:2]
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def auto_tone_color(bgr, clip_percent=1.0, use_clahe=True):
    img = bgr.astype(np.float32); out = img.copy()
    for c in range(3):
        lo = np.percentile(out[...,c], clip_percent)
        hi = np.percentile(out[...,c], 100-clip_percent)
        if hi > lo:
            out[...,c] = (out[...,c]-lo) * (255.0/(hi-lo))
            out[...,c] = np.clip(out[...,c], 0, 255)
    if use_clahe:
        ycrcb = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        ycrcb[...,0] = clahe.apply(ycrcb[...,0])
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return out.astype(np.uint8)

def edge_aware_sharpen(bgr, amount=0.6, radius=1.5):
    img = bgr.astype(np.float32)
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=radius, sigmaY=radius)
    detail = img - blur
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge = np.clip(np.abs(lap)/(np.percentile(np.abs(lap),95)+1e-6), 0, 1)
    edge = cv2.GaussianBlur(edge, (3,3), 0)[...,None]
    sharp = img + amount * detail * edge
    return np.clip(sharp, 0, 255).astype(np.uint8)

def write_video(frames, path, fps):
    if not frames: raise RuntimeError("No frames to write.")
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for f in frames:
        out.write(f)
    out.release()

def process(args):
    if args.ai == "basicvsrpp" and args.export_frames is None and args.import_frames is None:
        raise SystemExit("[BasicVSR++] Choose either --export-frames (before AI) or --import-frames (after AI).")

    # --- EXPORT PATH: preprocess and dump frames for BasicVSR++ ---
    if args.ai == "basicvsrpp" and args.export_frames is not None:
        os.makedirs(args.export_frames, exist_ok=True)
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened(): raise RuntimeError(f"Cannot open input: {args.input}")
        fps = cap.get(cv2.CAP_PROP_FPS) or args.fps or 30.0
        prev_gray = None; M_smooth = None; t = 0

        while True:
            ok, frame = cap.read()
            if not ok: break
            gray = to_gray(frame)
            if prev_gray is None:
                stab = frame
            else:
                M, okM = estimate_global_transform(prev_gray, gray, mode=args.mode, max_iters=40, eps=1e-4)
                if not okM: M = np.eye(2,3, dtype=np.float32)
                M_smooth = M if M_smooth is None else args.stab_alpha*M_smooth + (1-args.stab_alpha)*M
                H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                stab = warp_affine(frame, M_smooth, (H, W))
            toned = auto_tone_color(stab, clip_percent=args.clip, use_clahe=not args.no_clahe)
            sharp = edge_aware_sharpen(toned, amount=args.sharpen, radius=args.radius)
            cv2.imwrite(os.path.join(args.export_frames, f"{t:08d}.png"), sharp)
            prev_gray = gray; t += 1
            if args.max_frames and t >= args.max_frames: break

        cap.release()
        print(f"[Exported] {t} preprocessed frames -> {args.export_frames}")
        print("\nNext, run BasicVSR++ (MMEditing) on that frames folder, e.g.:")
        print("  mmedit_infer_restorer \\")
        print("    --model basicvsrpp --weights /path/to/basicvsrpp_reds4.pth \\")
        print("    --input", args.export_frames, "\\")
        print("    --output enhanced_frames --window-size 15 --fps", int(fps))
        print("\nThen import the enhanced frames and encode:")
        print("  python hybrid_soupliotis_anandan_ai.py --input", args.input, "--output", args.output, "\\")
        print("    --ai basicvsrpp --import-frames enhanced_frames --fps", int(fps))
        return

    # --- IMPORT PATH: read enhanced frames and encode video ---
    if args.ai == "basicvsrpp" and args.import_frames is not None:
        # Collect frames
        files = sorted([f for f in os.listdir(args.import_frames) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        if not files: raise RuntimeError("No images found in --import-frames directory.")
        frames = []
        for name in files:
            img = cv2.imread(os.path.join(args.import_frames, name), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read {name}")
            frames.append(img)
        fps = args.fps or 30.0
        write_video(frames, args.output, fps)
        print(f"[Done] Encoded {len(frames)} frames to {args.output}")
        return

    # --- Built-in path: classical + optional Real-ESRGAN ---
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open input: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    upsampler = None
    if args.ai == "realesrgan":
        upsampler = load_realesrgan(half=args.half, device=args.device)
        if upsampler is None:
            print("[AI] Proceeding without AI (Real-ESRGAN failed to load).")
            args.ai = "none"

    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if args.codec=='mp4v' else 'avc1'))
    outW, outH = (W, H)
    if args.ai == "realesrgan" and args.upscale > 1:
        outW, outH = int(W*args.upscale), int(H*args.upscale)
    out = cv2.VideoWriter(args.output, fourcc, fps, (outW, outH))
    if not out.isOpened(): raise RuntimeError(f"Cannot open output: {args.output}")

    prev_gray = None; M_smooth = None; t = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = to_gray(frame)

        if prev_gray is None:
            stab = frame
        else:
            M, okM = estimate_global_transform(prev_gray, gray, mode=args.mode, max_iters=40, eps=1e-4)
            if not okM: M = np.eye(2,3, dtype=np.float32)
            M_smooth = M if M_smooth is None else args.stab_alpha*M_smooth + (1-args.stab_alpha)*M
            stab = warp_affine(frame, M_smooth, (H, W))

        toned = auto_tone_color(stab, clip_percent=args.clip, use_clahe=not args.no_clahe)
        sharp = edge_aware_sharpen(toned, amount=args.sharpen, radius=args.radius)

        if args.ai == "realesrgan" and upsampler is not None:
            rgb = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
            try:
                output, _ = upsampler.enhance(rgb, outscale=args.upscale)
                enhanced_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print("[AI] Real-ESRGAN failed on frame", t, ":", e)
                enhanced_bgr = sharp
        else:
            enhanced_bgr = sharp

        out.write(enhanced_bgr)
        prev_gray = gray
        t += 1
        if args.max_frames and t >= args.max_frames: break

    cap.release(); out.release()
    print(f"[Done] {t} frames written to {args.output}")

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid Soupliotis/Anandan + AI (Real-ESRGAN or BasicVSR++)")
    p.add_argument('--input', required=True, help='Input video')
    p.add_argument('--output', required=True, help='Output video')

    # Classical CV
    p.add_argument('--mode', default='ecc', choices=['ecc','orb'], help='Global motion method')
    p.add_argument('--stab-alpha', type=float, default=0.9, help='Stabilization smoothing alpha')
    p.add_argument('--clip', type=float, default=1.0, help='Percentile clip for auto tone')
    p.add_argument('--sharpen', type=float, default=0.6, help='Sharpen amount')
    p.add_argument('--radius', type=float, default=1.5, help='Sharpen radius (Gaussian sigma)')
    p.add_argument('--no-clahe', action='store_true', help='Disable CLAHE')

    # AI selection
    p.add_argument('--ai', default='none', choices=['none','realesrgan','basicvsrpp'], help='AI enhancer')
    p.add_argument('--upscale', type=int, default=2, help='Real-ESRGAN upscale factor')

    # BasicVSR++ export/import workflow
    p.add_argument('--export-frames', type=str, default=None, help='Export preprocessed frames for BasicVSR++')
    p.add_argument('--import-frames', type=str, default=None, help='Import enhanced frames from BasicVSR++')
    p.add_argument('--fps', type=float, default=0.0, help='FPS override for writing video (import path)')

    # Real-ESRGAN device
    p.add_argument('--device', default='cuda', choices=['cuda','cpu'], help='Torch device for AI')
    p.add_argument('--half', action='store_true', help='Use FP16 for Real-ESRGAN if available')

    # IO
    p.add_argument('--codec', default='mp4v', choices=['mp4v','avc1'], help='Output codec')
    p.add_argument('--max-frames', type=int, default=0, help='Process only first N frames')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    process(args)
