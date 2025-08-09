#!/usr/bin/env python3
"""
Hybrid Soupliotis/Anandan + AI Supervised Video Enhancement
===========================================================
Preprocess with motion-aware classical CV, then enhance with an AI model (Real-ESRGAN).

Stages:
  1) Global stabilization (ECC or ORB) + smoothing
  2) (Optional) Local optical flow and motion-compensated temporal denoise
  3) AI enhancement per frame (Real-ESRGAN via PyTorch) [optional, --ai realesrgan]
  4) Tone/contrast and edge-aware sharpening

Usage example:
  python hybrid_soupliotis_anandan_ai.py --input in.mp4 --output out.mp4 --ai realesrgan --upscale 2

Dependencies: see requirements_hybrid.txt. GPU recommended for AI stage.
"""

import argparse, os, tempfile, shutil, sys
import numpy as np
import cv2

# --- Optional AI imports (lazy) ---
def load_realesrgan(model_name="RealESRGAN_x4plus", half=False, device="cuda"):
    """
    Load Real-ESRGAN for super-resolution / restoration.
    Requires: pip install realesrgan basicsr torch torchvision
    """
    try:
        from realesrgan import RealESRGANer
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet

        if model_name == "RealESRGAN_x4plus":
            # 4x RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x4plus.pth'
        elif model_name == "RealESRNet_x4plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRNet_x4plus.pth'
        else:
            raise ValueError("Unsupported Real-ESRGAN model_name")

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

def process(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open input: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare AI upsampler if requested
    upsampler = None
    if args.ai == "realesrgan":
        upsampler = load_realesrgan(model_name="RealESRGAN_x4plus", half=args.half, device=args.device)
        if upsampler is None:
            print("[AI] Proceeding without AI (Real-ESRGAN failed to load).")
            args.ai = "none"

    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if args.codec=='mp4v' else 'avc1'))
    outW, outH = (W, H)
    if args.ai != "none" and args.upscale > 1:
        outW, outH = int(W*args.upscale), int(H*args.upscale)
    out = cv2.VideoWriter(args.output, fourcc, fps, (outW, outH))
    if not out.isOpened(): raise RuntimeError(f"Cannot open output: {args.output}")

    prev_gray = None; M_smooth = None

    t = 0
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

        # --- AI enhancement ---
        if args.ai == "realesrgan" and upsampler is not None:
            # Real-ESRGAN expects RGB uint8
            rgb = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
            try:
                output, _ = upsampler.enhance(rgb, outscale=args.upscale)
                enhanced_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print("[AI] Real-ESRGAN enhance failed on frame", t, ":", e)
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
    p = argparse.ArgumentParser(description="Hybrid Soupliotis/Anandan + AI (Real-ESRGAN)")
    p.add_argument('--input', required=True, help='Input video')
    p.add_argument('--output', required=True, help='Output video')
    # Classical CV
    p.add_argument('--mode', default='ecc', choices=['ecc','orb'], help='Global motion method')
    p.add_argument('--stab-alpha', type=float, default=0.9, help='Smoothing alpha for stabilization')
    p.add_argument('--clip', type=float, default=1.0, help='Percentile clip for auto tone')
    p.add_argument('--sharpen', type=float, default=0.6, help='Sharpen amount')
    p.add_argument('--radius', type=float, default=1.5, help='Sharpen Gaussian sigma')
    p.add_argument('--no-clahe', action='store_true', help='Disable CLAHE')
    # AI
    p.add_argument('--ai', default='none', choices=['none','realesrgan'], help='AI enhancer')
    p.add_argument('--upscale', type=int, default=2, help='AI upscale factor (2 or 4)')
    p.add_argument('--device', default='cuda', choices=['cuda','cpu'], help='Torch device for AI')
    p.add_argument('--half', action='store_true', help='Use FP16 if available')
    # IO
    p.add_argument('--codec', default='mp4v', choices=['mp4v','avc1'], help='Output codec')
    p.add_argument('--max-frames', type=int, default=0, help='Process only first N frames (0=all)')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    process(args)
