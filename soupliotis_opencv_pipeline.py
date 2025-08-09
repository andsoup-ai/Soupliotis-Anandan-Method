#!/usr/bin/env python3
"""
Soupliotis/Anandan Method for Automated, Motion-Aware Video Enhancement
=======================================================================

This reference implementation follows the approach described in the patents:

    - US 2004/0001705 A1 — Video processing system and method for automatic
      enhancement of digital video (Soupliotis & Anandan, Microsoft)
    - US 2006/0290821 A1 — Automated video enhancement system and method
      (Soupliotis & Anandan)
    - US 7,746,382 B2 — Granted version of the above

Core principles:
  1) Estimate and smooth **global motion** (camera) for stabilization.
  2) Compute **local motion** (optical flow) for motion-compensated filtering.
  3) Apply **temporal denoising**, **tone/contrast correction**, and **edge-aware sharpening**,
     all guided by motion data to avoid ghosting or flicker.
  4) Fully automated, requiring no manual shot-by-shot adjustments.

This script is a practical, OpenCV-based reference for the method.
It is CPU-friendly and intended for educational and prototyping purposes.
For production deployment, optimize I/O, consider GPU acceleration,
and integrate with a robust video decode/encode pipeline.

Authors of the method: Andreas Soupliotis & Padmanabhan Anandan
Implementation: (your name or organization)
"""

import argparse
import cv2
import numpy as np


def read_video_props(cap: cv2.VideoCapture):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, w, h, n


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def estimate_global_transform(prev_gray, gray, mode='ecc', max_iters=100, eps=1e-5):
    """
    Estimate global motion from prev_gray -> gray.
    Returns M (2x3) affine warp and a flag indicating success.
    mode: 'ecc' or 'orb'
    """
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
        # ORB + RANSAC as alternative
        orb = cv2.ORB_create(1000)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return np.eye(2, 3, dtype=np.float32), False
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) < 8:
            return np.eye(2, 3, dtype=np.float32), False
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            M = np.eye(2, 3, dtype=np.float32)
            return M, False
        return M.astype(np.float32), True


def smooth_transform(M, M_prev_smooth, alpha=0.9):
    """Exponential moving average smoothing of the affine parameters."""
    if M_prev_smooth is None:
        return M
    return alpha * M_prev_smooth + (1.0 - alpha) * M


def warp_affine(frame, M, shape):
    h, w = shape[:2]
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def optical_flow(prev_gray, gray, algo='farneback'):
    if algo == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None,
                                            pyr_scale=0.5, levels=3, winsize=21,
                                            iterations=3, poly_n=5, poly_sigma=1.1, flags=0)
        return flow
    elif algo == 'dis':
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        flow = dis.calc(prev_gray, gray, None)
        return flow
    else:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = tvl1.calc(prev_gray, gray, None)
        return flow


def forward_backward_consistency(flow_fwd, flow_bwd):
    """
    Compute per-pixel forward-backward error to detect unreliable flow.
    Returns a mask in [0,1], where 1 is good flow.
    """
    h, w = flow_fwd.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    x2 = grid_x + flow_fwd[..., 0]
    y2 = grid_y + flow_fwd[..., 1]
    map_x = x2.astype(np.float32)
    map_y = y2.astype(np.float32)
    bwd_at_fwd = cv2.remap(flow_bwd, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    err = np.linalg.norm(bwd_at_fwd + flow_fwd, axis=2)
    med = np.median(err)
    scale = max(med * 2.0, 1e-3)
    mask = np.exp(-(err / scale) ** 2)
    return mask


def warp_by_flow(img, flow):
    h, w = img.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def motion_compensated_denoise(stabilized_prev, stabilized_cur, stabilized_next,
                               flow_prev_to_cur, flow_next_to_cur,
                               fb_mask_prev, fb_mask_next):
    """
    Simple 3-frame temporal denoise (median-weighted mean) with motion compensation and masks.
    """
    aligned_prev = warp_by_flow(stabilized_prev, flow_prev_to_cur)
    aligned_next = warp_by_flow(stabilized_next, flow_next_to_cur)

    w_prev = fb_mask_prev[..., None]
    w_next = fb_mask_next[..., None]
    w_cur = np.ones_like(w_prev)

    stack = np.stack([aligned_prev, stabilized_cur, aligned_next], axis=0)
    weights = np.stack([w_prev, w_cur, w_next], axis=0)
    weighted = np.sum(stack * weights, axis=0) / (np.sum(weights, axis=0) + 1e-6)

    med = np.median(stack, axis=0)
    denoised = 0.7 * weighted + 0.3 * med
    return denoised.astype(stabilized_cur.dtype)


def auto_tone_color(bgr, clip_percent=1.0, use_clahe=True):
    """
    Percentile-based contrast stretching + optional CLAHE in luminance.
    """
    img = bgr.astype(np.float32)
    out = img.copy()

    for c in range(3):
        lo = np.percentile(out[..., c], clip_percent)
        hi = np.percentile(out[..., c], 100 - clip_percent)
        if hi <= lo:
            continue
        out[..., c] = (out[..., c] - lo) * (255.0 / (hi - lo))
        out[..., c] = np.clip(out[..., c], 0, 255)

    if use_clahe:
        ycrcb = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb[..., 0] = clahe.apply(ycrcb[..., 0])
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    return out.astype(np.uint8)


def edge_aware_sharpen(bgr, amount=0.6, radius=1.5):
    """
    Unsharp mask with a simple edge mask to avoid ringing in flat regions.
    """
    img = bgr.astype(np.float32)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=radius, sigmaY=radius)
    detail = img - blur

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge = np.clip(np.abs(lap) / (np.percentile(np.abs(lap), 95) + 1e-6), 0, 1)
    edge = cv2.GaussianBlur(edge, (3, 3), 0)[..., None]

    sharpened = img + amount * detail * edge
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def process_video(args):
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input: {args.input}")
    fps, w, h, _ = read_video_props(cap)

    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if args.codec == 'mp4v' else 'avc1'))
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Cannot open output: {args.output}")

    prev_gray = None
    M_smooth = None
    last_stab = None
    last_gray = None
    t = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_bgr = frame_bgr.astype(np.uint8)
        gray = to_gray(frame_bgr)

        if prev_gray is None:
            enhanced = edge_aware_sharpen(auto_tone_color(frame_bgr), amount=args.sharpen, radius=args.radius)
            out.write(enhanced)
            prev_gray = gray
            last_stab = frame_bgr.copy()
            last_gray = prev_gray.copy()
            t += 1
            continue

        # (1) Global motion estimation and stabilization
        M, okM = estimate_global_transform(prev_gray, gray, mode=args.mode)
        if not okM:
            M = np.eye(2, 3, dtype=np.float32)
        M_smooth = smooth_transform(M, M_smooth, alpha=args.stab_alpha)
        cur_stab = warp_affine(frame_bgr, M_smooth, (h, w))
        cur_stab_gray = to_gray(cur_stab)

        # (2) Local motion estimation (optical flow) for MC denoise
        if last_gray is None:
            denoised = cur_stab.copy()
        else:
            flow_prev = optical_flow(last_gray, cur_stab_gray, algo=args.flow)
            flow_bwd = optical_flow(cur_stab_gray, last_gray, algo=args.flow)
            fb_mask = forward_backward_consistency(flow_prev, flow_bwd)

            denoised = motion_compensated_denoise(last_stab, cur_stab, cur_stab,
                                                  flow_prev, np.zeros_like(flow_prev),
                                                  fb_mask, np.ones_like(fb_mask))

        # (3) Auto tone & color
        toned = auto_tone_color(denoised, clip_percent=args.clip, use_clahe=not args.no_clahe)

        # (4) Edge-aware sharpening
        sharp = edge_aware_sharpen(toned, amount=args.sharpen, radius=args.radius)

        out.write(sharp)

        # Update state
        last_stab = cur_stab
        last_gray = cur_stab_gray
        prev_gray = gray
        t += 1
        if args.max_frames and t >= args.max_frames:
            break

    cap.release()
    out.release()
    print(f"[Done] Wrote: {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="Soupliotis/Anandan Method for Automated, Motion-Aware Video Enhancement (OpenCV)")
    p.add_argument('--input', required=True, help='Input video file')
    p.add_argument('--output', required=True, help='Output video file (e.g., out.mp4)')
    p.add_argument('--mode', default='ecc', choices=['ecc', 'orb'], help='Global motion estimation method')
    p.add_argument('--flow', default='farneback', choices=['farneback', 'dis', 'tvl1'], help='Optical flow algorithm')
    p.add_argument('--stab-alpha', type=float, default=0.9, help='Stabilization smoothing alpha (EMA, higher = smoother)')
    p.add_argument('--clip', type=float, default=1.0, help='Percentile clipping for auto tone (e.g., 1.0)')
    p.add_argument('--sharpen', type=float, default=0.6, help='Sharpen amount (0..1 recommended)')
    p.add_argument('--radius', type=float, default=1.5, help='Sharpen Gaussian sigma')
    p.add_argument('--codec', default='mp4v', choices=['mp4v', 'avc1'], help='Output codec FourCC')
    p.add_argument('--max-frames', type=int, default=0, help='Limit frames for testing (0 = all)')
    p.add_argument('--no-clahe', action='store_true', help='Disable CLAHE step')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    process_video(args)
