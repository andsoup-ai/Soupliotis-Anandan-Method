#!/usr/bin/env python3
# encode_from_frames.py
"""
Encode a folder of numbered frames (e.g., 00000001.png) into an MP4 file.
"""
import os, cv2, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True, help='Folder with frames')
    ap.add_argument('--output', required=True, help='Output MP4')
    ap.add_argument('--fps', type=float, default=30.0, help='Frames per second')
    args = ap.parse_args()

    files = sorted([f for f in os.listdir(args.frames) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    if not files:
        raise SystemExit("No frames found.")
    sample = cv2.imread(os.path.join(args.frames, files[0]))
    H, W = sample.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (W, H))
    for name in files:
        img = cv2.imread(os.path.join(args.frames, name))
        out.write(img)
    out.release()
    print("[Done] Wrote", args.output)

if __name__ == '__main__':
    main()
