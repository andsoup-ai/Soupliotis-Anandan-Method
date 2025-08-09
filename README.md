# Soupliotis/Anandan Method for Automated, Motion-Aware Video Enhancement

This is a reference OpenCV-based implementation of the **Soupliotis/Anandan Method** described in Microsoft patents by Andreas Soupliotis and Padmanabhan Anandan.

It performs:
- **Global motion estimation & stabilization**
- **Local motion estimation (optical flow) & motion-compensated denoising**
- **Automated tone & color correction**
- **Edge-aware sharpening**

## Installation
```bash
pip install opencv-python numpy
```

## Basic usage
```bash
python soupliotis_opencv_pipeline.py --input in.mp4 --output out.mp4
```

## Preset Recommendations

### 1. Handheld Smartphone Footage
- Best for: Family videos, walking shots, selfie videos
```bash
python soupliotis_opencv_pipeline.py --input phone.mp4 --output phone_enhanced.mp4     --mode ecc --flow farneback --stab-alpha 0.95 --clip 1.0 --sharpen 0.6 --radius 1.5
```

### 2. Dashcam Footage
- Best for: Road trips, vehicle-mounted video
```bash
python soupliotis_opencv_pipeline.py --input dashcam.mp4 --output dashcam_enhanced.mp4     --mode orb --flow dis --stab-alpha 0.9 --clip 0.5 --sharpen 0.5 --radius 1.0 --no-clahe
```

### 3. Drone Aerial Video
- Best for: Sweeping, wide shots with gradual motion
```bash
python soupliotis_opencv_pipeline.py --input drone.mp4 --output drone_enhanced.mp4     --mode ecc --flow tvl1 --stab-alpha 0.98 --clip 1.0 --sharpen 0.4 --radius 2.0
```

### 4. Low-Light Video
- Best for: Night scenes, dimly lit interiors
```bash
python soupliotis_opencv_pipeline.py --input lowlight.mp4 --output lowlight_enhanced.mp4     --mode ecc --flow tvl1 --stab-alpha 0.95 --clip 2.0 --sharpen 0.5 --radius 1.2
```

## Notes
- **Global motion mode**: `ecc` is best for consistent lighting; `orb` can handle lighting changes better.
- **Flow mode**: `farneback` is balanced; `dis` is faster; `tvl1` is more accurate for challenging motion.
- **CLAHE**: Enabled by default; disable with `--no-clahe` if it causes over-enhancement.

## Reference Patents
- US 2004/0001705 A1
- US 2006/0290821 A1
- US 7,746,382 B2
