"""
Crop hand ROI for every image in `data/raw/<class>/` using a simple skin-color mask + largest contour
and save processed images to `data/processed/roi/<class>/`.

Usage:
python hand_gesture_recognition/scripts/preprocess_roi.py --input hand_gesture_recognition/data/raw --output hand_gesture_recognition/data/processed/roi --size 224
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np


def detect_hand_bbox(img: np.ndarray) -> tuple[int,int,int,int] | None:
    # Convert to HSV and threshold for skin-like colors (works reasonably for many skin tones)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # broad range for skin color - may need adjustment per environment
    lower = np.array([0, 15, 60], dtype=np.uint8)
    upper = np.array([25, 200, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # pick the largest contour by area
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 500:  # too small
        return None
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h


def expand_bbox(x, y, w, h, img_w, img_h, pad=0.25):
    px = int(w * pad)
    py = int(h * pad)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(img_w, x + w + px)
    y2 = min(img_h, y + h + py)
    return x1, y1, x2 - x1, y2 - y1


def process_and_save(img_path: Path, out_path: Path, size: int):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    bbox = detect_hand_bbox(img)
    if bbox is not None:
        x, y, bw, bh = bbox
        x, y, bw, bh = expand_bbox(x, y, bw, bh, w, h, pad=0.3)
        crop = img[y:y+bh, x:x+bw]
    else:
        # fallback: center-crop square
        s = min(h, w)
        cx = w // 2
        cy = h // 2
        x1 = max(0, cx - s//2)
        y1 = max(0, cy - s//2)
        crop = img[y1:y1+s, x1:x1+s]

    resized = cv2.resize(crop, (size, size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), resized)
    return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='hand_gesture_recognition/data/raw')
    p.add_argument('--output', default='hand_gesture_recognition/data/processed/roi')
    p.add_argument('--size', type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        raise FileNotFoundError(f'Input folder not found: {inp}')

    total = 0
    saved = 0
    for cls_dir in sorted([d for d in inp.iterdir() if d.is_dir()]):
        for img_path in sorted(cls_dir.glob('*')):
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            rel = img_path.relative_to(inp)
            out_path = out / rel
            ok = process_and_save(img_path, out_path, args.size)
            total += 1
            if ok:
                saved += 1
        print(f'Processed class {cls_dir.name}')
    print(f'Processed {total} images, saved {saved} preprocessed images to {out}')

if __name__ == '__main__':
    main()
