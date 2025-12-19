"""Simple dataset augmenter: reads `data/raw/*` and writes augmented copies to `data/processed/augmented/<class>/`.

This helps create more training data for quick experiments. It performs random flips, rotations,
brightness/contrast changes and small crops.
"""
import os
from pathlib import Path
import cv2
import numpy as np


def augment_image(img):
    # img: RGB uint8
    img = img.copy()
    h, w = img.shape[:2]

    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # Random rotation
    angle = np.random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random scale and crop
    scale = np.random.uniform(0.9, 1.1)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img, (new_w, new_h))
    # center crop/pad back to original
    if new_w >= w:
        x1 = (new_w - w)//2
        img = img[:, x1:x1+w]
    else:
        pad = ((0,0),( (w-new_w)//2, w-new_w-(w-new_w)//2 ), (0,0))
        img = np.pad(img, pad, mode='reflect')

    # Random brightness/contrast
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.uniform(-20, 20)
    img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)

    # Small gaussian blur occasionally
    if np.random.rand() > 0.8:
        k = np.random.choice([3,5])
        img = cv2.GaussianBlur(img, (k,k), 0)

    return img


def run_augment(source_dir='data/raw', out_dir='data/processed/augmented', target_per_class=200):
    src = Path(source_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for cls in sorted([p.name for p in src.iterdir() if p.is_dir()]):
        in_cls = src/cls
        out_cls = out/cls
        out_cls.mkdir(parents=True, exist_ok=True)

        imgs = [p for p in in_cls.glob('*') if p.suffix.lower() in ('.jpg','.png','.jpeg')]
        print(f"Class {cls}: {len(imgs)} source images")
        if not imgs:
            continue

        # copy originals
        for i, p in enumerate(imgs):
            dst = out_cls / f"orig_{i:04d}{p.suffix}"
            if not dst.exists():
                img = cv2.imread(str(p))
                cv2.imwrite(str(dst), img)

        # generate augmentations until target reached
        idx = len(list(out_cls.glob('*')))
        while idx < target_per_class:
            src_img = cv2.imread(str(np.random.choice(imgs)))
            if src_img is None:
                continue
            aug = augment_image(src_img)
            dst = out_cls / f"aug_{idx:05d}.jpg"
            cv2.imwrite(str(dst), aug)
            idx += 1

        print(f" -> Augmented to {target_per_class} images in {out_cls}")


if __name__ == '__main__':
    run_augment()
