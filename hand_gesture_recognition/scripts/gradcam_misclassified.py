"""Find misclassified images and save Grad-CAM overlays.
Usage:
  python scripts/gradcam_misclassified.py --model models/efficient_finetuned_roi.h5 --data-dir data/processed/roi --out-dir predictions/gradcam --max-per-pair 8
"""
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
from src.visualization import gradcam_for_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out-dir', default='predictions/gradcam')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--max-per-pair', type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(str(model_path))

    # Build class name list from subdirectories
    class_dirs = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    name_to_idx = {n: i for i, n in enumerate(class_dirs)}

    # Iterate files and collect misclassifications by (true, pred)
    mis = {}
    for true_name in class_dirs:
        for img_path in (data_dir / true_name).glob('*'):
            try:
                res = gradcam_for_image(model, str(img_path), out_path=None, target_size=(args.image_size, args.image_size), top_k=3)
            except Exception:
                # Fallback: try loading as numpy array via TF preprocessing
                continue
            top_pred = res['top_predictions'][0][0]
            pred_name = class_dirs[top_pred]
            if pred_name != true_name:
                key = (true_name, pred_name)
                mis.setdefault(key, []).append((img_path, res))

    # For each mispair, save up to max_per_pair overlays
    saved = []
    for (true_name, pred_name), examples in mis.items():
        examples = examples[: args.max_per_pair]
        for i, (img_path, res) in enumerate(examples, start=1):
            safe_name = f"gradcam_{true_name}_to_{pred_name}_{i}.png"
            out_path = out_dir / safe_name
            # Save overlay image (res['overlay'] is RGB numpy)
            import cv2
            overlay = res['overlay']
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_path), overlay_bgr)
            saved.append(out_path)

    if not saved:
        print('No misclassified images found.')
    else:
        print(f'Saved {len(saved)} Grad-CAM overlays to {out_dir}')
        for p in saved:
            print(p)


if __name__ == '__main__':
    main()
