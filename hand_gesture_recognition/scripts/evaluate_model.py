"""
Evaluate a saved Keras model on a directory-structured dataset and save a confusion matrix.
Usage:
python hand_gesture_recognition/scripts/evaluate_model.py --model hand_gesture_recognition/models/efficient_finetuned.h5 --data-dir hand_gesture_recognition/data/raw --batch-size 32
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--batch-size', type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    img_size = (args.image_size, args.image_size)
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_dir), image_size=img_size, batch_size=args.batch_size, shuffle=False
    )
    class_names = ds.class_names
    num_classes = len(class_names)

    print('Loading model:', model_path)
    model = tf.keras.models.load_model(str(model_path))

    print('Predicting...')
    y_true = []
    y_pred = []
    for x, y in ds:
        preds = model.predict(x)
        preds_idx = np.argmax(preds, axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds_idx.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print('\nConfusion matrix:')
    print(cm)

    out_png = model_path.parent / (model_path.stem + '_confusion.png')
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion matrix')
    plt.tight_layout()
    plt.savefig(out_png)
    print(f'Saved confusion matrix to {out_png}')

if __name__ == '__main__':
    main()
