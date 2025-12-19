"""
Webcam capture tool to build a dataset organized as:
  hand_gesture_recognition/data/raw/<class>/*.jpg

Usage (from project root):
python hand_gesture_recognition/scripts/capture_dataset.py --output hand_gesture_recognition/data/raw --classes palm fist thumbs_up victory ok --samples 200

Controls (interactive window):
 - SPACE: capture current frame and save to current class
 - n: move to next class
 - p: move to previous class
 - q or ESC: quit
 - c: toggle continuous capture mode (auto capture every N frames)

This script is intended for interactive use; run it on the machine with the webcam you want to record from.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import time


def ensure_dirs(output: Path, classes: list[str]):
    for cls in classes:
        d = output / cls
        d.mkdir(parents=True, exist_ok=True)


def capture_loop(output: Path, classes: list[str], samples_per_class: int, target_size: tuple[int, int]):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0).")

    idx = 0
    class_idx = 0
    counts = {c: len(list((output / c).glob('*.jpg'))) for c in classes}
    print("Starting capture. Classes:", classes)
    print("Initial counts:", counts)
    continuous = False
    cont_interval = 0.5  # seconds
    last_cont_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed; retrying...")
                time.sleep(0.1)
                continue

            disp = frame.copy()
            h, w = disp.shape[:2]
            # overlay current state
            cv2.putText(disp, f"Class [{class_idx}/{len(classes)-1}]: {classes[class_idx]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.putText(disp, f"Saved: {counts[classes[class_idx]]}/{samples_per_class}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(disp, "Controls: SPACE capture | n next | p prev | c toggle cont | q quit", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow('Capture', disp)
            key = cv2.waitKey(1) & 0xFF

            if continuous and (time.time() - last_cont_time) >= cont_interval:
                key = ord(' ')
                last_cont_time = time.time()

            if key == ord('q') or key == 27:  # ESC
                print("Exiting capture.")
                break
            elif key == ord('n'):
                class_idx = (class_idx + 1) % len(classes)
                print(f"Switched to class: {classes[class_idx]}")
            elif key == ord('p'):
                class_idx = (class_idx - 1) % len(classes)
                print(f"Switched to class: {classes[class_idx]}")
            elif key == ord('c'):
                continuous = not continuous
                print("Continuous mode:", continuous)
            elif key == ord(' '):
                cls = classes[class_idx]
                count = counts.get(cls, 0)
                if count >= samples_per_class:
                    print(f"Class {cls} already has {count} samples (target {samples_per_class}).")
                else:
                    img = cv2.resize(frame, target_size)
                    out_path = output / cls / f"{int(time.time()*1000)}.jpg"
                    cv2.imwrite(str(out_path), img)
                    counts[cls] = counts.get(cls, 0) + 1
                    print(f"Saved {out_path} ({counts[cls]}/{samples_per_class})")

            # optionally exit once all classes have enough samples
            if all(counts.get(c, 0) >= samples_per_class for c in classes):
                print("All classes reached target sample count. Exiting.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="hand_gesture_recognition/data/raw", help="Output folder for class subfolders")
    p.add_argument("--classes", nargs='+', default=["palm","fist","thumbs_up","victory","ok"], help="List of class folder names")
    p.add_argument("--samples", type=int, default=200, help="Target samples per class")
    p.add_argument("--width", type=int, default=224, help="Saved image width")
    p.add_argument("--height", type=int, default=224, help="Saved image height")
    return p.parse_args()


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    classes = args.classes
    ensure_dirs(output, classes)
    capture_loop(output, classes, args.samples, (args.width, args.height))


if __name__ == "__main__":
    main()
