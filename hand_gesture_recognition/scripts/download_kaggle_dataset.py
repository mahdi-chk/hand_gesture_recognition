"""
Download Kaggle datasets with a fallback strategy:
- Try to use `kagglehub` (if available) using user-provided snippet style.
- Fallback to the `kaggle` CLI (requires `kaggle` credentials in env or ~/.kaggle/kaggle.json).

This script downloads and extracts the dataset into the project's `data/raw/` folder
(or a custom output directory). It does basic post-checks and prints next-step hints.

Usage (from project root):
python scripts/download_kaggle_dataset.py --dataset <owner/dataset-name> [--output data/raw]

Examples:
python scripts/download_kaggle_dataset.py --dataset ctsrdm/sign-language-mnist --output data/raw
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
import os
import glob
import tempfile
import json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "raw"


def download_with_kagglehub(dataset: str, dest: Path) -> bool:
    try:
        kagglehub = __import__("kagglehub")
    except Exception:
        print("[kagglehub] not installed or import failed")
        return False

    # Try a couple of possible function names to be resilient to different versions
    for fn in ("dataset_download", "download_dataset", "download"):
        if hasattr(kagglehub, fn):
            try:
                print(f"[kagglehub] using {fn} to download {dataset} -> {dest}")
                func = getattr(kagglehub, fn)
                # Many user snippets call dataset_download(dataset, path=...) or similar
                try:
                    func(dataset, path=str(dest), unzip=True)
                except TypeError:
                    # try positional args
                    func(dataset, str(dest))
                return True
            except Exception as e:
                print(f"[kagglehub] {fn} failed: {e}")
                return False

    print("[kagglehub] no known download function found")
    return False


def download_with_kaggle_cli(dataset: str, dest: Path) -> bool:
    # Ensure destination exists
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"]
    print(f"[kaggle CLI] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[kaggle CLI] failed: {e}")
        return False
    except FileNotFoundError:
        print("[kaggle CLI] 'kaggle' module/CLI not found. Install the kaggle package and configure credentials.")
        return False


def extract_zip_files(dest: Path) -> None:
    # If there are any zip files present, extract them in-place
    for z in dest.glob("*.zip"):
        try:
            print(f"[extract] extracting {z}")
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(dest)
            z.unlink()
        except Exception as e:
            print(f"[extract] failed to extract {z}: {e}")


def find_image_files(dest: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")
    files = []
    for ext in exts:
        files.extend(dest.rglob(ext))
    return files


def summarize_and_hint(dest: Path) -> None:
    files = find_image_files(dest)
    if not files:
        print(f"[result] No image files detected under {dest}")
        # special-case common Kaggle datasets delivered as CSV
        csvs = list(dest.rglob("*.csv"))
        if csvs:
            print("[hint] Found CSV files. If this is a label/image CSV dataset (e.g. sign-language-mnist), run your preprocessing to convert CSV -> images.")
        else:
            print("[hint] Dataset downloaded but contains no images. Inspect the extracted folders to map files into `data/raw/<class>/`.")
        return

    # count images by top-level folder (relative to dest)
    top_counts: dict[str, int] = {}
    for f in files:
        try:
            rel = f.relative_to(dest)
            # top folder is the first part if present
            parts = rel.parts
            if len(parts) > 1:
                top = parts[0]
            else:
                top = "(root)"
            top_counts[top] = top_counts.get(top, 0) + 1
        except Exception:
            top_counts["(unknown)"] = top_counts.get("(unknown)", 0) + 1

    print("[result] Detected image files. Top-level folder counts:")
    for k, v in sorted(top_counts.items(), key=lambda kv: -kv[1]):
        print(f"  - {k}: {v}")

    print("")
    print("If your dataset is already organized with class subfolders (e.g. 'palm/', 'fist/'), it's ready to use in `data/raw/`.")
    print("If images are all in one folder or in CSVs, you'll need to run a conversion/preprocessing step to map samples into `data/raw/<class>/`.")


def safe_resolve_output(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Kaggle dataset identifier, e.g. 'owner/dataset-name'")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Output folder (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--prefer-kagglehub", action="store_true", help="Prefer using kagglehub if available")
    args = parser.parse_args()

    dest = safe_resolve_output(args.output)
    print(f"[info] Destination: {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    ok = False
    if args.prefer_kagglehub:
        print("[info] Attempting download with kagglehub first (per user request)")
        ok = download_with_kagglehub(args.dataset, dest)
        if not ok:
            print("[info] kagglehub attempt failed; falling back to kaggle CLI")
            ok = download_with_kaggle_cli(args.dataset, dest)
    else:
        # try kagglehub first but accept failure
        ok = download_with_kagglehub(args.dataset, dest)
        if not ok:
            print("[info] Falling back to kaggle CLI")
            ok = download_with_kaggle_cli(args.dataset, dest)

    if not ok:
        print("[error] All download attempts failed. Please ensure you have credentials configured and the required packages installed.")
        print("[hint] For the kaggle CLI, install with `pip install kaggle` and place your kaggle.json under ~/.kaggle/kaggle.json or set env vars KAGGLE_USERNAME and KAGGLE_KEY.")
        sys.exit(2)

    # Try to extract zip files and then summarize
    extract_zip_files(dest)
    summarize_and_hint(dest)

    print("[done] Download step complete. Review the output and run preprocessing if required.")


if __name__ == "__main__":
    main()
