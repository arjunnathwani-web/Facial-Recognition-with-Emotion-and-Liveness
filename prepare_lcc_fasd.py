"""
Prepare LCC FASD folders for the liveness model.

Dataset:
    https://www.kaggle.com/datasets/faber24/lcc-fasd

Usage:
    python prepare_lcc_fasd.py --source-dir "Large Crowdcollected Face Anti-Spoofing Dataset/LCC_FASD" --out data/liveness_data

This creates:
    data/liveness_data/train/real
    data/liveness_data/train/fake
    data/liveness_data/val/real
    data/liveness_data/val/fake
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path


SPLIT_MAP = {
    "LCC_FASD_training": "train",
    "LCC_FASD_development": "val",
}

CLASS_MAP = {
    "real": "real",
    "spoof": "fake",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare LCC FASD folders for liveness training."
    )
    parser.add_argument(
        "--source-dir",
        default="Large Crowdcollected Face Anti-Spoofing Dataset/LCC_FASD",
        help="Path to the LCC_FASD folder.",
    )
    parser.add_argument(
        "--out",
        default="data/liveness_data",
        help="Output folder for train/val real/fake folders.",
    )
    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="Optional max number of images to copy per split/class for quick tests.",
    )
    return parser.parse_args()


def copy_images(source_dir, out_dir, sample_per_class=None):
    counters = Counter()
    for source_split, target_split in SPLIT_MAP.items():
        split_dir = source_dir / source_split
        if not split_dir.exists():
            raise FileNotFoundError(f"Could not find LCC FASD split folder: {split_dir}")

        for source_class, target_class in CLASS_MAP.items():
            class_dir = split_dir / source_class
            if not class_dir.exists():
                raise FileNotFoundError(f"Could not find LCC FASD class folder: {class_dir}")

            target_dir = out_dir / target_split / target_class
            target_dir.mkdir(parents=True, exist_ok=True)

            image_paths = [
                p for p in sorted(class_dir.iterdir())
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if sample_per_class is not None:
                image_paths = image_paths[:sample_per_class]

            for image_path in image_paths:
                target_path = target_dir / image_path.name
                if not target_path.exists():
                    shutil.copy2(image_path, target_path)
                counters[(target_split, target_class)] += 1

    return counters


def main():
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out)

    counters = copy_images(source_dir, out_dir, args.sample_per_class)
    print("Done. Copied LCC FASD folders:")
    for (split, label), count in sorted(counters.items()):
        print(f"  {split}/{label}: {count}")


if __name__ == "__main__":
    main()
