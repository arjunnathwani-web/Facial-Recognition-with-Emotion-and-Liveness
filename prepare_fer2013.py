"""
Convert FER2013 data into the folder layout expected by EmotionDataset.

Usage:
    # Folder dataset from https://www.kaggle.com/datasets/msambare/fer2013
    python prepare_fer2013.py --source-dir FER-2013 --out data/emotion_data

    # CSV-style FER2013 dataset
    python prepare_fer2013.py --csv data/fer2013.csv --out data/emotion_data

Expected FER2013 columns:
    emotion,pixels,Usage
"""

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


FER2013_LABELS = {
    "0": "Angry",
    "1": "Disgust",
    "2": "Fear",
    "3": "Happy",
    "4": "Sad",
    "5": "Surprise",
    "6": "Neutral",
}

USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare FER2013 folders for the emotion model."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to fer2013.csv downloaded from Kaggle.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Path to FER2013 folder dataset containing train/ and test/ folders.",
    )
    parser.add_argument(
        "--out",
        default="data/emotion_data",
        help="Output folder for train/val/test emotion image folders.",
    )
    parser.add_argument(
        "--private-test-to-val",
        action="store_true",
        help="Put PrivateTest rows into val instead of a separate test split.",
    )
    return parser.parse_args()


def copy_folder_dataset(source_dir, out_dir):
    split_map = {
        "train": "train",
        "test": "val",
    }
    counters = Counter()

    for source_split, target_split in split_map.items():
        split_dir = source_dir / source_split
        if not split_dir.exists():
            raise FileNotFoundError(f"Could not find FER2013 split folder: {split_dir}")

        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            target_dir = out_dir / target_split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    target_path = target_dir / image_path.name
                    if not target_path.exists():
                        shutil.copy2(image_path, target_path)
                    counters[(target_split, class_dir.name)] += 1

    return counters


def save_image(row, out_dir, counters, private_test_to_val=False):
    label = FER2013_LABELS[row["emotion"]]
    usage = row["Usage"]

    if private_test_to_val and usage == "PrivateTest":
        split = "val"
    else:
        split = USAGE_TO_SPLIT.get(usage)

    if split is None:
        raise ValueError(f"Unknown FER2013 Usage value: {usage}")

    pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
    if pixels.size != 48 * 48:
        raise ValueError(f"Expected 2304 pixels, got {pixels.size}")

    folder = out_dir / split / label
    folder.mkdir(parents=True, exist_ok=True)

    index = counters[(split, label)]
    counters[(split, label)] += 1
    image = Image.fromarray(pixels.reshape(48, 48), mode="L")
    image.save(folder / f"{label}_{index:05d}.png")


def main():
    args = parse_args()
    out_dir = Path(args.out)

    if args.source_dir:
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            raise FileNotFoundError(f"Could not find FER2013 source folder: {source_dir}")
        counters = copy_folder_dataset(source_dir, out_dir)
        print("Done. Copied FER2013 folders:")
        for (split, label), count in sorted(counters.items()):
            print(f"  {split}/{label}: {count}")
        return

    csv_path = Path(args.csv or "data/fer2013.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find FER2013 CSV: {csv_path}. "
            "Use --source-dir for the folder-based Kaggle dataset."
        )

    counters = Counter()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"emotion", "pixels", "Usage"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            save_image(
                row,
                out_dir,
                counters,
                private_test_to_val=args.private_test_to_val,
            )

    print("Done. Created FER2013 folders:")
    for (split, label), count in sorted(counters.items()):
        print(f"  {split}/{label}: {count}")


if __name__ == "__main__":
    main()
