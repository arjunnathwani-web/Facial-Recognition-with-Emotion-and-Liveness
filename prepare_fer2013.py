"""
Convert FER2013 CSV data into the folder layout expected by EmotionDataset.

Usage:
    python prepare_fer2013.py --csv data/fer2013.csv --out data/emotion_data

Expected FER2013 columns:
    emotion,pixels,Usage
"""

import argparse
import csv
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
        default="data/fer2013.csv",
        help="Path to fer2013.csv downloaded from Kaggle.",
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
    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find FER2013 CSV: {csv_path}")

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
