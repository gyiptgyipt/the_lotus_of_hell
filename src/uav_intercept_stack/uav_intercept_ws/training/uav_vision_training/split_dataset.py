"""
Splits dataset_raw/{images,labels} into a train/val layout ultralytics
expects, and writes dataset.yaml.

Usage:
  python3 split_dataset.py --src dataset_raw --dst dataset --val-frac 0.1
"""

import argparse
import os
import random
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="dataset_raw")
    parser.add_argument("--dst", type=str, default="dataset")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = os.path.join(args.src, "images")
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(".jpg"))
    random.shuffle(stems)

    n_val = max(1, int(len(stems) * args.val_frac))
    val_stems = set(stems[:n_val])

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.dst, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.dst, "labels", split), exist_ok=True)

    for stem in stems:
        split = "val" if stem in val_stems else "train"
        shutil.copy(
            os.path.join(args.src, "images", f"{stem}.jpg"),
            os.path.join(args.dst, "images", split, f"{stem}.jpg"),
        )
        shutil.copy(
            os.path.join(args.src, "labels", f"{stem}.txt"),
            os.path.join(args.dst, "labels", split, f"{stem}.txt"),
        )

    yaml_path = os.path.join(args.dst, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(args.dst)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: target_drone\n")

    print(f"{len(stems) - n_val} train / {n_val} val images. Wrote {yaml_path}")


if __name__ == "__main__":
    main()
