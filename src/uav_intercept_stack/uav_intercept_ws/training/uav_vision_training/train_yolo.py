"""
Trains a single-class YOLO detector on the auto-labeled dataset.

Usage:
  python3 train_yolo.py --data dataset/dataset.yaml --epochs 100 --model yolo11n.pt
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                         help="pretrained checkpoint to fine-tune from (downloads automatically)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="runs_yolo")
    parser.add_argument("--name", type=str, default="target_drone")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        single_cls=True,
    )


if __name__ == "__main__":
    main()
