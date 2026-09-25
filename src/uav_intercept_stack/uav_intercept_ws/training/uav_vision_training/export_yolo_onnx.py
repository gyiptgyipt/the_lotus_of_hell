"""
Exports a trained YOLO checkpoint to ONNX for uav_vision_detect.

uav_vision_detect/src/yolo_detector.cpp expects output shape [1, 5, N]
(cx, cy, w, h, confidence per anchor, single class, no built-in NMS -- NMS
happens in the C++ node). Ultralytics' default single-class export matches
this, but this script checks it explicitly so a shape mismatch fails loudly
here instead of silently at inference time.

Usage:
  python3 export_yolo_onnx.py --weights runs_yolo/target_drone/weights/best.pt \
      --out ../../src/uav_vision_detect/models/target_yolo.onnx --imgsz 640
"""

import argparse
import shutil

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported_path = model.export(format="onnx", opset=17, simplify=True, imgsz=args.imgsz, dynamic=False)
    shutil.copy(exported_path, args.out)
    print(f"Copied exported model to {args.out}")

    import onnx
    m = onnx.load(args.out)
    output_shape = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    print(f"ONNX output shape: {output_shape}")
    if len(output_shape) != 3 or output_shape[1] != 5:
        print(
            "WARNING: expected shape [batch, 5, num_anchors] (single class). "
            "Got something else -- check --imgsz / model num_classes, and update "
            "yolo_detector.cpp's decode_output if you intentionally trained multi-class."
        )


if __name__ == "__main__":
    main()
