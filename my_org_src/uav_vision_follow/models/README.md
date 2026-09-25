# Drone detector

`drone_yolo.pt` is a YOLOv11x checkpoint from
[`doguilmak/Drone-Detection-YOLOv11x`](https://huggingface.co/doguilmak/Drone-Detection-YOLOv11x).
It exposes one class, `drone`, matching `target_class_names` in
`config/follower.yaml`. The model is MIT licensed and was verified against a
live `/uav_2/camera/image` frame before installation.