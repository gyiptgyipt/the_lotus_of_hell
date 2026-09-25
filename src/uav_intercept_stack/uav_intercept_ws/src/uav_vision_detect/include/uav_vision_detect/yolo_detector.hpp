#ifndef UAV_VISION_DETECT__YOLO_DETECTOR_HPP_
#define UAV_VISION_DETECT__YOLO_DETECTOR_HPP_

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

namespace uav_vision_detect
{

struct Detection
{
  cv::Rect2f box;   // pixel coords in the ORIGINAL image (post-letterbox undone)
  float confidence{0.0f};
};

// Wraps a single-class Ultralytics-exported YOLO ONNX model (input
// 1x3xHxW float, output 1x5xN = [cx, cy, w, h, conf] per anchor, model-input
// pixel coordinates, no built-in NMS). If you export a different model
// shape, this is the class to adjust.
class YoloDetector
{
public:
  YoloDetector(const std::string & model_path, int input_size, float conf_threshold, float nms_iou_threshold);

  // Runs the full pipeline (letterbox -> inference -> decode -> NMS) and
  // returns detections sorted by descending confidence, in the coordinate
  // frame of `image` as passed in (no letterbox padding in the output).
  std::vector<Detection> detect(const cv::Mat & image);

private:
  struct LetterboxInfo
  {
    float scale;
    int pad_x;
    int pad_y;
  };

  cv::Mat letterbox(const cv::Mat & image, LetterboxInfo & info_out) const;
  std::vector<Detection> decode_output(const float * output, int64_t num_anchors) const;
  static std::vector<Detection> non_max_suppression(std::vector<Detection> dets, float iou_threshold);

  Ort::Env ort_env_;
  std::unique_ptr<Ort::Session> ort_session_;
  Ort::MemoryInfo memory_info_;
  std::string input_name_;
  std::string output_name_;

  int input_size_;
  float conf_threshold_;
  float nms_iou_threshold_;
};

}  // namespace uav_vision_detect

#endif  // UAV_VISION_DETECT__YOLO_DETECTOR_HPP_
