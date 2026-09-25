#include "uav_vision_detect/yolo_detector.hpp"
#include <algorithm>
#include <numeric>

namespace uav_vision_detect
{

YoloDetector::YoloDetector(
  const std::string & model_path, int input_size, float conf_threshold, float nms_iou_threshold)
: ort_env_(ORT_LOGGING_LEVEL_WARNING, "uav_vision_detect"),
  memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
  input_size_(input_size),
  conf_threshold_(conf_threshold),
  nms_iou_threshold_(nms_iou_threshold)
{
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(2);
  ort_session_ = std::make_unique<Ort::Session>(ort_env_, model_path.c_str(), session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::AllocatedStringPtr input_name_ptr = ort_session_->GetInputNameAllocated(0, allocator);
  Ort::AllocatedStringPtr output_name_ptr = ort_session_->GetOutputNameAllocated(0, allocator);
  input_name_ = input_name_ptr.get();
  output_name_ = output_name_ptr.get();
}

cv::Mat YoloDetector::letterbox(const cv::Mat & image, LetterboxInfo & info_out) const
{
  float scale = std::min(
    static_cast<float>(input_size_) / image.cols,
    static_cast<float>(input_size_) / image.rows);
  int new_w = static_cast<int>(std::round(image.cols * scale));
  int new_h = static_cast<int>(std::round(image.rows * scale));

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(new_w, new_h));

  int pad_x = (input_size_ - new_w) / 2;
  int pad_y = (input_size_ - new_h) / 2;

  cv::Mat padded(input_size_, input_size_, image.type(), cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

  info_out.scale = scale;
  info_out.pad_x = pad_x;
  info_out.pad_y = pad_y;
  return padded;
}

std::vector<Detection> YoloDetector::decode_output(const float * output, int64_t num_anchors) const
{
  // Ultralytics single-class export layout: output[0:4, i] = cx,cy,w,h (model
  // input pixel coords), output[4, i] = confidence. Stored channel-major, so
  // channel c, anchor i is at output[c * num_anchors + i].
  std::vector<Detection> dets;
  dets.reserve(64);
  for (int64_t i = 0; i < num_anchors; ++i) {
    float conf = output[4 * num_anchors + i];
    if (conf < conf_threshold_) {
      continue;
    }
    float cx = output[0 * num_anchors + i];
    float cy = output[1 * num_anchors + i];
    float w = output[2 * num_anchors + i];
    float h = output[3 * num_anchors + i];

    Detection d;
    d.box = cv::Rect2f(cx - w / 2.0f, cy - h / 2.0f, w, h);
    d.confidence = conf;
    dets.push_back(d);
  }
  return dets;
}

std::vector<Detection> YoloDetector::non_max_suppression(std::vector<Detection> dets, float iou_threshold)
{
  std::sort(dets.begin(), dets.end(),
    [](const Detection & a, const Detection & b) { return a.confidence > b.confidence; });

  std::vector<Detection> kept;
  std::vector<bool> suppressed(dets.size(), false);

  for (size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) continue;
    kept.push_back(dets[i]);
    for (size_t j = i + 1; j < dets.size(); ++j) {
      if (suppressed[j]) continue;
      float intersection = (dets[i].box & dets[j].box).area();
      float union_area = dets[i].box.area() + dets[j].box.area() - intersection;
      float iou = union_area > 0.0f ? intersection / union_area : 0.0f;
      if (iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

std::vector<Detection> YoloDetector::detect(const cv::Mat & image)
{
  LetterboxInfo info{};
  cv::Mat letterboxed = letterbox(image, info);

  cv::Mat rgb;
  cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  // HWC -> CHW
  std::vector<float> input_tensor_values(3 * input_size_ * input_size_);
  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; ++c) {
    channels[c] = cv::Mat(input_size_, input_size_, CV_32F,
      input_tensor_values.data() + c * input_size_ * input_size_);
  }
  cv::split(rgb, channels);

  std::array<int64_t, 4> input_shape{1, 3, input_size_, input_size_};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info_, input_tensor_values.data(), input_tensor_values.size(),
    input_shape.data(), input_shape.size());

  const char * input_names[] = {input_name_.c_str()};
  const char * output_names[] = {output_name_.c_str()};

  auto output_tensors = ort_session_->Run(
    Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

  auto shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
  // Expect [1, 5, num_anchors]
  int64_t num_anchors = shape.back();
  const float * out_data = output_tensors.front().GetTensorData<float>();

  std::vector<Detection> dets = decode_output(out_data, num_anchors);
  dets = non_max_suppression(std::move(dets), nms_iou_threshold_);

  // Undo letterbox: map from model-input pixels back to original image pixels.
  for (auto & d : dets) {
    d.box.x = (d.box.x - info.pad_x) / info.scale;
    d.box.y = (d.box.y - info.pad_y) / info.scale;
    d.box.width /= info.scale;
    d.box.height /= info.scale;
  }
  return dets;
}

}  // namespace uav_vision_detect
