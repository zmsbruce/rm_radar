/**
 * @file detector.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the declaration of the Detector class and functions
 * of object detection using NVIDIA CUDA and TensorRT.
 * @date 2024-03-06
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "common.h"
#include "detection.h"
#include "preparam.h"
#include "robot/robot.h"
#include "tensor.h"
#include "trt_logger.h"

namespace radar {

namespace detect {

__global__ void resizeKernel(const unsigned char* src, unsigned char* dst,
                             int channels, int src_w, int src_h, int dst_w,
                             int dst_h);

__global__ void copyMakeBorderKernel(const unsigned char* src,
                                     unsigned char* dst, int channels,
                                     int src_w, int src_h, int top, int bottom,
                                     int left, int right);

__global__ void blobKernel(const unsigned char* src, float* dst, int width,
                           int height, int channels, float scale);

__global__ void transposeKernel(const float* src, float* dst, int rows,
                                int cols);

__global__ void decodeKernel(const float* src, float* dst, int channels,
                             int anchors, int classes);

__global__ void NMSKernel(float* dev, float nms_thresh, float score_thresh,
                          int anchors);

}  // namespace detect

/**
 * @brief Checks if T is cv::Mat or a container of cv::Mat.
 *
 * This concept defines two valid scenarios for the given type T:
 * - T is a cv::Mat object.
 * - T is a container (such as std::vector) that supports .begin() and
 * .end() iterators, and its value_type is cv::Mat.
 *
 * @tparam T The type to check against the ImageOrImages concept.
 */
template <typename T>
concept ImageOrImages =
    std::is_same_v<std::decay_t<T>, cv::Mat> || requires(T t) {
        typename std::decay_t<T>::value_type;
        { t.begin() } -> std::same_as<typename std::decay_t<T>::iterator>;
        { t.end() } -> std::same_as<typename std::decay_t<T>::iterator>;
        std::is_same_v<typename std::decay_t<T>::value_type, cv::Mat>;
    };

/**
 * @brief The Detector class provides functionality for object detection
 * using a pre-trained model.
 *
 */
class Detector {
   public:
    Detector() = delete;
    explicit Detector(std::string_view engine_path, int classes,
                      int max_batch_size,
                      std::optional<int> opt_batch_size = std::nullopt,
                      size_t image_size = 1 << 24, float nms_thresh = 0.65,
                      float conf_thresh = 0.25, int input_width = 640,
                      int input_height = 640,
                      std::string_view input_name = "images",
                      int input_channels = 3, int opt_level = 3);
    ~Detector();

    /**
     * @brief Performs detection on an input image or images.
     *
     * This function processes an input, which can be either a single image or a
     * collection of images, and performs object detection. The detection
     * pipeline consists of preprocessing the input, running inference using
     * TensorRT's context (`context_`), and postprocessing to obtain the final
     * detection results. If the input is not a single `cv::Mat`, the CUDA
     * stream is synchronized after inference to ensure correct operation order.
     *
     * @tparam ImageOrImages A type satisfying a single `cv::Mat` or a container
     * of `cv::Mat` objects.
     * @param input A universal reference to the input image or images to be
     * processed.
     * @return `std::vector<Detection>` or `std::vector<std::vector<Detection>>`
     * A vector of detections if input is a single image, or a batch-sized
     * vector of vectors containing the detections.
     * @exception `noexcept` This function is declared to not throw exceptions.
     * However, if the function encounters a problem in CUDA checking, it will
     * call `std::abort()` directly.
     */
    template <ImageOrImages T>
    auto detect(T&& input) noexcept {
        spdlog::debug("Starting detection.");

        if constexpr (std::is_same_v<std::decay_t<T>, cv::Mat>) {
            if (input.empty()) {
                spdlog::error(
                    "Input image is empty, will skip detection and return "
                    "empty result.");
                return std::vector<Detection>();
            }
        } else {
            if (std::ranges::any_of(
                    input, [](const auto& image) { return image.empty(); })) {
                spdlog::error(
                    "Empty image exists in batch, will skip detection and "
                    "return empty result.");
                return std::vector<std::vector<Detection>>();
            }
        }

        std::vector<detect::PreParam> pparams{
            preprocess(std::forward<T>(input))};
        spdlog::debug(
            "Preprocessing completed. Number of preprocessed parameters: {}",
            pparams.size());

        spdlog::debug("Enqueuing inference context.");
        context_->enqueueV3(streams_[0]);
        if constexpr (!std::is_same_v<std::decay_t<T>, cv::Mat>) {
            spdlog::debug("Input is a batch, synchronizing CUDA stream.");
            CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(streams_[0]));
            spdlog::debug("CUDA stream synchronized.");
        }

        spdlog::debug("Postprocessing detection results.");
        auto detections{postprocess(pparams)};

        if constexpr (std::is_same_v<std::decay_t<T>, cv::Mat>) {
            spdlog::debug(
                "Returning single detection result for cv::Mat input.");
            return detections[0];
        } else {
            spdlog::debug("Returning detection results for batch input.");
            return detections;
        }
    }

   private:
    std::vector<detect::PreParam> preprocess(const cv::Mat& image) noexcept;
    std::vector<detect::PreParam> preprocess(
        const std::span<cv::Mat> images) noexcept;
    std::vector<std::vector<Detection>> postprocess(
        std::span<detect::PreParam> pparams) noexcept;
    std::pair<std::shared_ptr<char[]>, size_t> serializeEngine(
        std::string_view onnx_path, int opt_batch_size, int max_batch_size,
        int opt_level);
    void restoreDetection(Detection& detection,
                          const detect::PreParam& pparam) const noexcept;
    static void writeToFile(std::span<const char> data, std::string_view path);
    static std::pair<std::shared_ptr<char[]>, size_t> loadFromFile(
        std::string_view path);
    std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};
    int input_width_, input_height_, input_channels_;
    std::string_view input_name_;
    int classes_;
    float nms_thresh_, conf_thresh_;
    detect::Logger logger_;
    detect::Tensor input_tensor_, output_tensor_;
    std::vector<cudaStream_t> streams_;
    unsigned char* image_ptr_{nullptr};
    unsigned char* dev_resize_ptr_{nullptr};
    unsigned char* dev_border_ptr_{nullptr};
    float* dev_transpose_ptr_{nullptr};
    float* dev_decode_ptr_{nullptr};
    float* nms_ptr_{nullptr};
    int output_channels_{0};
    int output_anchors_{0};
    int batch_size_{0};
};

class RobotDetector {
   public:
    explicit RobotDetector(
        std::string_view car_engine_path, std::string_view armor_engine_path,
        int armor_classes, int max_cars, int opt_cars, float iou_thresh = 0.75f,
        float car_nms_thresh = 0.65f, float car_conf_thresh = 0.25f,
        float armor_nms_thresh = 0.65f, float armor_conf_thresh = 0.50f,
        size_t image_size = 1 << 24, float input_width = 640,
        float input_height = 640, std::string_view input_name = "images",
        int input_channels = 3, int opt_level = 5);

    RobotDetector() = delete;

    std::vector<Robot> detect(const cv::Mat& image) noexcept;

   private:
    float iou_thresh_;
    std::unique_ptr<Detector> car_detector_, armor_detector_;
    std::vector<cv::Mat> car_images_;
};

}  // namespace radar