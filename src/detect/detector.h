/**
 * @file detector.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the declaration of several classes and functions
 * related to object detection using NVIDIA TensorRT.
 * @date 2024-03-06
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include "detection.h"
#include "utils.h"

namespace radar {

/**
 * @brief Ensures that a CUDA call returns success.
 *
 * This macro wraps a CUDA API function call and checks its return value. If the
 * return value indicates that an error has occurred, it throws a
 * `std::runtime_error` with a message that includes the CUDA error string.
 *
 * @param call The CUDA API function call to check.
 * @throws std::runtime_error If the CUDA API call does not return cudaSuccess.
 * @note This macro is intended for use in functions that allow exceptions.
 */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        const cudaError_t error_code = call;                        \
        if (error_code != cudaSuccess) {                            \
            std::stringstream ss;                                   \
            ss << "CUDA Error: " << cudaGetErrorString(error_code); \
            throw std::runtime_error(ss.str());                     \
        }                                                           \
    } while (0)

/**
 * @brief Ensures that a CUDA call returns success without throwing exceptions.
 *
 * Similar to `CUDA_CHECK`, this macro wraps a CUDA API function call and checks
 * its return value. However, if the return value indicates that an error has
 * occurred, it writes an error message to `std::cerr` and then calls
 * `std::abort` to terminate the program.
 *
 * @param call The CUDA API function call to check.
 *
 * @note This macro is intended for use in functions that do not allow
 * exceptions (e.g., noexcept).
 */
#define CUDA_CHECK_NOEXCEPT(call)                                         \
    do {                                                                  \
        const cudaError_t error_code = call;                              \
        if (error_code != cudaSuccess) {                                  \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error_code) \
                      << std::endl;                                       \
            std::abort();                                                 \
        }                                                                 \
    } while (0)

namespace detect {

/**
 * @brief Returns the size in bytes of the given data type.
 *
 * @param dataType The nvinfer1::DataType object.
 * @return The size in bytes of the data type.
 */
constexpr inline int sizeOfDataType(
    const nvinfer1::DataType& dataType) noexcept {
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 0;
    }
}

/**
 * @brief A class that implements the nvinfer1::ILogger interface for logging
 * messages with different severities.
 *
 */
class Logger : public nvinfer1::ILogger {
    using Severity = nvinfer1::ILogger::Severity;

   public:
    /**
     * @brief Constructs a Logger object with the specified severity level.
     *
     * @param severity The severity level for reporting messages. Defaults to
     * Severity::kWARNING.
     */
    explicit Logger(Severity severity = Severity::kWARNING)
        : reportable_severity_(severity) {}

    /**
     * @brief Logs a message with the specified severity level.
     *
     * @param severity The severity level of the message.
     * @param msg The message to be logged.
     */
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > reportable_severity_) {
            return;
        }
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "[Fatal] " << msg << std::endl;
                std::abort();
            case Severity::kERROR:
                std::cerr << "[Error] " << msg << std::endl;
                std::abort();
            case Severity::kWARNING:
                std::cerr << "[Warning] " << msg << std::endl;
                break;
            case Severity::kINFO:
                std::cout << "[Info] " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
                std::cout << "[Verbose] " << msg << std::endl;
                break;
            default:
                assert(0);
        }
    }

   private:
    Severity reportable_severity_;
};

/**
 * @brief A class representing a tensor in CUDA memory.
 *
 */
class Tensor {
   public:
    Tensor() : device_ptr_{nullptr} {}
    Tensor(Tensor&& rhs) : name_{rhs.name_}, device_ptr_{rhs.device_ptr_} {
        // Prevent repeated release of device resources
        rhs.device_ptr_ = nullptr;
    }
    Tensor& operator=(Tensor&& rhs) {
        if (this != &rhs) {
            dims_ = rhs.dims_;
            name_ = rhs.name_;
            device_ptr_ = rhs.device_ptr_;
            // Prevent repeated release of device resources
            rhs.device_ptr_ = nullptr;
        }
        return *this;
    }
    ~Tensor() {
        if (device_ptr_) {
            try {
                CUDA_CHECK(cudaFree(device_ptr_));
            } catch (std::runtime_error& err) {
                std::cerr << err.what() << std::endl;
            }
        }
    }

    /**
     * @brief Constructs a Tensor object with the given dimensions, data type,
     * name, and maximum batch size.
     *
     * @param dims The dimensions of the tensor.
     * @param dtype The data type of the tensor.
     * @param name The name of the tensor.
     * @param max_batch_size The maximum batch size for the tensor.
     */
    explicit Tensor(const nvinfer1::Dims& dims, nvinfer1::DataType dtype,
                    const char* name, int max_batch_size)
        : name_{name}, dims_{dims} {
        if (dims.d[0] != -1 && dims.d[0] != max_batch_size) {
            throw std::logic_error("invalid dims");
        }
        // Start with dims.d[1] because dims.d[0] is -1 in dynamic network.
        auto dim_size{std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1,
                                      std::multiplies<int32_t>())};
        CUDA_CHECK(cudaMalloc(
            &device_ptr_, dim_size * sizeOfDataType(dtype) * max_batch_size));
    }

    /**
     * @brief Get the name of the tensor.
     *
     * @return The name of the tensor.
     */
    inline const char* name() const noexcept { return name_.data(); }

    /**
     * @brief Get the device pointer of the tensor.
     *
     * @return The device pointer of the tensor.
     */
    inline void* data() const noexcept { return device_ptr_; }

    /**
     * @brief Get the dimensions of the tensor.
     *
     * @return The dimensions of the tensor.
     */
    inline nvinfer1::Dims dims() const noexcept { return dims_; }

   private:
    /**
     * @brief The Tensor class does not allow copying and copy assignment,
     * because the existence of two same device pointers at the same time will
     * cause two cudaFrees in one specific address during destruction.
     *
     */
    Tensor(const Tensor& rhs) = delete;

    /**
     * @brief The Tensor class does not allow copying and copy assignment,
     * because the existence of two same device pointers at the same time will
     * cause two cudaFrees in one specific address during destruction.
     *
     */
    Tensor& operator=(const Tensor& rhs) = delete;
    std::string_view name_;
    nvinfer1::Dims dims_;
    void* device_ptr_;
};

/**
 * @brief Parameters obtained by preprocessing, will be used in postprocessing
 *
 */
struct PreParam {
    PreParam() = default;
    PreParam(float width, float height, float ratio, float dw, float dh)
        : width{width}, height{height}, ratio{ratio}, dw{dw}, dh{dh} {}

    PreParam(cv::Size input, cv::Size output) {
        height = static_cast<float>(input.height);
        width = static_cast<float>(input.width);
        ratio = 1 / (std::min(output.height / height, output.width / width));
        dw = (output.width - std::round(width / ratio)) * 0.5f;
        dh = (output.height - std::round(height / ratio)) * 0.5f;
    }
    float width;
    float height;
    float ratio;
    float dw;
    float dh;
};

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

}  // namespace detect

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
                      float nms_thresh = 0.65, float conf_thresh = 0.25,
                      int input_width = 640, int input_height = 640,
                      std::string_view input_name = "images",
                      int input_channels = 3, int opt_level = 5);
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
    template <detect::ImageOrImages T>
    auto detect(T&& input) noexcept {
        std::vector<detect::PreParam> pparams{
            preprocess(std::forward<T>(input))};

        context_->enqueueV3(streams_[0]);
        if constexpr (!std::is_same_v<std::decay_t<T>, cv::Mat>) {
            CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(streams_[0]));
        }

        auto detections{postprocess(pparams)};

        if constexpr (std::is_same_v<std::decay_t<T>, cv::Mat>) {
            return detections[0];
        } else {
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

}  // namespace radar