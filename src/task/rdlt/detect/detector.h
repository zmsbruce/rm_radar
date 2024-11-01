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

/**
 * @brief Resizes an image using bilinear interpolation.
 *
 * This CUDA kernel is used to resize an image to a new size using bilinear
 * interpolation. It maps each pixel in the destination image to a corresponding
 * pixel in the source image. The value of a pixel in the destination image
 * is calculated using a weighted average of the 4 nearest pixels in the source
 * image. This kernel handles multi-channel images.
 *
 * @param src Pointer to the source image in global memory.
 * @param dst Pointer to the destination image in global memory.
 * @param channels Number of channels in the source and destination images.
 * @param src_w Width of the source image.
 * @param src_h Height of the source image.
 * @param dst_w Width of the destination image.
 * @param dst_h Height of the destination image.
 */
__global__ void resizeKernel(const unsigned char* src, unsigned char* dst,
                             int channels, int src_w, int src_h, int dst_w,
                             int dst_h);

/**
 * @brief Copies an image and makes a border around it.
 *
 * This CUDA kernel copies a source image into a new destination image and adds
 * borders on all four sides with a specified color. The size of the borders
 * on each side can be different. The border color is constant and specified
 * for each channel of the image. If the source image has multiple channels,
 * the border value for each channel must be provided.
 *
 * @param src Pointer to the source image in global memory.
 * @param dst Pointer to the destination image in global memory.
 * @param channels Number of channels in the source and destination images.
 * @param src_w Width of the source image.
 * @param src_h Height of the source image.
 * @param top Height of the top border.
 * @param bottom Height of the bottom border.
 * @param left Width of the left border.
 * @param right Width of the right border.
 */
__global__ void copyMakeBorderKernel(const unsigned char* src,
                                     unsigned char* dst, int channels,
                                     int src_w, int src_h, int top, int bottom,
                                     int left, int right);

/**
 * @brief Kernel function to transform an image from BGR to a scaled float
 * planar representation.
 *
 * This kernel takes an image in BGR format (as an array of unsigned chars) and
 * converts it into a scaled floating-point representation (in an array of
 * floats). The scaling factor is applied to each pixel, and the channels are
 * reordered from "BGRBGRBGR" to "RRRGGGBBB".
 *
 * @param src Pointer to the input image data in BGR format.
 * @param dst Pointer to the output image data in scaled float representation.
 * @param width Width of the input image in pixels.
 * @param height Height of the input image in pixels.
 * @param channels Number of channels in the source and destination images.
 * @param scale Scaling factor to apply to each pixel's value.
 */
__global__ void blobKernel(const unsigned char* src, float* dst, int width,
                           int height, int channels, float scale);

/**
 * @brief Transposes a batch of matrices using CUDA parallelization.
 *
 * This CUDA kernel function transposes a batch of matrices represented as a
 * 1D array of float values. The transposition is performed in-place, modifying
 * the input array.
 *
 * @param src Pointer to the source array of input matrices.
 * @param dst Pointer to the destination array for the transposed matrices.
 * @param rows The number of rows in each matrix.
 * @param cols The number of columns in each matrix.
 */
__global__ void transposeKernel(const float* src, float* dst, int rows,
                                int cols);

/**
 * @brief Decode detection data from the neural network output.
 *
 * This kernel decodes the bounding box and class score data from the network
 * output. It iterates over all anchors for each batch item, determining the
 * class with the highest score and writing the results to the output array.
 *
 * @param src Pointer to the input data (raw neural network output).
 * @param dst Pointer to the output data (decoded detections).
 * @param channels Number of channels in the input data (e.g., bbox coordinates
 * + class scores).
 * @param anchors Number of anchors.
 * @param classes Number of classes.
 */
__global__ void decodeKernel(const float* src, float* dst, int channels,
                             int anchors, int classes);

/**
 * @brief Non-Maximum Suppression (NMS) kernel for object detection.
 *
 * This CUDA kernel performs NMS on detection results to eliminate redundant
 * overlapping bounding boxes based on their Intersection over Union (IoU)
 * value. If the confidence score of a detection is below the score threshold,
 * its label is set to NaN. For each detection, it compares with all other
 * detections and if the IoU is above the NMS threshold and the confidence score
 * of the compared detection is higher, the label of the current detection is
 * set to NaN.
 *
 * @param dev Pointer to the device memory where detections are stored. Each
 * detection has the following format: [x, y, width, height, label,
 * confidence].
 * @param nms_thresh The IoU threshold for determining when to suppress
 * overlapping detections.
 * @param score_thresh The minimum confidence score required to keep a
 * detection.
 * @param anchors The total number of detections (anchors).
 */
__global__ void NMSKernel(float* dev, float nms_thresh, float score_thresh,
                          int anchors);

/**
 * @brief Calculate Intersection over Union (IoU) for two bounding boxes.
 *
 * This function is designed to be compatible with both the host and device side
 * in CUDA. It calculates the IoU based on the coordinates and dimensions of two
 * bounding boxes.
 *
 * @param x1 The x-coordinate of the top-left corner of the first bounding box.
 * @param y1 The y-coordinate of the top-left corner of the first bounding box.
 * @param width1 The width of the first bounding box.
 * @param height1 The height of the first bounding box.
 * @param x2 The x-coordinate of the top-left corner of the second bounding box.
 * @param y2 The y-coordinate of the top-left corner of the second bounding box.
 * @param width2 The width of the second bounding box.
 * @param height2 The height of the second bounding box.
 * @return The IoU score as a floating point value. If there is no intersection,
 * returns 0.0f.
 */
__host__ __device__ float IoU(float x1, float y1, float width1, float height1,
                              float x2, float y2, float width2, float height2);

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

    /**
     * @brief Constructs a Detector object with the specified parameters.
     * @param engine_path The path to the engine file.
     * @param classes The number of classes in detection.
     * @param max_batch_size The maximum batch size.
     * @param opt_barch_size The optimized batch size of `std::optional<int>`
     * type.
     * @param image_size The size of input image or images as bytes.
     * @param nms_thresh The threshold in nms suppression.
     * @param input_width The input width.
     * @param input_height The input height.
     * @param input_name The input name.
     * @param input_channels The number of channels.
     * @param opt_level The optimization level from 0 to 5.
     * @throws `std::invalid_argument` if engine file does not exist and given
     * engine filename does not contain delimeter "."
     */
    explicit Detector(std::string_view engine_path, int classes,
                      int max_batch_size,
                      std::optional<int> opt_batch_size = std::nullopt,
                      size_t image_size = 1 << 24, float nms_thresh = 0.65,
                      float conf_thresh = 0.25, int input_width = 640,
                      int input_height = 640,
                      std::string_view input_name = "images",
                      int input_channels = 3, int opt_level = 3);

    /**
     * @brief Destructor for the Detector class.
     *
     * This destructor is responsible for releasing all allocated resources,
     * including device memory, host memory, and CUDA streams. It logs detailed
     * trace information regarding the resource cleanup process.
     *
     * Each resource is freed using the appropriate CUDA API function, and
     * errors are checked using `CUDA_CHECK_NOEXCEPT`. If any errors occur
     * during cleanup, the program will abort.
     *
     * The destructor ensures that all resources are properly released before
     * the `Detector` object is destroyed.
     */
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

    /**
     * @brief Computes the Intersection over Union (IoU) of two rectangles.
     *
     * The IoU is a measure used in object detection to quantify the accuracy of
     * an object detector on a particular dataset. It calculates the ratio of
     * intersection area to the union area of two rectangles.
     *
     * @param rect1 The first rectangle as a cv::Rect2f.
     * @param rect2 The second rectangle as a cv::Rect2f.
     * @return The IoU ratio as a float. Returns 0.0 if the union area is zero.
     */
    static float computeIoU(const cv::Rect2f& rect1,
                            const cv::Rect2f& rect2) noexcept;

   private:
    /**
     * @brief Preprocesses a single image using the Detector class.
     *
     * This function preprocesses a single image using the provided Detector
     * class. It performs several operations including resizing, padding, and
     * normalization to prepare the image for further processing.
     *
     * @param image The input image to be preprocessed.
     * @return A vector of preprocessed image parameters with only a single
     * element.
     * @note The number of channels of input image must be equal to
     * `input_channels_` or it will trigger assertion failure.
     */
    std::vector<detect::PreParam> preprocess(const cv::Mat& image) noexcept;

    /**
     * @brief Preprocesses a batch of images using the Detector class.
     *
     * This function preprocesses a batch of images using the provided Detector
     * class. It performs several operations including resizing, padding, and
     * normalization for each image in the batch, and returns the preprocessed
     * image parameters for each image.
     *
     * @param first Iterator pointing to the first image in the batch.
     * @param last Iterator pointing to the position after the last image in the
     * batch.
     * @return A vector of preprocessed image parameters for each image in the
     * batch.
     * @note The number of channels of each input image must be equal to
     * `input_channels_` or it will trigger assertion failure.
     */
    std::vector<detect::PreParam> preprocess(
        const std::span<cv::Mat> images) noexcept;

    /**
     * @brief Post-process the detections using CUDA kernels.
     *
     * This function post-processes the raw output of a detection model using
     * CUDA. It consists of transposing the output matrix, decoding the
     * detections, applying non-maximum suppression (NMS), sorting the
     * detections, and filtering them based on a confidence threshold. The
     * results are copied back to the host and returned as a vector of Detection
     * objects. Each Detection object is restored with corresponding `PreParam`
     * data before being added to the results.
     *
     * @param pparams A `std::span` of PreParam objects that contain
     * pre-processing parameters.
     * @return `std::vector<std::vector<Detection>>` A batch-sized vector of
     * vectors containing the detections.
     * @note This function will call `std::abort()` if problems have been
     * encountered in CUDA operations or memory allocation.
     */
    std::vector<std::vector<Detection>> postprocess(
        std::span<detect::PreParam> pparams) noexcept;

    /**
     * @brief Serialize the TensorRT engine from an ONNX model file.
     *
     * @param onnx_path The path to the ONNX model file.
     * @param engine_path The path where the serialized engine will be saved.
     * @param opt_batch_size The optimized batch size for the engine.
     * @param max_batch_size The maximum batch size supported by the engine.
     * @param opt_level The level of optimization ranging from 0 to 5.
     * @return The serialized engine model as a
     * `std::pair<std::shared_ptr<char[]>, size_t>`.
     * @throws `std::invalid_argument` if the batch size is invalid.
     * @throws `std::runtime_error` if the ONNX file does not exist or there is
     * an error in the process.
     */
    std::pair<std::shared_ptr<char[]>, size_t> serializeEngine(
        std::string_view onnx_path, int opt_batch_size, int max_batch_size,
        int opt_level);

    /**
     * @brief Restores the detection scaling and translation to the original
     * image dimensions.
     *
     * This function adjusts the detection's coordinates and size from the
     * processed scale back to the original image scale. It accounts for any
     * padding and scaling that was applied to the image before processing it
     * through the detection network.
     *
     * @param[out] detection A reference to the Detection object to be restored.
     * @param[in] pparam The preprocessing parameters that contain scaling
     * factors and padding offsets used to adjust the detection's properties.
     */
    void restoreDetection(Detection& detection,
                          const detect::PreParam& pparam) const noexcept;

    /**
     * @brief Writes data to a file at the specified path.
     *
     * This function takes a span of characters as input and writes them to
     * a file specified by the path. It opens the file in binary mode and throws
     * exceptions on failure.
     *
     * @param data A std::span<const char> representing the data to be written.
     * @param path A std::string_view representing the file path where data is
     * to be written.
     */
    static void writeToFile(std::span<const char> data, std::string_view path);

    /**
     * @brief Loads data from a file into memory.
     *
     * Opens a file in binary mode specified by the path. It reads the entire
     * content into a dynamically allocated buffer and returns it along with its
     * size.
     *
     * @param path A std::string_view representing the path of the file to read
     * from.
     * @return A pair consisting of a shared pointer to the loaded data and the
     * size of the data.
     */
    static std::pair<std::shared_ptr<char[]>, size_t> loadFromFile(
        std::string_view path);

    /**
     * @brief TensorRT runtime pointer used for engine deserialization.
     */
    std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};

    /**
     * @brief TensorRT engine pointer used for inference execution.
     */
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};

    /**
     * @brief TensorRT execution context pointer used for running inference.
     */
    std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

    /**
     * @brief The width of the input image.
     */
    int input_width_;

    /**
     * @brief The height of the input image.
     */
    int input_height_;

    /**
     * @brief The number of channels in the input image.
     */
    int input_channels_;

    /**
     * @brief The name of the input tensor.
     */
    std::string_view input_name_;

    /**
     * @brief The number of detection classes.
     */
    int classes_;

    /**
     * @brief The threshold for Non-Maximum Suppression (NMS).
     */
    float nms_thresh_;

    /**
     * @brief The confidence threshold for filtering detections.
     */
    float conf_thresh_;

    /**
     * @brief Custom logger for TensorRT messages.
     */
    detect::Logger logger_;

    /**
     * @brief Input tensor used for storing input image data.
     */
    detect::Tensor input_tensor_;

    /**
     * @brief Output tensor used for storing inference results.
     */
    detect::Tensor output_tensor_;

    /**
     * @brief CUDA streams used for asynchronous execution.
     */
    std::vector<cudaStream_t> streams_;

    /**
     * @brief Pointer to the device memory where the resized image is stored.
     */
    unsigned char* image_ptr_{nullptr};

    /**
     * @brief Device memory pointer for storing resized image data.
     */
    unsigned char* dev_resize_ptr_{nullptr};

    /**
     * @brief Device memory pointer for storing image data with borders.
     */
    unsigned char* dev_border_ptr_{nullptr};

    /**
     * @brief Device memory pointer for storing transposed data.
     */
    float* dev_transpose_ptr_{nullptr};

    /**
     * @brief Device memory pointer for storing decoded detection results.
     */
    float* dev_decode_ptr_{nullptr};

    /**
     * @brief Device memory pointer used for Non-Maximum Suppression (NMS).
     */
    float* nms_ptr_{nullptr};

    /**
     * @brief The number of output channels.
     */
    int output_channels_{0};

    /**
     * @brief The number of output anchors (detections).
     */
    int output_anchors_{0};

    /**
     * @brief The batch size for inference.
     */
    int batch_size_{0};
};

/**
 * @brief A class for detecting robots by identifying cars and their
 * corresponding armor.
 *
 * The RobotDetector class is designed to detect robots in an image by using
 * two separate detectors: one for detecting cars and another for detecting
 * armor on the detected cars. Once cars are detected, sub-images of the cars
 * are extracted and further processed to detect armor. The class also applies
 * Intersection over Union (IoU) thresholds to handle overlapping detections
 * and determine whether two detections are referring to the same object.
 */
class RobotDetector {
   public:
    /**
     * @brief Constructs a RobotDetector object.
     *
     * This constructor initializes the RobotDetector with the specified
     * parameters for car detection and armor detection engines, as well as
     * various thresholds and input configurations.
     *
     * @param car_engine_path The file path to the car detection engine.
     * @param armor_engine_path The file path to the armor detection engine.
     * @param armor_classes The number of armor classes.
     * @param max_cars The maximum number of cars detected in one frame.
     * @param opt_cars The optimized number of cars detected in one frame.
     * @param iou_thresh The Intersection over Union (IoU) threshold for
     * detection.
     * @param car_nms_thresh The Non-Maximum Suppression (NMS) threshold for car
     * detection.
     * @param car_conf_thresh The confidence threshold for car detection.
     * @param armor_nms_thresh The Non-Maximum Suppression (NMS) threshold for
     * armor detection.
     * @param armor_conf_thresh The confidence threshold for armor detection.
     * @param image_size The size of the input images.
     * @param input_width The width of the input images.
     * @param input_height The height of the input images.
     * @param input_name The name of the input node in the detection engine.
     * @param input_channels The number of channels in the input images.
     * @param opt_level The optimization level for the detection engine.
     */
    explicit RobotDetector(
        std::string_view car_engine_path, std::string_view armor_engine_path,
        int armor_classes, int max_cars, int opt_cars, float iou_thresh = 0.75f,
        float car_nms_thresh = 0.65f, float car_conf_thresh = 0.25f,
        float armor_nms_thresh = 0.65f, float armor_conf_thresh = 0.50f,
        size_t image_size = 1 << 24, float input_width = 640,
        float input_height = 640, std::string_view input_name = "images",
        int input_channels = 3, int opt_level = 5);

    /**
     * @brief Deleted default constructor.
     *
     */
    RobotDetector() = delete;

    /**
     * @brief Detects robots within an image using separate detectors for cars
     * and armor.
     *
     * This function first uses a car detector to identify potential car
     * locations in an image. For each detected car, a sub-image is extracted
     * and these images are then passed to an armor detector. It constructs a
     * collection of Robot objects based on the detections from both detectors.
     * It also handles overlapping detections by using an IoU threshold to
     * determine whether two detections are referring to the same object.
     *
     * @param image The input image in which to detect robots.
     *
     * @return A `std::vector<Robot>` containing all detected robots. Each robot
     * is represented by a Robot object, which includes information about the
     * car and armor detections.
     */
    std::vector<Robot> detect(const cv::Mat& image) noexcept;

   private:
    /**
     * @brief The Intersection over Union (IoU) threshold for determining
     * whether two detections overlap.
     */
    float iou_thresh_;

    /**
     * @brief Detector object used for detecting cars.
     */
    std::unique_ptr<Detector> car_detector_;

    /**
     * @brief Detector object used for detecting armor on cars.
     */
    std::unique_ptr<Detector> armor_detector_;

    /**
     * @brief Vector of car images extracted from the input image.
     */
    std::vector<cv::Mat> car_images_;
};

}  // namespace radar