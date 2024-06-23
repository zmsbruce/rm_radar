/**
 * @file detect.cpp
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the definition of the Detector class.
 * @date 2024-03-06
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#include "detect/detector.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <execution>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>

namespace radar {

using namespace radar::detect;

/**
 * @brief Constructs a Detector object with the specified parameters.
 * @param engine_path The path to the engine file.
 * @param classes The number of classes in detection.
 * @param max_batch_size The maximum batch size.
 * @param opt_barch_size The optimized batch size of `std::optional<int>` type.
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
Detector::Detector(std::string_view engine_path, int classes,
                   int max_batch_size, std::optional<int> opt_batch_size,
                   size_t image_size, float nms_thresh, float conf_thresh,
                   int input_width, int input_height,
                   std::string_view input_name, int input_channels,
                   int opt_level)
    : input_width_{input_width},
      input_height_{input_height},
      input_channels_{input_channels},
      input_name_{input_name},
      classes_{classes},
      nms_thresh_{nms_thresh},
      conf_thresh_{conf_thresh} {
    CUDA_CHECK(cudaSetDevice(0));
    initLibNvInferPlugins(&logger_, "radar");

    for (int i = 0; i < max_batch_size; ++i) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        streams_.push_back(stream);
    }

    std::shared_ptr<char[]> model{nullptr};
    size_t size{0};
    // If the engine file does not exist, the data will be fetched through
    // serializing the onnx file with the same name.
    if (!std::filesystem::exists(engine_path)) {
        std::cout << engine_path << " does not exist" << std::endl;
        // Derives the onnx path based on the engine path, assuming that
        // *.engine and *.onnx are in the same directory
        auto pos = engine_path.find_last_of('.');
        if (pos == std::string_view::npos) {
            throw std::invalid_argument(
                "missing delimeter \".\" in engine path");
        }
        auto onnx_path = std::string(engine_path.substr(0, pos)) + ".onnx";
        std::cout << "onnx path should be " << onnx_path << std::endl;
        std::tie(model, size) = serializeEngine(
            onnx_path, opt_batch_size.value_or(std::max(max_batch_size / 2, 1)),
            max_batch_size, opt_level);
        // Writes the serialized model to disk
        try {
            writeToFile(std::span(model.get(), size), engine_path);
        } catch (const std::ios_base::failure& ex) {
            std::cerr << "exception in writing model: " << ex.what()
                      << std::endl;
        }
    }
    // Otherwise, it will be directly read from the engine file.
    else {
        std::tie(model, size) = loadFromFile(engine_path);
    }

    // Creates execution context of TensorRT
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_));
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(static_cast<void*>(model.get()), size));
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());

    // Sets tensor addresses for input and output tensors in the execution
    // context
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        auto name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            auto dims = engine_->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kMAX);
            auto dtype = engine_->getTensorDataType(name);
            input_tensor_ = Tensor(dims, dtype, name, max_batch_size);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            auto dims = context_->getTensorShape(name);
            auto dtype = engine_->getTensorDataType(name);
            output_tensor_ = Tensor(dims, dtype, name, max_batch_size);
        } else {
            continue;
        }
    }
    context_->setTensorAddress(input_tensor_.name(), input_tensor_.data());
    context_->setTensorAddress(output_tensor_.name(), output_tensor_.data());
    output_channels_ = output_tensor_.dims().d[1];
    output_anchors_ = output_tensor_.dims().d[2];

    // Allocate host and device memory
    CUDA_CHECK(cudaHostAlloc(&image_ptr_, image_size, cudaHostAllocMapped));
    CUDA_CHECK(cudaMalloc(&dev_resize_ptr_,
                          input_height * input_width * input_channels *
                              max_batch_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&dev_border_ptr_,
                          input_height * input_width * input_channels *
                              max_batch_size * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(
        &dev_transpose_ptr_,
        output_channels_ * output_anchors_ * max_batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dev_decode_ptr_, output_anchors_ * max_batch_size *
                                                sizeof(Detection)));
    CUDA_CHECK(cudaHostAlloc(
        &nms_ptr_, output_anchors_ * max_batch_size * sizeof(Detection),
        cudaHostAllocDefault));
}

Detector::~Detector() {
    CUDA_CHECK_NOEXCEPT(cudaFree(dev_border_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFree(dev_resize_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFree(dev_transpose_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFree(dev_decode_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFreeHost(image_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFreeHost(nms_ptr_));
    for (auto&& stream : streams_) {
        CUDA_CHECK_NOEXCEPT(cudaStreamDestroy(stream));
    }
}

/**
 * @brief Serialize the TensorRT engine from an ONNX model file.
 *
 * @param onnx_path The path to the ONNX model file.
 * @param engine_path The path where the serialized engine will be saved.
 * @param opt_batch_size The optimized batch size for the engine.
 * @param max_batch_size The maximum batch size supported by the engine.
 * @param opt_level The level of optimization ranging from 0 to 5.
 * @return The serialized engine model as a `std::pair<std::shared_ptr<char[]>,
 * size_t>`.
 * @throws `std::invalid_argument` if the batch size is invalid.
 * @throws `std::runtime_error` if the ONNX file does not exist or there is an
 * error in the process.
 */
std::pair<std::shared_ptr<char[]>, size_t> Detector::serializeEngine(
    std::string_view onnx_path, int opt_batch_size, int max_batch_size,
    int opt_level) {
    if (max_batch_size < opt_batch_size || opt_batch_size < 1) {
        throw std::invalid_argument("invalid batch size");
    }
    if (!std::filesystem::exists(onnx_path)) {
        throw std::runtime_error("path does not exist");
    }

    std::unique_ptr<nvinfer1::IBuilder> builder{
        nvinfer1::createInferBuilder(logger_)};
    uint32_t flags = 0;
#if NV_TENSORRT_MAJOR < 10
    flags = 1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#endif
    std::unique_ptr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(flags)};
    std::unique_ptr<nvonnxparser::IParser> parser{
        nvonnxparser::createParser(*network, logger_)};
    if (!parser) {
        throw std::runtime_error("error in creating parser");
    }

    if (!parser->parseFromFile(
            onnx_path.data(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("failed to parse file");
    }

    constexpr int min_batch_size{1};
    auto profile{builder->createOptimizationProfile()};
    profile->setDimensions(input_name_.data(),
                           nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4(min_batch_size, input_channels_,
                                           input_width_, input_height_));
    profile->setDimensions(input_name_.data(),
                           nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(opt_batch_size, input_channels_,
                                           input_width_, input_height_));
    profile->setDimensions(input_name_.data(),
                           nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(max_batch_size, input_channels_,
                                           input_width_, input_height_));

    std::unique_ptr<nvinfer1::IBuilderConfig> config{
        builder->createBuilderConfig()};
    config->setBuilderOptimizationLevel(opt_level);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->addOptimizationProfile(profile);

    std::cout << "Building network..." << std::endl;
    std::shared_ptr<nvinfer1::IHostMemory> model{
        builder->buildSerializedNetwork(*network, *config)};

    auto deleter = [model](char*) mutable {
        // Here, there is nothing that needs to be done, as the destructor of
        // `model` will automatically release the memory. The custom deleter for
        // the `shared_ptr` is simply for proper type matching and lifecycle
        // management.
        model.reset();
    };
    return std::make_pair(
        std::shared_ptr<char[]>(static_cast<char*>(model->data()), deleter),
        model->size());
}

/**
 * @brief Restores the detection scaling and translation to the original image
 * dimensions.
 *
 * This function adjusts the detection's coordinates and size from the processed
 * scale back to the original image scale. It accounts for any padding and
 * scaling that was applied to the image before processing it through the
 * detection network.
 *
 * @param[out] detection A reference to the Detection object to be restored.
 * @param[in] pparam The preprocessing parameters that contain scaling factors
 * and padding offsets used to adjust the detection's properties.
 */
void Detector::restoreDetection(Detection& detection,
                                const PreParam& pparam) const noexcept {
    detection.x = std::clamp((detection.x - pparam.dw) * pparam.ratio, 0.0f,
                             pparam.width);
    detection.y = std::clamp((detection.y - pparam.dh) * pparam.ratio, 0.0f,
                             pparam.height);
    detection.width = std::clamp(detection.width * pparam.ratio, 0.0f,
                                 pparam.width - detection.x);
    detection.height = std::clamp(detection.height * pparam.ratio, 0.0f,
                                  pparam.height - detection.height);
}

/**
 * @brief Writes data to a file at the specified path.
 *
 * This function takes a span of characters as input and writes them to
 * a file specified by the path. It opens the file in binary mode and throws
 * exceptions on failure.
 *
 * @param data A std::span<const char> representing the data to be written.
 * @param path A std::string_view representing the file path where data is to be
 * written.
 */
void Detector::writeToFile(std::span<const char> data, std::string_view path) {
    std::ofstream ofs(path.data(), std::ios::out | std::ios::binary);
    ofs.exceptions(ofs.failbit | ofs.badbit);
    ofs.write(data.data(), data.size());
    ofs.close();
}

/**
 * @brief Loads data from a file into memory.
 *
 * Opens a file in binary mode specified by the path. It reads the entire
 * content into a dynamically allocated buffer and returns it along with its
 * size.
 *
 * @param path A std::string_view representing the path of the file to read
 * from.
 * @return A pair consisting of a shared pointer to the loaded data and the size
 * of the data.
 */
auto Detector::loadFromFile(std::string_view path)
    -> std::pair<std::shared_ptr<char[]>, size_t> {
    std::ifstream ifs{path.data(), std::ios::binary};
    ifs.exceptions(ifs.failbit | ifs.badbit);
    auto pbuf = ifs.rdbuf();
    auto size = static_cast<size_t>(pbuf->pubseekoff(0, ifs.end, ifs.in));
    pbuf->pubseekpos(0, ifs.in);
    std::shared_ptr<char[]> buffer{new char[size]};
    pbuf->sgetn(buffer.get(), size);
    ifs.close();
    return std::make_pair(buffer, size);
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
static float computeIoU(const cv::Rect2f& rect1, const cv::Rect2f& rect2) {
    float x1, y1, x2, y2;
    cv::Rect2f intersectionRect, unionRect;

    x1 = std::max(rect1.x, rect2.x);
    y1 = std::max(rect1.y, rect2.y);
    x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
    intersectionRect = x1 < x2 && y1 < y2 ? cv::Rect2f(x1, y1, x2 - x1, y2 - y1)
                                          : cv::Rect2f(0, 0, 0, 0);

    x1 = std::min(rect1.x, rect2.x);
    y1 = std::min(rect1.y, rect2.y);
    x2 = std::max(rect1.x + rect1.width, rect2.x + rect2.width);
    y2 = std::max(rect1.y + rect1.height, rect2.y + rect2.height);
    unionRect = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);

    float intersectionArea = intersectionRect.width * intersectionRect.height;
    float unionArea = unionRect.width * unionRect.height;

    if (unionArea > 0) {
        return intersectionArea / unionArea;
    } else {
        return 0.0;
    }
}

/**
 * @brief Constructs a RobotDetector object.
 *
 * This constructor initializes the RobotDetector with the specified parameters
 * for car detection and armor detection engines, as well as various thresholds
 * and input configurations.
 *
 * @param car_engine_path The file path to the car detection engine.
 * @param armor_engine_path The file path to the armor detection engine.
 * @param armor_classes The number of armor classes.
 * @param max_cars The maximum number of cars detected in one frame.
 * @param opt_cars The optimized number of cars detected in one frame.
 * @param iou_thresh The Intersection over Union (IoU) threshold for detection.
 * @param car_nms_thresh The Non-Maximum Suppression (NMS) threshold for car
 * detection.
 * @param car_conf_thresh The confidence threshold for car detection.
 * @param armor_nms_thresh The Non-Maximum Suppression (NMS) threshold for armor
 * detection.
 * @param armor_conf_thresh The confidence threshold for armor detection.
 * @param image_size The size of the input images.
 * @param input_width The width of the input images.
 * @param input_height The height of the input images.
 * @param input_name The name of the input node in the detection engine.
 * @param input_channels The number of channels in the input images.
 * @param opt_level The optimization level for the detection engine.
 */
RobotDetector::RobotDetector(std::string_view car_engine_path,
                             std::string_view armor_engine_path,
                             int armor_classes, int max_cars, int opt_cars,
                             float iou_thresh, float car_nms_thresh,
                             float car_conf_thresh, float armor_nms_thresh,
                             float armor_conf_thresh, size_t image_size,
                             float input_width, float input_height,
                             std::string_view input_name, int input_channels,
                             int opt_level)
    : iou_thresh_(iou_thresh),
      car_detector_(std::make_unique<Detector>(
          car_engine_path, 1, 1, std::nullopt, image_size, car_nms_thresh,
          car_conf_thresh, input_width, input_height, input_name,
          input_channels, opt_level)),
      armor_detector_(std::make_unique<Detector>(
          armor_engine_path, armor_classes, max_cars, opt_cars, image_size,
          armor_nms_thresh, armor_conf_thresh, input_width, input_height,
          input_name, input_channels, opt_level)) {}

/**
 * @brief Detects robots within an image using separate detectors for cars and
 * armor.
 *
 * This function first uses a car detector to identify potential car locations
 * in an image. For each detected car, a sub-image is extracted and these images
 * are then passed to an armor detector. It constructs a collection of Robot
 * objects based on the detections from both detectors. It also handles
 * overlapping detections by using an IoU threshold to determine whether two
 * detections are referring to the same object.
 *
 * @param image The input image in which to detect robots.
 *
 * @return A `std::vector<Robot>` containing all detected robots. Each robot is
 * represented by a Robot object, which includes information about the car and
 * armor detections.
 */
std::vector<Robot> RobotDetector::detect(const cv::Mat& image) {
    car_images_.clear();
    auto car_detections = car_detector_->detect(image);

    std::for_each(std::execution::seq, car_detections.begin(),
                  car_detections.end(), [&](const Detection& detection) {
                      cv::Mat car_image =
                          image(cv::Rect(detection.x, detection.y,
                                         detection.width, detection.height))
                              .clone();
                      car_images_.emplace_back(std::move(car_image));
                  });
    auto armor_detections_batch = armor_detector_->detect(car_images_);

    std::vector<Robot> robots;
    robots.reserve(car_detections.size());

    std::map<int, Robot> robots_map;
    for (size_t i = 0; i < car_detections.size(); ++i) {
        Robot robot(car_detections[i], armor_detections_batch[i]);
        if (!robot.isDetected()) {
            robots.emplace_back(robot);
            continue;
        }
        int label = robot.label().value();
        if (!robots_map.contains(label)) {
            robots_map.emplace(label, robot);
        } else {
            auto& exist_robot = robots_map.at(label);
            if (computeIoU(exist_robot.rect().value(), robot.rect().value()) >
                iou_thresh_) {
                continue;
            } else if (exist_robot.confidence().value() <
                       robot.confidence().value()) {
                std::swap(exist_robot, robot);
            }
        }
    }

    std::for_each(robots_map.begin(), robots_map.end(),
                  [&](const auto& pair) { robots.emplace_back(pair.second); });
    return robots;
}

}  // namespace radar
