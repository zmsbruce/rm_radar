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

#include "detector.h"

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
      conf_thresh_{conf_thresh},
      logger_{nvinfer1::ILogger::Severity::kVERBOSE} {
    spdlog::info("Initializing detector with engine path: {}", engine_path);
    spdlog::debug(
        "Engine path: {}, Classes: {}, Max batch size: {}, Image size: {}, NMS "
        "threshold: {}, Confidence threshold: {}",
        engine_path, classes, max_batch_size, image_size, nms_thresh,
        conf_thresh);
    spdlog::debug("Input dimensions: width={}, height={}, channels={}",
                  input_width, input_height, input_channels);

    CUDA_CHECK(cudaSetDevice(0));
    spdlog::debug("Set CUDA device to 0");

    initLibNvInferPlugins(&logger_, "radar");
    spdlog::debug("Initialized TensorRT plugins with name 'radar'");

    for (int i = 0; i < max_batch_size; ++i) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        streams_.push_back(stream);
        spdlog::trace("Created CUDA stream {}", i);
    }

    std::shared_ptr<char[]> model{nullptr};
    size_t size{0};
    // If the engine file does not exist, the data will be fetched through
    // serializing the onnx file with the same name.
    if (!std::filesystem::exists(engine_path)) {
        spdlog::warn("{} does not exist", engine_path);
        // Derives the onnx path based on the engine path, assuming that
        // *.engine and *.onnx are in the same directory
        auto pos = engine_path.find_last_of('.');
        if (pos == std::string_view::npos) {
            throw std::invalid_argument(
                "Missing delimeter \".\" in engine path");
        }
        auto onnx_path = std::string(engine_path.substr(0, pos)) + ".onnx";
        spdlog::info("ONNX path: {}", onnx_path);

        spdlog::debug("Serializing engine from ONNX file: {}", onnx_path);
        std::tie(model, size) = serializeEngine(
            onnx_path, opt_batch_size.value_or(std::max(max_batch_size / 2, 1)),
            max_batch_size, opt_level);

        // Writes the serialized model to disk
        try {
            writeToFile(std::span(model.get(), size), engine_path);
            spdlog::info("Serialized engine written to {}", engine_path);
        } catch (const std::ios_base::failure& ex) {
            spdlog::error("Exception in writing model to file: {}", ex.what());
        }
    }
    // Otherwise, it will be directly read from the engine file.
    else {
        spdlog::info("Loading engine from {}", engine_path);
        std::tie(model, size) = loadFromFile(engine_path);
        spdlog::debug("Loaded engine, size: {} bytes", size);
    }

    // Creates execution context of TensorRT
    spdlog::info("Creating TensorRT runtime and engine context");
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_));
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(static_cast<void*>(model.get()), size));
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext());
    spdlog::debug("TensorRT execution context created successfully");

    // Sets tensor addresses for input and output tensors in the execution
    // context
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        auto name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        spdlog::trace("Processing tensor: {}", name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            spdlog::debug("Tensor {} is an input tensor", name);
            auto dims = engine_->getProfileShape(
                name, 0, nvinfer1::OptProfileSelector::kMAX);
            auto dtype = engine_->getTensorDataType(name);
            input_tensor_ = Tensor(dims, dtype, name, max_batch_size);
            spdlog::trace("Input tensor: {}, dims: [{}], dtype: {}", name,
                          fmt::join(dims.d, ", "), static_cast<int32_t>(dtype));
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            spdlog::debug("Tensor {} is an output tensor", name);
            auto dims = context_->getTensorShape(name);
            auto dtype = engine_->getTensorDataType(name);
            output_tensor_ = Tensor(dims, dtype, name, max_batch_size);
            spdlog::trace("Output tensor: {}, dims: [{}], dtype: {}", name,
                          fmt::join(dims.d, ", "), static_cast<int32_t>(dtype));
        } else {
            spdlog::warn("Unknown tensor mode for tensor: {}", name);
            continue;
        }
    }

    spdlog::debug("Setting tensor addresses in the execution context");
    context_->setTensorAddress(input_tensor_.name(), input_tensor_.data());
    context_->setTensorAddress(output_tensor_.name(), output_tensor_.data());
    output_channels_ = output_tensor_.dims().d[1];
    output_anchors_ = output_tensor_.dims().d[2];
    spdlog::debug("Output tensor channels: {}, anchors: {}", output_channels_,
                  output_anchors_);

    // Allocate host and device memory
    spdlog::info("Allocating host and device memory");
    CUDA_CHECK(cudaHostAlloc(&image_ptr_, image_size, cudaHostAllocMapped));
    spdlog::trace("Allocated host memory for image: {:.2f} MB",
                  image_size / (1024.0 * 1024.0));

    size_t resize_size = input_height * input_width * input_channels *
                         max_batch_size * sizeof(unsigned char);
    CUDA_CHECK(cudaMalloc(&dev_resize_ptr_, resize_size));
    spdlog::trace("Allocated device memory for resize buffer: {:.2f} MB",
                  resize_size / (1024.0 * 1024.0));

    size_t border_size = input_height * input_width * input_channels *
                         max_batch_size * sizeof(unsigned char);
    CUDA_CHECK(cudaMalloc(&dev_border_ptr_, border_size));
    spdlog::trace("Allocated device memory for border buffer: {:.2f} MB",
                  border_size / (1024.0 * 1024.0));

    size_t transpose_size =
        output_channels_ * output_anchors_ * max_batch_size * sizeof(float);
    CUDA_CHECK(cudaMalloc(&dev_transpose_ptr_, transpose_size));
    spdlog::trace("Allocated device memory for transpose buffer: {:.2f} MB",
                  transpose_size / (1024.0 * 1024.0));

    size_t decode_size = output_anchors_ * max_batch_size * sizeof(Detection);
    CUDA_CHECK(cudaMalloc(&dev_decode_ptr_, decode_size));
    spdlog::trace("Allocated device memory for decode buffer: {:.2f} MB",
                  decode_size / (1024.0 * 1024.0));

    size_t nms_size = output_anchors_ * max_batch_size * sizeof(Detection);
    CUDA_CHECK(cudaHostAlloc(&nms_ptr_, nms_size, cudaHostAllocDefault));
    spdlog::trace("Allocated host memory for NMS buffer: {:.2f} MB",
                  nms_size / (1024.0 * 1024.0));

    spdlog::info("Detector initialized successfully.");
}

Detector::~Detector() {
    spdlog::info("Destroying Detector and releasing resources");

    spdlog::trace("Freeing device memory for dev_border_ptr_: {}",
                  fmt::ptr(dev_border_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFree(dev_border_ptr_));
    spdlog::debug("Freed device memory for dev_border_ptr_");

    spdlog::trace("Freeing device memory for dev_resize_ptr_: {}",
                  fmt::ptr(dev_resize_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFree(dev_resize_ptr_));
    spdlog::debug("Freed device memory for dev_resize_ptr_");

    spdlog::trace("Freeing device memory for dev_transpose_ptr_: {}",
                  fmt::ptr(dev_transpose_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFree(dev_transpose_ptr_));
    spdlog::debug("Freed device memory for dev_transpose_ptr_");

    spdlog::trace("Freeing device memory for dev_decode_ptr_: {}",
                  fmt::ptr(dev_decode_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFree(dev_decode_ptr_));
    spdlog::debug("Freed device memory for dev_decode_ptr_");

    spdlog::trace("Freeing host memory for image_ptr_: {}",
                  fmt::ptr(image_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFreeHost(image_ptr_));
    spdlog::debug("Freed host memory for image_ptr_");

    spdlog::trace("Freeing host memory for nms_ptr_: {}", fmt::ptr(nms_ptr_));
    CUDA_CHECK_NOEXCEPT(cudaFreeHost(nms_ptr_));
    spdlog::debug("Freed host memory for nms_ptr_");

    for (auto&& stream : streams_) {
        spdlog::trace("Destroying CUDA stream: {}", fmt::ptr(stream));
        CUDA_CHECK_NOEXCEPT(cudaStreamDestroy(stream));
        spdlog::debug("Destroyed CUDA stream");
    }

    spdlog::info("Detector destroyed and all resources released");
}

std::pair<std::shared_ptr<char[]>, size_t> Detector::serializeEngine(
    std::string_view onnx_path, int opt_batch_size, int max_batch_size,
    int opt_level) {
    // Logging input parameters
    spdlog::info("Starting engine serialization with ONNX path: {}", onnx_path);
    spdlog::debug(
        "Input parameters - ONNX path: {}, Opt batch size: {}, Max batch size: "
        "{}, Opt level: {}",
        onnx_path, opt_batch_size, max_batch_size, opt_level);

    // Batch size validation
    if (max_batch_size < opt_batch_size || opt_batch_size < 1) {
        throw std::invalid_argument(
            fmt::format("Invalid batch size parameters: opt_batch_size={}, "
                        "max_batch_size={}",
                        opt_batch_size, max_batch_size));
    }

    // Check whether ONNX file exists
    if (!std::filesystem::exists(onnx_path)) {
        throw std::runtime_error(
            fmt::format("ONNX file does not exist: {}", onnx_path));
    }

    spdlog::info("Creating TensorRT builder and network");
    // Create TensorRT builder and network
    std::unique_ptr<nvinfer1::IBuilder> builder{
        nvinfer1::createInferBuilder(logger_)};
    uint32_t flags = 0;
#if NV_TENSORRT_MAJOR < 10
    flags = 1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    spdlog::debug("Using EXPLICIT_BATCH flag for TensorRT builder");
#endif
    std::unique_ptr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(flags)};
    std::unique_ptr<nvonnxparser::IParser> parser{
        nvonnxparser::createParser(*network, logger_)};

    // Check if parser creation was successful
    if (!parser) {
        throw std::runtime_error("error in creating parser");
    }

    // Parse the ONNX file
    spdlog::info("Parsing ONNX file: {}", onnx_path);
    if (!parser->parseFromFile(
            onnx_path.data(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error(
            fmt::format("Failed to parse ONNX file: {}", onnx_path));
    }

    spdlog::info("Setting optimization profiles");
    // Set optimization profiles
    constexpr int min_batch_size{1};
    auto profile{builder->createOptimizationProfile()};
    profile->setDimensions(input_name_.data(),
                           nvinfer1::OptProfileSelector::kMIN,
                           nvinfer1::Dims4(min_batch_size, input_channels_,
                                           input_width_, input_height_));
    spdlog::trace(
        "Set MIN dimensions: batch_size={}, channels={}, width={}, height={}",
        min_batch_size, input_channels_, input_width_, input_height_);

    profile->setDimensions(input_name_.data(),
                           nvinfer1::OptProfileSelector::kOPT,
                           nvinfer1::Dims4(opt_batch_size, input_channels_,
                                           input_width_, input_height_));
    spdlog::trace(
        "Set OPT dimensions: batch_size={}, channels={}, width={}, height={}",
        opt_batch_size, input_channels_, input_width_, input_height_);

    profile->setDimensions(input_name_.data(),
                           nvinfer1::OptProfileSelector::kMAX,
                           nvinfer1::Dims4(max_batch_size, input_channels_,
                                           input_width_, input_height_));
    spdlog::trace(
        "Set MAX dimensions: batch_size={}, channels={}, width={}, height={}",
        max_batch_size, input_channels_, input_width_, input_height_);

    // Create builder config
    spdlog::info("Creating builder configuration");
    std::unique_ptr<nvinfer1::IBuilderConfig> config{
        builder->createBuilderConfig()};
    config->setBuilderOptimizationLevel(opt_level);
    spdlog::debug("Set optimization level: {}", opt_level);

    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    spdlog::debug("Enabled FP16 precision mode");

    config->addOptimizationProfile(profile);
    spdlog::info("Added optimization profile to builder config");

    // Start building the network
    spdlog::info("Building the TensorRT network");
    std::shared_ptr<nvinfer1::IHostMemory> model{
        builder->buildSerializedNetwork(*network, *config)};

    // Check if model building was successful
    if (!model) {
        throw std::runtime_error("Failed to build serialized network");
    }

    spdlog::info(
        "Successfully built the TensorRT network, serialized model size: {} "
        "bytes",
        model->size());

    // Create a custom deleter for the shared_ptr
    auto deleter = [model](char*) mutable {
        // Here, there is nothing that needs to be done, as the destructor of
        // `model` will automatically release the memory. The custom deleter for
        // the `shared_ptr` is simply for proper type matching and lifecycle
        // management.
        model.reset();
    };

    spdlog::info("Engine serialization completed successfully");
    return std::make_pair(
        std::shared_ptr<char[]>(static_cast<char*>(model->data()), deleter),
        model->size());
}

void Detector::restoreDetection(Detection& detection,
                                const PreParam& pparam) const noexcept {
    spdlog::debug(
        "Restoring detection with PreParam: dw={}, dh={}, ratio={}, width={}, "
        "height={}",
        pparam.dw, pparam.dh, pparam.ratio, pparam.width, pparam.height);
    spdlog::debug("Initial detection values: x={}, y={}, width={}, height={}",
                  detection.x, detection.y, detection.width, detection.height);

    // Restore x-coordinate
    float original_x = detection.x;
    detection.x = std::clamp((detection.x - pparam.dw) * pparam.ratio, 0.0f,
                             pparam.width);
    spdlog::trace("Updated detection.x: original={}, adjusted={}", original_x,
                  detection.x);

    // Restore y-coordinate
    float original_y = detection.y;
    detection.y = std::clamp((detection.y - pparam.dh) * pparam.ratio, 0.0f,
                             pparam.height);
    spdlog::trace("Updated detection.y: original={}, adjusted={}", original_y,
                  detection.y);

    // Restore width
    float original_width = detection.width;
    detection.width = std::clamp(detection.width * pparam.ratio, 0.0f,
                                 pparam.width - detection.x);
    spdlog::trace("Updated detection.width: original={}, adjusted={}",
                  original_width, detection.width);

    // Restore height
    float original_height = detection.height;
    detection.height = std::clamp(detection.height * pparam.ratio, 0.0f,
                                  pparam.height - detection.height);
    spdlog::trace("Updated detection.height: original={}, adjusted={}",
                  original_height, detection.height);

    spdlog::debug("Final restored detection: x={}, y={}, width={}, height={}",
                  detection.x, detection.y, detection.width, detection.height);
}

void Detector::writeToFile(std::span<const char> data, std::string_view path) {
    spdlog::info("Writing data to file: {}", path);
    spdlog::debug("Data size to be written: {} bytes", data.size());

    std::ofstream ofs(path.data(), std::ios::out | std::ios::binary);
    ofs.exceptions(ofs.failbit | ofs.badbit);
    spdlog::debug("File {} opened successfully", path);

    ofs.write(data.data(), data.size());
    spdlog::info("Successfully wrote {} bytes to file {}", data.size(), path);

    ofs.close();
    spdlog::debug("File {} closed successfully", path);
}

auto Detector::loadFromFile(std::string_view path)
    -> std::pair<std::shared_ptr<char[]>, size_t> {
    spdlog::info("Loading file: {}", path);

    std::ifstream ifs{path.data(), std::ios::binary};
    ifs.exceptions(ifs.failbit | ifs.badbit);
    spdlog::debug("File {} opened successfully", path);

    auto pbuf = ifs.rdbuf();
    auto size = static_cast<size_t>(pbuf->pubseekoff(0, ifs.end, ifs.in));
    spdlog::debug("File size: {} bytes", size);

    pbuf->pubseekpos(0, ifs.in);
    spdlog::trace("File pointer reset to beginning");

    std::shared_ptr<char[]> buffer{new char[size]};
    spdlog::debug("Buffer of size {} allocated", size);

    pbuf->sgetn(buffer.get(), size);
    spdlog::info("Successfully read {} bytes from file {}", size, path);

    ifs.close();
    spdlog::debug("File {} closed successfully", path);

    return std::make_pair(buffer, size);
}

float Detector::computeIoU(const cv::Rect2f& rect1,
                           const cv::Rect2f& rect2) noexcept {
    spdlog::debug("Computing IoU for two rectangles:");
    spdlog::debug("Rect1: x={}, y={}, width={}, height={}", rect1.x, rect1.y,
                  rect1.width, rect1.height);
    spdlog::debug("Rect2: x={}, y={}, width={}, height={}", rect2.x, rect2.y,
                  rect2.width, rect2.height);

    float x1, y1, x2, y2;
    cv::Rect2f intersectionRect, unionRect;

    x1 = std::max(rect1.x, rect2.x);
    y1 = std::max(rect1.y, rect2.y);
    x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
    intersectionRect = x1 < x2 && y1 < y2 ? cv::Rect2f(x1, y1, x2 - x1, y2 - y1)
                                          : cv::Rect2f(0, 0, 0, 0);
    spdlog::trace("Intersection rect: x={}, y={}, width={}, height={}",
                  intersectionRect.x, intersectionRect.y,
                  intersectionRect.width, intersectionRect.height);

    x1 = std::min(rect1.x, rect2.x);
    y1 = std::min(rect1.y, rect2.y);
    x2 = std::max(rect1.x + rect1.width, rect2.x + rect2.width);
    y2 = std::max(rect1.y + rect1.height, rect2.y + rect2.height);
    unionRect = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
    spdlog::trace("Union rect: x={}, y={}, width={}, height={}", unionRect.x,
                  unionRect.y, unionRect.width, unionRect.height);

    float intersectionArea = intersectionRect.width * intersectionRect.height;
    float unionArea = unionRect.width * unionRect.height;

    spdlog::debug("Intersection area: {}", intersectionArea);
    spdlog::debug("Union area: {}", unionArea);

    if (unionArea > 0) {
        float iou = intersectionArea / unionArea;
        spdlog::debug("IoU: {}", iou);
        return iou;
    } else {
        spdlog::debug("Union area is zero, returning IoU as 0.0");
        return 0.0;
    }
}

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

std::vector<Robot> RobotDetector::detect(const cv::Mat& image) noexcept {
    spdlog::debug("Starting robot detection.");

    car_images_.clear();
    spdlog::trace("Cleared cached car images.");

    auto car_detections = car_detector_->detect(image);
    spdlog::debug("Detected {} cars.", car_detections.size());

    std::for_each(std::execution::seq, car_detections.begin(),
                  car_detections.end(), [&](const Detection& detection) {
                      spdlog::trace(
                          "Processing car detection at [x={}, y={}, width={}, "
                          "height={}].",
                          detection.x, detection.y, detection.width,
                          detection.height);
                      cv::Mat car_image =
                          image(cv::Rect(detection.x, detection.y,
                                         detection.width, detection.height))
                              .clone();
                      car_images_.emplace_back(std::move(car_image));
                      spdlog::trace("Car image extracted and stored.");
                  });
    auto armor_detections_batch = armor_detector_->detect(car_images_);
    spdlog::debug("Detected armor for {} cars.", armor_detections_batch.size());

    std::vector<Robot> robots;
    robots.reserve(car_detections.size());

    std::map<int, Robot> robots_map;
    for (size_t i = 0; i < car_detections.size(); ++i) {
        spdlog::trace("Processing robot for car detection {}.", i);

        Robot robot(car_detections[i], armor_detections_batch[i]);
        if (!robot.isDetected()) {
            spdlog::debug("Robot detection failed for car {}.", i);
            robots.emplace_back(robot);
            continue;
        }
        int label = robot.label().value();
        spdlog::trace("Detected robot with label {}.", label);

        if (!robots_map.contains(label)) {
            spdlog::trace(
                "Robot with label {} not found in map. Adding new robot.",
                label);
            robots_map.emplace(label, robot);
        } else {
            auto& exist_robot = robots_map.at(label);
            float iou = Detector::computeIoU(exist_robot.rect().value(),
                                             robot.rect().value());
            spdlog::trace("IoU between existing robot and new robot: {}", iou);

            if (iou > iou_thresh_) {
                spdlog::debug(
                    "IoU threshold exceeded for robot {}. Skipping update.",
                    label);
                continue;
            } else if (exist_robot.confidence().value() <
                       robot.confidence().value()) {
                spdlog::debug(
                    "Updating robot {} with higher confidence detection.",
                    label);
                std::swap(exist_robot, robot);
            }
        }
    }

    std::for_each(robots_map.begin(), robots_map.end(), [&](const auto& pair) {
        spdlog::trace("Adding robot with label {} to final result.",
                      pair.first);
        robots.emplace_back(pair.second);
    });

    spdlog::debug("Robot detection finished. Total robots detected: {}",
                  robots.size());
    return robots;
}

}  // namespace radar
