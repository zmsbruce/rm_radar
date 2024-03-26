/**
 * @file detect.cpp
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the definition of several classes and functions
 * related to object detection using NVIDIA TensorRT.
 * @date 2024-03-06
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#include "detect/detector.h"

#include <NvOnnxParser.h>

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

//! We utilize locked page memory to hold the preprocessed data of input images.
//! Allocating this type of memory is a demanding process in terms of time,
//! which is why we allocate it at the construction phase. The exact size of
//! this memory is indeterminate, so we initially establish certain parameters
//! based on an estimate of the input field picture's size (we assume that the
//! total size of detected car images will be less than that of the field
//! picture). These parameters are adjustable to fit specific requirements or
//! situations.
constexpr int kImageWidth{2592};
constexpr int kImageHeight{2048};

/**
 * @brief Constructs a Detector object with the specified parameters.
 * @param engine_path The path to the engine file.
 * @param classes The number of classes in detection.
 * @param max_batch_size The maximum batch size.
 * @param opt_barch_size The optimized batch size of `std::optional<int>` type.
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
                   float nms_thresh, float conf_thresh, int input_width,
                   int input_height, std::string_view input_name,
                   int input_channels, int opt_level)
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
    CUDA_CHECK(cudaHostAlloc(
        &image_ptr_,
        kImageWidth * kImageHeight * input_channels * sizeof(unsigned char),
        cudaHostAllocMapped));
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
    std::unique_ptr<nvinfer1::INetworkDefinition> network{
        builder->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))};
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

void Detector::writeToFile(std::span<const char> data, std::string_view path) {
    std::ofstream ofs(path.data(), std::ios::out | std::ios::binary);
    ofs.exceptions(ofs.failbit | ofs.badbit);
    ofs.write(data.data(), data.size());
    ofs.close();
}

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

}  // namespace radar
