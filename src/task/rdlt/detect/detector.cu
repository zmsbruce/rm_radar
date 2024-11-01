/**
 * @file detect.cu
 * @author zmsbruce (zmsbruce@163.com)
 * @brief Provides functions and CUDA kernels for image pre-processing and
 * post-processing.
 * @date 2024-03-09
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#include <cuda_runtime.h>

#include <execution>
#include <opencv2/opencv.hpp>
#include <ranges>

#include "detector.h"

namespace radar::detect {

__global__ void resizeKernel(const unsigned char* src, unsigned char* dst,
                             int channels, int src_w, int src_h, int dst_w,
                             int dst_h) {
    int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    int dst_x = blockDim.x * blockIdx.x + threadIdx.x;

    int src_step = src_w * channels;
    int dst_step = dst_w * channels;

    if (dst_y >= dst_h || dst_x >= dst_w) {
        return;
    }

    float src_y = dst_y * static_cast<float>(src_h) / dst_h;
    float src_x = dst_x * static_cast<float>(src_w) / dst_w;

    int src_y_low = static_cast<int>(src_y);
    int src_y_high = min(src_y_low + 1, src_h - 1);
    int src_x_low = static_cast<int>(src_x);
    int src_x_high = min(src_x_low + 1, src_w - 1);

    float ly = src_y - src_y_low;
    float lx = src_x - src_x_low;
    float hy = 1.f - ly;
    float hx = 1.f - lx;

#pragma unroll
    for (int c = 0; c < channels; ++c) {
        float tl =
            src[src_y_low * src_step + src_x_low * channels + c] * hy * hx;
        float tr =
            src[src_y_low * src_step + src_x_high * channels + c] * hy * lx;
        float bl =
            src[src_y_high * src_step + src_x_low * channels + c] * ly * hx;
        float br =
            src[src_y_high * src_step + src_x_high * channels + c] * ly * lx;

        float value = tl + tr + bl + br;
        dst[dst_y * dst_step + dst_x * channels + c] =
            static_cast<unsigned char>(value);
    }
}

__global__ void copyMakeBorderKernel(const unsigned char* src,
                                     unsigned char* dst, int channels,
                                     int src_w, int src_h, int top, int bottom,
                                     int left, int right) {
    int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    int dst_x = blockDim.x * blockIdx.x + threadIdx.x;

    int dst_w = src_w + left + right;
    int dst_h = src_h + top + bottom;

    int src_step = src_w * channels;
    int dst_step = dst_w * channels;

    if (dst_y >= dst_h || dst_x >= dst_w) {
        return;
    }

    int src_y = dst_y - top;
    int src_x = dst_x - left;

    int src_index = src_y * src_step + src_x * channels;
    int dst_index = dst_y * dst_step + dst_x * channels;

#pragma unroll
    for (int c = 0; c < channels; ++c) {
        if (src_y >= 0 && src_y < src_h && src_x >= 0 && src_x < src_w) {
            dst[dst_index + c] = src[src_index + c];
        } else {
            dst[dst_index + c] = 128;
        }
    }
}

__global__ void blobKernel(const unsigned char* src, float* dst, int width,
                           int height, int channels, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    dst[y * width + x + width * height * 0] =
        src[(y * width + x) * 3 + 2] * scale;
    dst[y * width + x + width * height * 1] =
        src[(y * width + x) * 3 + 1] * scale;
    dst[y * width + x + width * height * 2] =
        src[(y * width + x) * 3 + 0] * scale;

    if (channels == 4) {
        dst[y * width + x + width * height * 3] =
            src[(y * width + x) * 3 + 3] * scale;
    }
}

__global__ void transposeKernel(const float* src, float* dst, int rows,
                                int cols) {
    int src_row = blockIdx.y * blockDim.y + threadIdx.y;
    int src_col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared[32][33];
    if (src_row < rows && src_col < cols) {
        shared[threadIdx.y][threadIdx.x] = src[src_col + src_row * cols];
    } else {
        shared[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int dst_col = threadIdx.x + blockIdx.y * blockDim.y;
    int dst_row = threadIdx.y + blockIdx.x * blockDim.x;
    if (dst_col < rows && dst_row < cols) {
        dst[dst_col + dst_row * rows] = shared[threadIdx.x][threadIdx.y];
    }
}

__global__ void decodeKernel(const float* src, float* dst, int channels,
                             int anchors, int classes) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= anchors) {
        return;
    }

    const float* row_ptr = src + channels * row;
    const float* bbox_ptr = row_ptr;
    const float* score_ptr = row_ptr + 4;

    float* max_score_ptr = const_cast<float*>(score_ptr);
    for (int j = 0; j < classes; ++j) {
        if (score_ptr[j] > *max_score_ptr) {
            max_score_ptr = const_cast<float*>(score_ptr) + j;
        }
    }

    float x = max(bbox_ptr[0] - 0.5 * bbox_ptr[2], 0.0f);
    float y = max(bbox_ptr[1] - 0.5 * bbox_ptr[3], 0.0f);
    float width = bbox_ptr[2];
    float height = bbox_ptr[3];
    float label = max_score_ptr - score_ptr;
    float confidence = *max_score_ptr;

    int offset = row * sizeof(Detection) / sizeof(float);
    dst[offset + 0] = x;
    dst[offset + 1] = y;
    dst[offset + 2] = width;
    dst[offset + 3] = height;
    dst[offset + 4] = label;
    dst[offset + 5] = confidence;
}

__host__ __device__ float IoU(float x1, float y1, float width1, float height1,
                              float x2, float y2, float width2, float height2) {
    float x_left = max(x1, x2);
    float y_top = max(y1, y2);
    float x_right = min(x1 + width1, x2 + width2);
    float y_bottom = min(y1 + height1, y2 + height2);

    if (x_right < x_left || y_bottom < y_top) {
        return 0.0f;
    }

    float intersection_width = x_right - x_left;
    float intersection_height = y_bottom - y_top;

    float intersection_area = intersection_width * intersection_height;

    float area1 = width1 * height1;
    float area2 = width2 * height2;

    float union_area = area1 + area2 - intersection_area;

    return intersection_area / union_area;
}

__global__ void NMSKernel(float* dev, float nms_thresh, float score_thresh,
                          int anchors) {
    const int block_size = blockDim.x;
    const int thread_id = threadIdx.x;
    const int row = blockIdx.y;
    const int col = blockIdx.x;
    constexpr int num_attrs = sizeof(Detection) / sizeof(float);

    const int rows = min(anchors - row * block_size, block_size);
    const int cols = min(anchors - col * block_size, block_size);

    extern __shared__ float shared_data[];

    for (int i = thread_id; i < cols * num_attrs; i += block_size) {
        int col_offset = block_size * col * num_attrs;
        shared_data[i] = dev[col_offset + i];
    }
    __syncthreads();

    if (thread_id < rows) {
        int row_index = (block_size * row + thread_id) * num_attrs;
        float row_bbox[4] = {dev[row_index], dev[row_index + 1],
                             dev[row_index + 2], dev[row_index + 3]};
        float row_label = dev[row_index + 4];
        float row_conf = dev[row_index + 5];

        if (row_conf < score_thresh) {
            dev[row_index + 4] = NAN;
            return;
        }

        for (int i = 0; i < cols; ++i) {
            float* comp_bbox = &shared_data[i * num_attrs];
            float comp_label = comp_bbox[4];
            float comp_conf = comp_bbox[5];
            if (comp_label == row_label && comp_conf > row_conf) {
                if (IoU(row_bbox[0], row_bbox[1], row_bbox[2], row_bbox[3],
                        comp_bbox[0], comp_bbox[1], comp_bbox[2],
                        comp_bbox[3]) > nms_thresh) {
                    dev[row_index + 4] = NAN;
                    return;
                }
            }
        }
    }
}

}  // namespace radar::detect

namespace radar {

using namespace radar::detect;

std::vector<PreParam> Detector::preprocess(const cv::Mat& image) noexcept {
    spdlog::debug("Starting preprocessing single image.");

    // Make sure the input image meets expectations
    assert(!image.empty() && image.channels() == input_channels_);
    spdlog::trace("Processing image with size: {}x{} and {} channels.",
                  image.cols, image.rows, image.channels());

    // Defining CUDA block and grid sizes
    dim3 block_size(16, 16);
    dim3 grid_size;

    batch_size_ = 1;
    spdlog::debug("Batch size set to 1.");

    // Copy image data to image_ptr_
    std::memcpy(image_ptr_, image.data,
                image.total() * image.elemSize() * sizeof(unsigned char));
    spdlog::debug("Copied image data to device memory.");

    auto& stream{streams_[0]};

    // Calculate preprocessing parameters
    PreParam pparam(image.size(), cv::Size(input_width_, input_height_));
    float padding_width{pparam.width / pparam.ratio};
    float padding_height{pparam.height / pparam.ratio};
    spdlog::trace(
        "PreParam: width = {}, height = {}, ratio = {}, padding_width = {}, "
        "padding_height = {}",
        pparam.width, pparam.height, pparam.ratio, padding_width,
        padding_height);

    // Calculate the grid size for the resize kernel
    grid_size = dim3((padding_width + block_size.x - 1) / block_size.x,
                     (padding_height + block_size.y - 1) / block_size.y);
    spdlog::trace("Grid size for resize kernel: ({}, {})", grid_size.x,
                  grid_size.y);

    // Call the resize kernel
    resizeKernel<<<grid_size, block_size, 0, stream>>>(
        image_ptr_, dev_resize_ptr_, input_channels_, pparam.width,
        pparam.height, padding_width, padding_height);
    spdlog::debug("Resize kernel launched.");

    // Calculate the top, bottom, left and right values ​​of the border
    int top{static_cast<int>(std::round(pparam.dh - 0.1))};
    int bottom{static_cast<int>(std::round(pparam.dh + 0.1))};
    int left{static_cast<int>(std::round(pparam.dw - 0.1))};
    int right{static_cast<int>(std::round(pparam.dw + 0.1))};
    spdlog::trace("Border values: top = {}, bottom = {}, left = {}, right = {}",
                  top, bottom, left, right);

    // Calculate the grid size for the copyMakeBorder kernel
    grid_size = dim3((input_width_ + block_size.x - 1) / block_size.x,
                     (input_height_ + block_size.y - 1) / block_size.y);
    spdlog::trace("Grid size for copyMakeBorder kernel: ({}, {})", grid_size.x,
                  grid_size.y);

    copyMakeBorderKernel<<<grid_size, block_size, 0, stream>>>(
        dev_resize_ptr_, dev_border_ptr_, input_channels_, padding_width,
        padding_height, top, bottom, left, right);
    spdlog::debug("copyMakeBorder kernel launched.");

    // Call the blob kernel
    blobKernel<<<grid_size, block_size, 0, stream>>>(
        dev_border_ptr_, static_cast<float*>(input_tensor_.data()),
        input_width_, input_height_, input_channels_, 1 / 255.f);
    spdlog::debug("Blob kernel launched.");

    // Set the shape of the input tensor
    context_->setInputShape(input_tensor_.name(),
                            nvinfer1::Dims4(batch_size_, input_channels_,
                                            input_width_, input_height_));
    spdlog::debug(
        "Set input tensor shape: batch_size = {}, channels = {}, width = {}, "
        "height = {}",
        batch_size_, input_channels_, input_width_, input_height_);

    spdlog::debug("Preprocessing completed.");

    return std::vector<PreParam>{pparam};
}

std::vector<PreParam> Detector::preprocess(
    const std::span<cv::Mat> images) noexcept {
    spdlog::debug("Starting preprocessing of {} images.", images.size());

    // Set block and grid sizes for CUDA kernels
    dim3 block_size(16, 16);
    dim3 grid_size;

    // Set the batch size
    batch_size_ = images.size();
    spdlog::debug("Batch size set to {}.", batch_size_);

    // Prepare vector for preprocessed parameters
    std::vector<PreParam> pparams;
    pparams.reserve(batch_size_);

    // Initialize offsets for image, resize, and input memory
    size_t offset_image{0}, offset_resize{0}, offset_input{0};

    // Loop through each image in the batch
    for (size_t i = 0; i < batch_size_; ++i) {
        cv::Mat& image = images[i];

        // Ensure the input image is not empty and has the expected number of
        // channels
        assert(!image.empty() && image.channels() == input_channels_);
        spdlog::trace("Processing image {} with size: {}x{} and {} channels.",
                      i, image.cols, image.rows, image.channels());

        // Copy image data to device memory at the current offset
        std::memcpy(image_ptr_ + offset_image, image.data,
                    image.total() * image.elemSize() * sizeof(unsigned char));
        spdlog::trace("Copied image {} data to device memory (offset: {}).", i,
                      offset_image);

        // Compute preprocessing parameters
        PreParam pparam(image.size(), cv::Size(input_width_, input_height_));

        float padding_width{pparam.width / pparam.ratio};
        float padding_height{pparam.height / pparam.ratio};
        grid_size = dim3((padding_width + block_size.x - 1) / block_size.x,
                         (padding_height + block_size.y - 1) / block_size.y);
        spdlog::trace("Grid size for resize kernel on image {}: ({}, {}).", i,
                      grid_size.x, grid_size.y);

        // Launch resize kernel
        resizeKernel<<<grid_size, block_size, 0, streams_[i]>>>(
            image_ptr_ + offset_image, dev_resize_ptr_ + offset_resize,
            input_channels_, pparam.width, pparam.height, padding_width,
            padding_height);
        spdlog::debug("Resize kernel launched for image {}.", i);

        // Compute border values based on the PreParam object
        int top{static_cast<int>(std::round(pparam.dh - 0.1))};
        int bottom{static_cast<int>(std::round(pparam.dh + 0.1))};
        int left{static_cast<int>(std::round(pparam.dw - 0.1))};
        int right{static_cast<int>(std::round(pparam.dw + 0.1))};
        spdlog::trace(
            "Border values for image {}: top = {}, bottom = {}, left = {}, "
            "right = {}.",
            i, top, bottom, left, right);

        // Set grid size for the border kernel
        grid_size = dim3((input_width_ + block_size.x - 1) / block_size.x,
                         (input_height_ + block_size.y - 1) / block_size.y);
        spdlog::trace(
            "Grid size for copyMakeBorder kernel on image {}: ({}, {}).", i,
            grid_size.x, grid_size.y);

        // Launch copyMakeBorder kernel
        copyMakeBorderKernel<<<grid_size, block_size, 0, streams_[i]>>>(
            dev_resize_ptr_ + offset_resize, dev_border_ptr_ + offset_input,
            input_channels_, padding_width, padding_height, top, bottom, left,
            right);
        spdlog::debug("copyMakeBorder kernel launched for image {}.", i);

        // Launch blob kernel
        blobKernel<<<grid_size, block_size, 0, streams_[i]>>>(
            dev_border_ptr_ + offset_input,
            static_cast<float*>(input_tensor_.data()) + offset_input,
            input_width_, input_height_, input_channels_, 1 / 255.f);
        spdlog::debug("Blob kernel launched for image {}.", i);

        // Update offsets for the next image
        offset_image += image.total() * image.elemSize();
        offset_resize += padding_height * padding_width * input_channels_;
        offset_input += input_width_ * input_height_ * input_channels_;
        spdlog::trace(
            "Updated offsets: offset_image = {}, offset_resize = {}, "
            "offset_input = {}.",
            offset_image, offset_resize, offset_input);

        // Store preprocessing parameter for this image
        pparams.emplace_back(pparam);
    }

    // Set input tensor shape for the batch
    context_->setInputShape(input_tensor_.name(),
                            nvinfer1::Dims4(batch_size_, input_channels_,
                                            input_width_, input_height_));
    spdlog::debug(
        "Input tensor shape set: batch_size = {}, channels = {}, width = {}, "
        "height = {}.",
        batch_size_, input_channels_, input_width_, input_height_);

    // Synchronize all CUDA streams
    spdlog::debug("Synchronizing {} CUDA streams.", batch_size_);
    std::for_each_n(streams_.begin(), batch_size_, [this](auto&& stream) {
        CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(stream));
    });
    spdlog::debug("All CUDA streams synchronized.");

    spdlog::debug("Preprocessing completed.");
    return pparams;
}

std::vector<std::vector<Detection>> Detector::postprocess(
    std::span<PreParam> pparams) noexcept {
    spdlog::debug("Starting postprocessing for {} images.", batch_size_);

    size_t offset_output{0}, offset_decode{0};

    dim3 block_size, grid_size;

    // Iterate through the batch
    for (int i = 0; i < batch_size_; ++i) {
        spdlog::trace("Processing image {} in the batch.", i);

        // Set block and grid sizes for transpose kernel
        block_size = dim3(32, 32);
        grid_size = dim3((output_anchors_ + block_size.x - 1) / block_size.x,
                         (output_channels_ + block_size.y - 1) / block_size.y);
        spdlog::trace(
            "Grid size for transpose kernel: ({}, {}), block size: ({}, {}).",
            grid_size.x, grid_size.y, block_size.x, block_size.y);

        // Launch transpose kernel
        transposeKernel<<<grid_size, block_size, 0, streams_[i]>>>(
            static_cast<float*>(output_tensor_.data()) + offset_output,
            dev_transpose_ptr_ + offset_output, output_channels_,
            output_anchors_);
        spdlog::debug("Transpose kernel launched for image {}.", i);

        // Set block and grid sizes for decode kernel
        block_size = dim3(32);
        grid_size = dim3((output_anchors_ + block_size.x - 1) / block_size.x);
        spdlog::trace("Grid size for decode kernel: ({}, {}), block size: {}.",
                      grid_size.x, grid_size.y, block_size.x);
        // Launch decode kernel
        decodeKernel<<<grid_size, block_size, 0, streams_[i]>>>(
            dev_transpose_ptr_ + offset_output, dev_decode_ptr_ + offset_decode,
            output_channels_, output_anchors_, classes_);
        spdlog::debug("Decode kernel launched for image {}.", i);

        // Set block and grid sizes for NMS (Non-Maximum Suppression) kernel
        block_size = dim3(16 * 16);
        grid_size = dim3((output_anchors_ + block_size.x - 1) / block_size.x,
                         (output_anchors_ + block_size.x - 1) / block_size.x);
        spdlog::trace("Grid size for NMS kernel: ({}, {}), block size: {}.",
                      grid_size.x, grid_size.y, block_size.x);
        // Launch NMS kernel
        NMSKernel<<<grid_size, block_size, block_size.x * sizeof(Detection),
                    streams_[i]>>>(dev_decode_ptr_ + offset_decode, nms_thresh_,
                                   conf_thresh_, output_anchors_);
        spdlog::debug("NMS kernel launched for image {}.", i);

        // Copy the NMS results back to host memory asynchronously
        cudaMemcpyAsync(nms_ptr_, dev_decode_ptr_,
                        output_anchors_ * batch_size_ * sizeof(Detection),
                        cudaMemcpyDeviceToHost, streams_[i]);
        spdlog::debug(
            "Asynchronous copy from device to host initiated for image {}.", i);

        // Update offsets for the next image
        offset_output += output_channels_ * output_anchors_;
        offset_decode += sizeof(Detection) / sizeof(float) * output_anchors_;
        spdlog::trace(
            "Updated offsets: offset_output = {}, offset_decode = {}.",
            offset_output, offset_decode);
    }

    // Synchronize all CUDA streams
    spdlog::debug("Synchronizing {} CUDA streams.", batch_size_);
    std::for_each_n(streams_.begin(), batch_size_, [this](auto&& stream) {
        CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(stream));
    });
    spdlog::debug("All CUDA streams synchronized.");

    // Prepare the results for each image
    std::vector<std::vector<Detection>> results(batch_size_);
    spdlog::debug("Starting final postprocessing on the host.");

    // Parallel process the detections for each image
    std::for_each(
        std::execution::par_unseq, results.begin(), results.end(),
        [&](std::vector<Detection>& result) {
            // Calculate the index of the current image
            int i = &result - &results[0];
            spdlog::trace("Postprocessing detections for image {}.", i);

            // Get the span of detections for the current image
            std::span<Detection> detections(
                reinterpret_cast<Detection*>(nms_ptr_) + output_anchors_ * i,
                output_anchors_);

            // Filter out NaN detections
            auto nan_filtered_detections =
                detections | std::views::filter([](const auto& detection) {
                    return !std::isnan(detection.label);
                });

            // Get the corresponding PreParam for the current image
            const auto& pparam{pparams[i]};
            // Restore each valid detection and add it to the result
            for (auto& detection : nan_filtered_detections) {
                restoreDetection(detection, pparam);
                result.emplace_back(detection);
            }

            spdlog::trace("Completed postprocessing for image {}.", i);
        });

    spdlog::debug("Postprocessing complete. Returning results.");
    return results;
}

}  // namespace radar