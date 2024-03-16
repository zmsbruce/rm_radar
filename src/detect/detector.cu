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

#include "detector.h"

namespace radar {

#define BLOCK_SIZE 16

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

    if (src_y >= 0 && src_y < src_h && src_x >= 0 && src_x < src_w) {
        for (int c = 0; c < channels; ++c) {
            dst[dst_index + c] = src[src_index + c];
        }
    } else {
        for (int c = 0; c < channels; ++c) {
            dst[dst_index + c] = 128;
        }
    }
}

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
 * @param scale Scaling factor to apply to each pixel's value.
 */
__global__ void blobKernel(const unsigned char* src, float* dst, int width,
                           int height, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int offset = y * width * 3 + x * 3;
    dst[offset / 3] = (float)src[offset + 2] * scale;
    dst[offset / 3 + width * height] = (float)src[offset + 1] * scale;
    dst[offset / 3 + width * height * 2] = (float)src[offset + 0] * scale;
}

/**
 * @brief Preprocesses a single image using the Detector class.
 *
 * This function preprocesses a single image using the provided Detector class.
 * It performs several operations including resizing, padding, and normalization
 * to prepare the image for further processing.
 *
 * @param image The input image to be preprocessed.
 * @return A vector of preprocessed image parameters with only a single element.
 * @note The number of channels of input image must be equal to
 * `input_channels_` or it will trigger assertion failure.
 */
std::vector<PreParam> Detector::preprocess(const cv::Mat& image) noexcept {
    assert(image.channels() == input_channels_);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size;

    batch_size_ = 1;

    std::memcpy(image_ptr_, image.data,
                image.total() * image.elemSize() * sizeof(unsigned char));

    auto& stream{streams_[0]};

    PreParam pparam(image.size(), cv::Size(input_width_, input_height_));
    float padding_width{pparam.width / pparam.ratio};
    float padding_height{pparam.height / pparam.ratio};
    grid_size = dim3((padding_width + block_size.x - 1) / block_size.x,
                     (padding_height + block_size.y - 1) / block_size.y);
    resizeKernel<<<grid_size, block_size, 0, stream>>>(
        image_ptr_, dev_resize_ptr_, input_channels_, pparam.width,
        pparam.height, padding_width, padding_height);

    int top{static_cast<int>(std::round(pparam.dh - 0.1))};
    int bottom{static_cast<int>(std::round(pparam.dh + 0.1))};
    int left{static_cast<int>(std::round(pparam.dw - 0.1))};
    int right{static_cast<int>(std::round(pparam.dw + 0.1))};
    grid_size = dim3((input_width_ + block_size.x - 1) / block_size.x,
                     (input_height_ + block_size.y - 1) / block_size.y);
    copyMakeBorderKernel<<<grid_size, block_size, 0, stream>>>(
        dev_resize_ptr_, dev_border_ptr_, input_channels_, padding_width,
        padding_height, top, bottom, left, right);

    blobKernel<<<grid_size, block_size, 0, stream>>>(
        dev_border_ptr_, static_cast<float*>(input_tensor_.data()),
        input_width_, input_height_, 1 / 255.f);

    context_->setInputShape(input_tensor_.name(),
                            nvinfer1::Dims4(batch_size_, input_channels_,
                                            input_width_, input_height_));

    return std::vector<PreParam>{pparam};
}

/**
 * @brief Preprocesses a batch of images using the Detector class.
 *
 * This function preprocesses a batch of images using the provided Detector
 * class. It performs several operations including resizing, padding, and
 * normalization for each image in the batch, and returns the preprocessed image
 * parameters for each image.
 *
 * @param first Iterator pointing to the first image in the batch.
 * @param last Iterator pointing to the position after the last image in the
 * batch.
 * @return A vector of preprocessed image parameters for each image in the
 * batch.
 * @note The number of channels of each input image must be equal to
 * `input_channels_` or it will trigger assertion failure.
 */
std::vector<PreParam> Detector::preprocess(
    const std::span<cv::Mat> images) noexcept {
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size;

    batch_size_ = images.size();

    std::vector<PreParam> pparams;
    pparams.reserve(batch_size_);

    size_t offset_image{0}, offset_resize{0}, offset_input{0};

    for (size_t i = 0; i < batch_size_; ++i) {
        cv::Mat& image = images[i];

        assert(image.channels() == input_channels_);

        std::memcpy(image_ptr_ + offset_image, image.data,
                    image.total() * image.elemSize() * sizeof(unsigned char));

        PreParam pparam(image.size(), cv::Size(input_width_, input_height_));

        float padding_width{pparam.width / pparam.ratio};
        float padding_height{pparam.height / pparam.ratio};
        grid_size = dim3((padding_width + block_size.x - 1) / block_size.x,
                         (padding_height + block_size.y - 1) / block_size.y);
        resizeKernel<<<grid_size, block_size, 0, streams_[i]>>>(
            image_ptr_ + offset_image, dev_resize_ptr_ + offset_resize,
            input_channels_, pparam.width, pparam.height, padding_width,
            padding_height);

        int top{static_cast<int>(std::round(pparam.dh - 0.1))};
        int bottom{static_cast<int>(std::round(pparam.dh + 0.1))};
        int left{static_cast<int>(std::round(pparam.dw - 0.1))};
        int right{static_cast<int>(std::round(pparam.dw + 0.1))};
        grid_size = dim3((input_width_ + block_size.x - 1) / block_size.x,
                         (input_height_ + block_size.y - 1) / block_size.y);
        copyMakeBorderKernel<<<grid_size, block_size, 0, streams_[i]>>>(
            dev_resize_ptr_ + offset_resize, dev_border_ptr_ + offset_input,
            input_channels_, padding_width, padding_height, top, bottom, left,
            right);

        blobKernel<<<grid_size, block_size, 0, streams_[i]>>>(
            dev_border_ptr_ + offset_input,
            static_cast<float*>(input_tensor_.data()) + offset_input,
            input_width_, input_height_, 1 / 255.f);

        offset_image += image.total() * image.elemSize();
        offset_resize += padding_height * padding_width * input_channels_;
        offset_input += input_width_ * input_height_ * input_channels_;

        pparams.emplace_back(pparam);
    }

    context_->setInputShape(input_tensor_.name(),
                            nvinfer1::Dims4(batch_size_, input_channels_,
                                            input_width_, input_height_));

    std::for_each_n(streams_.begin(), batch_size_, [this](auto&& stream) {
        CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(stream));
    });

    return pparams;
}

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
                                int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        dst[col * rows + row] = src[row * cols + col];
    }
}

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

/**
 * @brief Post-process the detections using CUDA kernels.
 *
 * This function post-processes the raw output of a detection model using CUDA.
 * It consists of transposing the output matrix, decoding the detections,
 * applying non-maximum suppression (NMS), sorting the detections, and filtering
 * them based on a confidence threshold. The results are copied back to the host
 * and returned as a vector of Detection objects. Each Detection object is
 * restored with corresponding `PreParam` data before being added to the
 * results.
 *
 * @param pparams A `std::span` of PreParam objects that contain pre-processing
 * parameters.
 * @return `std::vector<std::vector<Detection>>` A batch-sized vector of vectors
 * containing the detections.
 * @note This function will call `std::abort()` if problems have been
 * encountered in CUDA operations or memory allocation.
 */
std::vector<std::vector<Detection>> Detector::postprocess(
    std::span<PreParam> pparams) noexcept {
    size_t offset_output{0}, offset_decode{0};

    dim3 block_size, grid_size;
    for (int i = 0; i < batch_size_; ++i) {
        block_size = dim3(BLOCK_SIZE, BLOCK_SIZE);
        grid_size = dim3((output_anchors_ + block_size.x - 1) / block_size.x,
                         (output_channels_ + block_size.y - 1) / block_size.y);
        transposeKernel<<<grid_size, block_size, 0, streams_[i]>>>(
            static_cast<float*>(output_tensor_.data()) + offset_output,
            dev_transpose_ptr_ + offset_output, output_channels_,
            output_anchors_);

        block_size = dim3(BLOCK_SIZE * BLOCK_SIZE);
        grid_size = dim3((output_anchors_ + block_size.x - 1) / block_size.x);
        decodeKernel<<<grid_size, block_size, 0, streams_[i]>>>(
            dev_transpose_ptr_ + offset_output, decode_ptr_ + offset_decode,
            output_channels_, output_anchors_, classes_);

        offset_output += output_channels_ * output_anchors_;
        offset_decode += sizeof(Detection) / sizeof(float) * output_anchors_;
    }

    for (int i = 0; i < batch_size_; ++i) {
        CUDA_CHECK_NOEXCEPT(cudaStreamSynchronize(streams_[i]));
    }

    std::vector<std::vector<Detection>> results(batch_size_);
    std::for_each(
        std::execution::par_unseq, results.begin(), results.end(),
        [&](std::vector<Detection>& result) {
            int i = &result - &results[0];
            std::span<Detection> detections(
                reinterpret_cast<Detection*>(decode_ptr_) + output_anchors_ * i,
                output_anchors_);

            std::vector<cv::Rect> bboxes;
            std::vector<float> scores;
            bboxes.reserve(detections.size());
            scores.reserve(detections.size());
            for (auto&& detection : detections) {
                bboxes.emplace_back(cv::Rect(detection.x, detection.y,
                                             detection.width,
                                             detection.height));
                scores.emplace_back(detection.confidence);
            }
            std::vector<int> indices;
            cv::dnn::NMSBoxes(bboxes, scores, conf_thresh_, nms_thresh_,
                              indices);

            const auto& pparam(pparams[i]);
            for (auto index : indices) {
                auto& detection = detections[index];
                restoreDetection(detection, pparam);
                result.emplace_back(detection);
            }
        });

    return results;
}

}  // namespace radar