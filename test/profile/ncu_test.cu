#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include "detection.h"

using radar::Detection;

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

__global__ void blobKernel(const unsigned char* src, float* dst, int width,
                           int height, float scale) {
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
}

__global__ void transposeKernel(const float* src, float* dst, int rows,
                                int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        dst[col * rows + row] = src[row * cols + col];
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

__device__ float IoU(float x1, float y1, float width1, float height1, float x2,
                     float y2, float width2, float height2) {
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

int main() {
    constexpr int image_width{2592}, image_height{2048};
    constexpr int input_width{640}, input_height{640}, input_channels{3};
    constexpr int output_channels{84}, output_anchors{8400};
    constexpr int block_dim(32);
    constexpr int classes{12};
    constexpr float nms_thresh{0.65}, score_thresh{0.50};

    unsigned char *dev_image_ptr, *dev_resize_ptr, *dev_border_ptr;
    float *dev_input_ptr, *dev_output_ptr, *dev_transpose_ptr, *dev_decode_ptr,
        *dev_nms_ptr;

    cudaMalloc(&dev_image_ptr, image_width * image_height * input_channels *
                                   sizeof(unsigned char));
    cudaMalloc(&dev_resize_ptr, input_width * input_height * input_channels *
                                    sizeof(unsigned char));
    cudaMalloc(&dev_border_ptr, input_width * input_height * input_channels *
                                    sizeof(unsigned char));
    cudaMalloc(&dev_input_ptr,
               input_width * input_height * input_channels * sizeof(float));
    cudaMalloc(&dev_output_ptr,
               output_channels * output_anchors * sizeof(float));
    cudaMalloc(&dev_transpose_ptr,
               output_channels * output_anchors * sizeof(float));
    cudaMalloc(&dev_decode_ptr, output_anchors * sizeof(Detection));
    cudaMalloc(&dev_nms_ptr, output_anchors * sizeof(Detection));

    cudaMemset(dev_output_ptr, 0,
               output_channels * output_anchors * sizeof(float));
    cudaMemset(
        dev_image_ptr, 0,
        image_width * image_height * input_channels * sizeof(unsigned char));

    dim3 block_size, grid_size;

    PreParam pparam(cv::Size(image_width, image_height),
                    cv::Size(input_width, input_height));
    float padding_width{pparam.width / pparam.ratio};
    float padding_height{pparam.height / pparam.ratio};
    block_size = dim3(block_dim, block_dim);
    grid_size = dim3((padding_width + block_size.x - 1) / block_size.x,
                     (padding_height + block_size.y - 1) / block_size.y);
    resizeKernel<<<grid_size, block_size>>>(
        dev_image_ptr, dev_resize_ptr, input_channels, pparam.width,
        pparam.height, padding_width, padding_height);

    int top{static_cast<int>(std::round(pparam.dh - 0.1))};
    int bottom{static_cast<int>(std::round(pparam.dh + 0.1))};
    int left{static_cast<int>(std::round(pparam.dw - 0.1))};
    int right{static_cast<int>(std::round(pparam.dw + 0.1))};
    grid_size = dim3((input_width + block_size.x - 1) / block_size.x,
                     (input_height + block_size.y - 1) / block_size.y);
    copyMakeBorderKernel<<<grid_size, block_size>>>(
        dev_resize_ptr, dev_border_ptr, input_channels, padding_width,
        padding_height, top, bottom, left, right);

    blobKernel<<<grid_size, block_size>>>(dev_border_ptr, dev_input_ptr,
                                          input_width, input_height, 1 / 255.f);

    grid_size = dim3((output_anchors + block_size.x - 1) / block_size.x,
                     (output_channels + block_size.y - 1) / block_size.y);
    transposeKernel<<<grid_size, block_size>>>(
        dev_output_ptr, dev_transpose_ptr, output_channels, output_anchors);

    block_size = dim3(block_dim);
    grid_size = dim3((output_anchors + block_size.x - 1) / block_size.x);
    decodeKernel<<<grid_size, block_size>>>(dev_transpose_ptr, dev_decode_ptr,
                                            output_channels, output_anchors,
                                            classes);

    block_size = dim3(block_dim * block_dim);
    grid_size = dim3((output_anchors + block_size.x - 1) / block_size.x,
                     (output_anchors + block_size.x - 1) / block_size.x);
    NMSKernel<<<grid_size, block_size, block_size.x * sizeof(Detection)>>>(
        dev_decode_ptr, nms_thresh, score_thresh, output_anchors);

    return EXIT_SUCCESS;
}