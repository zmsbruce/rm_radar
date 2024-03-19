#include <cuda_runtime.h>

#include "detect/detector.h"
#include "detection.h"

using namespace radar;

int main() {
    constexpr int image_width{2592}, image_height{2048};
    constexpr int input_width{640}, input_height{640}, input_channels{3};
    constexpr int output_channels{84}, output_anchors{8400};
    constexpr int block_dim(16);
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
    CUDA_CHECK(cudaGetLastError());

    cudaMemset(dev_output_ptr, 0,
               output_channels * output_anchors * sizeof(float));
    cudaMemset(
        dev_image_ptr, 0,
        image_width * image_height * input_channels * sizeof(unsigned char));
    CUDA_CHECK(cudaGetLastError());

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

    block_size = dim3(block_dim * block_dim);
    grid_size = dim3((output_anchors + block_size.x - 1) / block_size.x);
    decodeKernel<<<grid_size, block_size>>>(dev_transpose_ptr, dev_decode_ptr,
                                            output_channels, output_anchors,
                                            classes);

    grid_size = dim3((output_anchors + block_size.x - 1) / block_size.x,
                     (output_anchors + block_size.x - 1) / block_size.x);
    NMSKernel<<<grid_size, block_size, block_size.x * sizeof(Detection)>>>(
        dev_decode_ptr, nms_thresh, score_thresh, output_anchors);

    CUDA_CHECK(cudaGetLastError());

    return EXIT_SUCCESS;
}