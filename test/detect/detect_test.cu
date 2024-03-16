#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "utils.h"

#define private public
#define protected public

#include "detect/detector.h"

#undef private
#undef protected

class PreProcessTest : public ::testing::Test {
   protected:
    unsigned char* d_src;

    int src_w, src_h, channels;
    cv::Mat src;
    dim3 block_size;

    virtual void SetUp() {
        block_size = dim3(16, 16);

        src = cv::Mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
        src_w = src.cols;
        src_h = src.rows;
        channels = src.channels();

        std::span<uchar> span(src.data, src.total() * src.elemSize());
        int value = 0;
        std::generate(span.begin(), span.end(), [&value] { return value++; });

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&d_src, src.total() * src.elemSize()));
        CUDA_CHECK(cudaMemcpy(d_src, src.data, src.total() * src.elemSize(),
                              cudaMemcpyHostToDevice));
    }

    virtual void TearDown() { CUDA_CHECK(cudaFree(d_src)); }
};

class ResizeTest : public PreProcessTest {
   protected:
    unsigned char* h_dst;

    void TestResize(float scale_x, float scale_y,
                    std::span<unsigned char> truth) {
        int dst_w = src_w * scale_x;
        int dst_h = src_h * scale_y;

        CUDA_CHECK(cudaHostAlloc(
            &h_dst, dst_w * dst_h * channels * sizeof(unsigned char),
            cudaHostAllocMapped));

        dim3 grid_size((dst_w + block_size.x - 1) / block_size.x,
                       (dst_h + block_size.y - 1) / block_size.y);
        radar::resizeKernel<<<grid_size, block_size>>>(
            d_src, h_dst, channels, src_w, src_h, dst_w, dst_h);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::span<unsigned char> dst(h_dst, dst_w * dst_h * channels);
        for (int i = 0; i < dst_w * dst_h * channels; ++i) {
            auto dst_value{dst[i]};
            auto truth_value{truth[i]};
            ASSERT_EQ(dst_value, truth_value);
        }

        CUDA_CHECK(cudaFreeHost(h_dst));
    }
};

TEST_F(ResizeTest, ResizeDouble) {
    std::vector<unsigned char> truth{
        0,  1,  2,  1,  2,  3,  3,  4,  5,  4,  5,  6,  6,  7,  8,  7,  8,  9,
        9,  10, 11, 9,  10, 11, 6,  7,  8,  7,  8,  9,  9,  10, 11, 10, 11, 12,
        12, 13, 14, 13, 14, 15, 15, 16, 17, 15, 16, 17, 12, 13, 14, 13, 14, 15,
        15, 16, 17, 16, 17, 18, 18, 19, 20, 19, 20, 21, 21, 22, 23, 21, 22, 23,
        18, 19, 20, 19, 20, 21, 21, 22, 23, 22, 23, 24, 24, 25, 26, 25, 26, 27,
        27, 28, 29, 27, 28, 29, 24, 25, 26, 25, 26, 27, 27, 28, 29, 28, 29, 30,
        30, 31, 32, 31, 32, 33, 33, 34, 35, 33, 34, 35, 30, 31, 32, 31, 32, 33,
        33, 34, 35, 34, 35, 36, 36, 37, 38, 37, 38, 39, 39, 40, 41, 39, 40, 41,
        36, 37, 38, 37, 38, 39, 39, 40, 41, 40, 41, 42, 42, 43, 44, 43, 44, 45,
        45, 46, 47, 45, 46, 47, 36, 37, 38, 37, 38, 39, 39, 40, 41, 40, 41, 42,
        42, 43, 44, 43, 44, 45, 45, 46, 47, 45, 46, 47};
    TestResize(2.0f, 2.0f, truth);
}

TEST_F(ResizeTest, ResizeHalf) {
    std::vector<unsigned char> truth{0, 1, 2, 6, 7, 8, 24, 25, 26, 30, 31, 32};
    TestResize(0.5f, 0.5f, truth);
}

class CopyMakeBorderTest : public PreProcessTest {
   protected:
    unsigned char* h_dst;

    void TestCopyMakeBorder(int top, int bottom, int left, int right,
                            std::span<unsigned char> truth) {
        int dst_w = src_w + left + right;
        int dst_h = src_h + top + bottom;

        CUDA_CHECK(cudaHostAlloc(
            &h_dst, dst_w * dst_h * channels * sizeof(unsigned char),
            cudaHostAllocMapped));

        dim3 grid_size((dst_w + block_size.x - 1) / block_size.x,
                       (dst_h + block_size.y - 1) / block_size.y);
        const std::vector<unsigned char> color(channels, 128);
        radar::copyMakeBorderKernel<<<grid_size, block_size>>>(
            d_src, h_dst, channels, src_w, src_h, top, bottom, left, right);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::span<unsigned char> dst(h_dst, dst_w * dst_h * channels);
        for (int i = 0; i < dst_w * dst_h * channels; ++i) {
            auto dst_value{dst[i]};
            auto truth_value{truth[i]};
            ASSERT_EQ(dst_value, truth_value);
        }

        CUDA_CHECK(cudaFreeHost(h_dst));
    }
};

TEST_F(CopyMakeBorderTest, CopyMakeBorder) {
    std::vector<unsigned char> truth{
        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 0,   1,   2,
        3,   4,   5,   6,   7,   8,   9,   10,  11,  128, 128, 128, 128, 128,
        128, 12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  128,
        128, 128, 128, 128, 128, 24,  25,  26,  27,  28,  29,  30,  31,  32,
        33,  34,  35,  128, 128, 128, 128, 128, 128, 36,  37,  38,  39,  40,
        41,  42,  43,  44,  45,  46,  47,  128, 128, 128, 128, 128, 128, 128,
        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
        128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
        128, 128, 128, 128};
    TestCopyMakeBorder(2, 2, 1, 1, truth);
}

class BlobTest : public PreProcessTest {
   protected:
    float* h_dst;

    void TestBlob(float scale) {
        CUDA_CHECK(cudaHostAlloc(&h_dst,
                                 src_w * src_h * channels * sizeof(float),
                                 cudaHostAllocMapped));

        dim3 grid_size((src_w + block_size.x - 1) / block_size.x,
                       (src_h + block_size.y - 1) / block_size.y);
        radar::blobKernel<<<grid_size, block_size>>>(d_src, h_dst, src_w, src_h,
                                                     scale);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::span<float> dst(h_dst, src_w * src_h * channels);
        auto truth =
            cv::dnn::blobFromImage(src, scale, cv::Size(), cv::Scalar(), true);
        for (int i = 0; i < src_w * src_h * channels; ++i) {
            ASSERT_EQ(dst[i], truth.ptr<float>()[i]);
        }

        CUDA_CHECK(cudaFreeHost(h_dst));
    }
};

TEST_F(BlobTest, Blob) {
    constexpr float scale{0.01};
    TestBlob(scale);
}

class DetectTest : public ::testing::Test {
   protected:
    std::string_view model_path{"../test/detect/yolov8n.engine"};
    std::unique_ptr<radar::Detector> detector{nullptr};

    virtual void SetUp() {
        detector = std::make_unique<radar::Detector>(model_path, 80, 10,
                                                     std::nullopt, 0.65f, 0.25f,
                                                     640, 640, "images", 3, 0);
    }
};

TEST_F(DetectTest, TestSingleBatchPreprocess) {
    std::string image_path{"../test/detect/bus.jpg"};
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    auto pparams = detector->preprocess(image);
    CUDA_CHECK(cudaStreamSynchronize(detector->streams_[0]));

    ASSERT_EQ(pparams.size(), 1);

    auto& pparam{pparams[0]};
    ASSERT_FLOAT_EQ(pparam.height, 1080);
    ASSERT_FLOAT_EQ(pparam.width, 810);
    ASSERT_FLOAT_EQ(pparam.dw, 80);
    ASSERT_FLOAT_EQ(pparam.dh, 0);
}

TEST_F(DetectTest, TestMultiBatchPreprocess) {
    cv::Mat image_bus = cv::imread("../test/detect/bus.jpg", cv::IMREAD_COLOR);
    cv::Mat image_zidane =
        cv::imread("../test/detect/zidane.jpg", cv::IMREAD_COLOR);

    std::vector<cv::Mat> images{image_bus, image_zidane};
    auto pparams = detector->preprocess(images);
    for (size_t i = 0; i < images.size(); ++i) {
        CUDA_CHECK(cudaStreamSynchronize(detector->streams_[i]));
    }

    ASSERT_EQ(pparams.size(), images.size());

    auto& pparam_bus{pparams[0]};
    ASSERT_FLOAT_EQ(pparam_bus.height, 1080);
    ASSERT_FLOAT_EQ(pparam_bus.width, 810);
    ASSERT_FLOAT_EQ(pparam_bus.dw, 80);
    ASSERT_FLOAT_EQ(pparam_bus.dh, 0);

    auto pparam_zidane{pparams[1]};
    ASSERT_FLOAT_EQ(pparam_zidane.height, 720);
    ASSERT_FLOAT_EQ(pparam_zidane.width, 1280);
    ASSERT_FLOAT_EQ(pparam_zidane.dw, 0);
    ASSERT_FLOAT_EQ(pparam_zidane.dh, 140);
}

TEST_F(DetectTest, TestSingleBatchDetect) {
    cv::Mat image_bus = cv::imread("../test/detect/bus.jpg", cv::IMREAD_COLOR);
    auto detections{detector->detect(image_bus)};
    ASSERT_EQ(detections.size(), 5);
}

TEST_F(DetectTest, TestMultiBatchDetect) {
    cv::Mat image_bus = cv::imread("../test/detect/bus.jpg", cv::IMREAD_COLOR);
    cv::Mat image_zidane =
        cv::imread("../test/detect/zidane.jpg", cv::IMREAD_COLOR);
    std::vector<cv::Mat> images{image_bus, image_zidane};
    auto detections{detector->detect(images)};
    ASSERT_EQ(detections.size(), 2);

    ASSERT_EQ(detections[0].size(), 5);
    ASSERT_EQ(detections[1].size(), 4);
}