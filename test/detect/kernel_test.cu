#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#define private public
#define protected public

#include "detect/detector.h"

#undef private
#undef protected

template <typename T, int Width, int Height, int Channels>
class KernelTest : public ::testing::Test {
   protected:
    int src_w = Width;
    int src_h = Height;
    int channels = Channels;
    T* h_src;

    virtual void SetUp() {
        size_t src_total = src_w * src_h * channels;

        CUDA_CHECK(cudaSetDevice(0));
        cudaHostAlloc(&h_src, src_total * sizeof(T), cudaHostAllocMapped);

        std::span<T> span(h_src, src_total);
        int value = 0;
        std::generate(span.begin(), span.end(), [&value] { return value++; });
    }

    virtual void TearDown() { CUDA_CHECK(cudaFreeHost(h_src)); }
};

class ResizeTest : public KernelTest<unsigned char, 4, 4, 3> {
   protected:
    unsigned char* h_dst;

    void TestResize(float scale_x, float scale_y,
                    std::span<unsigned char> truth) {
        int dst_w = src_w * scale_x;
        int dst_h = src_h * scale_y;

        CUDA_CHECK(cudaHostAlloc(
            &h_dst, dst_w * dst_h * channels * sizeof(unsigned char),
            cudaHostAllocMapped));

        dim3 block_size(16, 16);
        dim3 grid_size((dst_w + block_size.x - 1) / block_size.x,
                       (dst_h + block_size.y - 1) / block_size.y);
        radar::detect::resizeKernel<<<grid_size, block_size>>>(
            h_src, h_dst, channels, src_w, src_h, dst_w, dst_h);
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

class CopyMakeBorderTest : public KernelTest<unsigned char, 4, 4, 3> {
   protected:
    unsigned char* h_dst;

    void TestCopyMakeBorder(int top, int bottom, int left, int right,
                            std::span<unsigned char> truth) {
        int dst_w = src_w + left + right;
        int dst_h = src_h + top + bottom;

        CUDA_CHECK(cudaHostAlloc(
            &h_dst, dst_w * dst_h * channels * sizeof(unsigned char),
            cudaHostAllocMapped));

        dim3 block_size(16, 16);
        dim3 grid_size((dst_w + block_size.x - 1) / block_size.x,
                       (dst_h + block_size.y - 1) / block_size.y);
        const std::vector<unsigned char> color(channels, 128);
        radar::detect::copyMakeBorderKernel<<<grid_size, block_size>>>(
            h_src, h_dst, channels, src_w, src_h, top, bottom, left, right);
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

class BlobTest : public KernelTest<unsigned char, 4, 4, 3> {
   protected:
    float* h_dst;

    void TestBlob(float scale) {
        CUDA_CHECK(cudaHostAlloc(&h_dst,
                                 src_w * src_h * channels * sizeof(float),
                                 cudaHostAllocMapped));

        dim3 block_size(16, 16);
        dim3 grid_size((src_w + block_size.x - 1) / block_size.x,
                       (src_h + block_size.y - 1) / block_size.y);
        radar::detect::blobKernel<<<grid_size, block_size>>>(
            h_src, h_dst, src_w, src_h, channels, scale);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::span<float> dst(h_dst, src_w * src_h * channels);
        auto src_mat = cv::Mat(cv::Size(src_w, src_h), CV_8UC3, h_src);
        auto truth = cv::dnn::blobFromImage(src_mat, scale, cv::Size(),
                                            cv::Scalar(), true);
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

class TransposeTest : public KernelTest<float, 36, 2, 1> {
    float* h_dst;

   protected:
    void TestTranspose() {
        CUDA_CHECK(cudaHostAlloc(&h_dst,
                                 src_w * src_h * channels * sizeof(float),
                                 cudaHostAllocMapped));

        dim3 block_size(32, 32);
        dim3 grid_size((src_w + block_size.x - 1) / block_size.x,
                       (src_h + block_size.y - 1) / block_size.y);
        radar::detect::transposeKernel<<<grid_size, block_size>>>(h_src, h_dst,
                                                                  src_h, src_w);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::span<float> dst(h_dst, src_w * src_h * channels);

        auto src_mat = cv::Mat(cv::Size(src_w, src_h), CV_32FC1, h_src);
        cv::Mat dst_mat;
        cv::transpose(src_mat, dst_mat);
        for (int i = 0; i < src_w * src_h * channels; ++i) {
            ASSERT_EQ(dst[i], dst_mat.ptr<float>()[i]);
        }

        CUDA_CHECK(cudaFreeHost(h_dst));
    }
};

TEST_F(TransposeTest, Transpose) { TestTranspose(); }
