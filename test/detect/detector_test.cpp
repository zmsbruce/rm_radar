#include <gtest/gtest.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <string_view>
#include <vector>

#define private public
#define protected public

#include "detect/detector.h"

#undef private
#undef protected

class DetectTest : public ::testing::Test {
   protected:
    std::string_view model_path{"../test/detect/yolov8n.engine"};
    std::unique_ptr<radar::Detector> detector{nullptr};

    virtual void SetUp() {
        detector = std::make_unique<radar::Detector>(
            model_path, 80, 10, std::nullopt, 1 << 24, 0.65f, 0.25f, 640, 640,
            "images", 3, 0);
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
    ASSERT_GT(detections.size(), 0);
    ASSERT_LT(detections.size(), 10);
}

TEST_F(DetectTest, TestMultiBatchDetect) {
    cv::Mat image_bus = cv::imread("../test/detect/bus.jpg", cv::IMREAD_COLOR);
    cv::Mat image_zidane =
        cv::imread("../test/detect/zidane.jpg", cv::IMREAD_COLOR);
    std::vector<cv::Mat> images{image_bus, image_zidane};
    auto detections{detector->detect(images)};
    ASSERT_EQ(detections.size(), 2);

    ASSERT_GT(detections[0].size(), 0);
    ASSERT_LT(detections[0].size(), 10);

    ASSERT_GT(detections[1].size(), 0);
    ASSERT_LT(detections[1].size(), 10);
}