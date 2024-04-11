#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "frame.h"
#include "radar.h"

using namespace radar;

static constexpr int kClassNum = 12;
static constexpr int kMaxBatchSize = 20;
static constexpr int kOptBatchSize = 4;

class SampleRadar {
   public:
    SampleRadar(std::string_view car_path, std::string_view armor_path,
                cv::Size image_size, const cv::Matx33f& intrinsic,
                const cv::Matx44f lidar_to_camera,
                const cv::Matx44f& world_to_camera,
                const cv::Point3f& lidar_noise)
        : car_detector_(std::make_unique<Detector>(car_path, 1, 1)),
          armor_detector_(std::make_unique<Detector>(
              armor_path, kClassNum, kMaxBatchSize, kOptBatchSize)),
          locator_(std::make_unique<Locator>(image_size.width,
                                             image_size.height, intrinsic,
                                             lidar_to_camera, world_to_camera)),
          tracker_(std::make_unique<Tracker>(lidar_noise)) {}

    std::vector<Robot> runOnce(const Frame& frame);

   private:
    std::vector<Robot> detect(const cv::Mat& image);
    SampleRadar() = delete;
    std::unique_ptr<Detector> car_detector_, armor_detector_;
    std::unique_ptr<Locator> locator_;
    std::unique_ptr<Tracker> tracker_;
};

std::vector<Robot> SampleRadar::detect(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "image is empty." << std::endl;
        return {};
    }

    std::vector<cv::Mat> car_images;
    auto car_detections = car_detector_->detect(image);
    car_images.reserve(car_detections.size());
    std::for_each(car_detections.begin(), car_detections.end(),
                  [&](const Detection& detection) {
                      car_images.emplace_back(std::move(
                          image(cv::Rect(detection.x, detection.y,
                                         detection.width, detection.height))
                              .clone()));
                  });
    auto armor_detections_batch = armor_detector_->detect(car_images);

    std::vector<Robot> robots;
    robots.reserve(car_detections.size());

    for (size_t i = 0; i < car_detections.size(); ++i) {
        Robot robot;
        robot.setDetection(car_detections[i], armor_detections_batch[i],
                           kClassNum);
        robots.emplace_back(std::move(robot));
    }

    return robots;
}

std::vector<Robot> SampleRadar::runOnce(const Frame& frame) {
    auto future_locate = std::async(std::launch::async, [&] {
        locator_->update(frame.point_cloud().value_or(nullptr));
    });

    auto future_detect = std::async(std::launch::async, [&] {
        return detect(frame.image().value_or(cv::Mat()));
    });

    future_locate.get();

    auto robots = future_detect.get();
    locator_->search(robots);

    tracker_->update(robots, frame.timestamp().value_or(
                                 std::chrono::high_resolution_clock::now()));

    return robots;
}

int main() {
    cv::Size image_size(2592, 2048);
    cv::Matx33f intrinsic(1685.51538398561, 0, 1278.99324114319, 0,
                          1685.26471848220, 1037.21273138299, 0, 0, 1);
    cv::Matx44f lidar_to_camera(0, -1, 0, 0.85443, 0, 0, -1, -37.6845, 1, 0, 0,
                                12.2631, 0.0, 0.0, 0.0, 1.0);
    cv::Matx44f world_to_camera(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    cv::Point3f lidar_noise(0.4, 0.4, 0.4);

    SampleRadar radar("../models/car.engine", "../models/armor.engine",
                      image_size, intrinsic, lidar_to_camera, world_to_camera,
                      lidar_noise);
    // TODO
    return EXIT_SUCCESS;
}