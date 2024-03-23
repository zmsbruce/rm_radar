#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "detect/detector.h"
#include "detect/detection.h"

using namespace radar;

int main() {
    std::unique_ptr<Detector> car_detector{nullptr}, armor_detector{nullptr};
    car_detector = std::make_unique<Detector>("../models/car.engine", 1, 1);
    armor_detector =
        std::make_unique<Detector>("../models/armor.engine", 12, 20, 4);

    cv::Mat image = cv::imread("../assets/field.jpg", cv::IMREAD_COLOR);
    std::vector<cv::Mat> car_images;

    std::vector<radar::Detection> car_detections;
    std::vector<std::vector<radar::Detection>> armor_detections_batch;

    for (int i = 0; i < 1000; ++i) {
        car_images.clear();
        car_detections.clear();
        armor_detections_batch.clear();

        car_detections = car_detector->detect(image);
        car_images.reserve(car_detections.size());
        std::for_each(car_detections.begin(), car_detections.end(),
                      [&](const Detection& detection) {
                          cv::Rect2f roi(detection.x, detection.y,
                                         detection.width, detection.height);
                          car_images.push_back(image(roi).clone());
                      });
        armor_detections_batch = armor_detector->detect(car_images);
    }

    return EXIT_SUCCESS;
}