#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "detect/detector.h"
#include "utils.h"

using namespace radar;

inline cv::Rect2f cv_rect(const Detection& detection) {
    return cv::Rect2f(detection.x, detection.y, detection.width,
                      detection.height);
}

int main() {
    std::unique_ptr<Detector> car_detector{nullptr}, armor_detector{nullptr};
    car_detector = std::make_unique<Detector>("../models/car.engine", 1, 1);
    armor_detector =
        std::make_unique<Detector>("../models/armor.engine", 12, 20, 4);

    cv::VideoCapture capture("../assets/field.mp4");
    if (!capture.isOpened()) {
        throw std::runtime_error("video open failed");
    }

    cv::Mat image;
    std::vector<cv::Mat> car_images;
    std::vector<radar::Detection> car_detections;
    std::vector<std::vector<radar::Detection>> armor_detections_batch;

    std::string window_name{"result"};
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, cv::Size(1296, 1024));

    while (capture.read(image)) {
        TIME_IT({
            car_images.clear();
            car_detections = car_detector->detect(image);
            car_images.reserve(car_detections.size());
            std::for_each(
                car_detections.begin(), car_detections.end(),
                [&](const Detection& detection) {
                    car_images.push_back(image(cv_rect(detection)).clone());
                });
            armor_detections_batch = armor_detector->detect(car_images);
        });

        for (size_t i = 0; i < car_detections.size(); ++i) {
            const auto& car_detection{car_detections[i]};
            auto car_rect{cv_rect(car_detection)};

            cv::rectangle(image, car_rect, cv::Scalar(0, 255, 0));
            const auto& armor_detections{armor_detections_batch[i]};
            for (const auto& detection : armor_detections) {
                auto armor_rect = cv_rect(detection);
                armor_rect.x += car_rect.x;
                armor_rect.y += car_rect.y;

                cv::rectangle(image, armor_rect, cv::Scalar(0, 255, 255));
            }
        }

        cv::imshow(window_name, image);
        cv::waitKey(0);
    }

    cv::destroyWindow(window_name);

    return EXIT_SUCCESS;
}