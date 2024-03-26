#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "detect/detector.h"
#include "utils.h"

using namespace radar;

std::vector<std::string> kArmorNames{"B1", "B2", "B3", "B4", "B5", "R1",
                                     "R2", "R3", "R4", "R5", "Bs", "Rs"};

inline cv::Rect2f cv_rect(const Detection& detection) {
    return cv::Rect2f(detection.x, detection.y, detection.width,
                      detection.height);
}

inline void drawDetection(cv::Mat& image, std::string_view text,
                          const cv::Rect2f& rect, const cv::Scalar& color,
                          int line_width = 2, float font_scale = 1,
                          int font_thickness = 4) {
    cv::rectangle(image, rect, color, line_width);

    int base_line = 0;
    auto label_size = cv::getTextSize(text.data(), cv::FONT_HERSHEY_SIMPLEX,
                                      font_scale, font_thickness, &base_line);
    cv::Rect2f label_rect(rect.x, rect.y - label_size.height - base_line / 2,
                          label_size.width, label_size.height + base_line);
    cv::rectangle(image, label_rect, color, -1);
    cv::putText(image, text.data(), cv::Point(rect.x, rect.y),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255),
                font_thickness);
}

int main() {
    std::unique_ptr<Detector> car_detector{nullptr}, armor_detector{nullptr};
    car_detector = std::make_unique<Detector>("../models/car.engine", 1, 1);
    armor_detector =
        std::make_unique<Detector>("../models/armor.engine", 12, 20, 4);

    cv::Mat image = cv::imread("../assets/field.jpg", cv::IMREAD_COLOR);
    std::vector<cv::Mat> car_images;

    std::vector<radar::Detection> car_detections;
    std::vector<std::vector<radar::Detection>> armor_detections_batch;

    car_detections = car_detector->detect(image);
    car_images.reserve(car_detections.size());
    std::for_each(car_detections.begin(), car_detections.end(),
                  [&](const Detection& detection) {
                      car_images.push_back(image(cv_rect(detection)).clone());
                  });
    armor_detections_batch = armor_detector->detect(car_images);

    for (size_t i = 0; i < car_detections.size(); ++i) {
        const auto& car_detection{car_detections[i]};
        auto car_rect{cv_rect(car_detection)};

        auto car_text = cv::format("Car %.2f", car_detection.confidence);
        drawDetection(image, car_text, car_rect, cv::Scalar(0, 200, 0));

        const auto& armor_detections{armor_detections_batch[i]};
        for (const auto& detection : armor_detections) {
            auto armor_rect = cv_rect(detection);
            armor_rect.x += car_rect.x;
            armor_rect.y += car_rect.y;

            auto armor_text =
                cv::format("%s %.2f", kArmorNames[detection.label].c_str(),
                           detection.confidence);
            drawDetection(image, armor_text, armor_rect,
                          cv::Scalar(0, 200, 200));
        }
    }

    std::string window_name{"result"};
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, cv::Size(1296, 1024));
    cv::imshow(window_name, image);
    cv::waitKey(0);
    cv::destroyWindow(window_name);

    return EXIT_SUCCESS;
}