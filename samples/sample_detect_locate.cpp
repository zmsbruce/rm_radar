#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <functional>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "detect/detection.h"
#include "detect/detector.h"
#include "locate/locator.h"
#include "robot/robot.h"

using namespace radar;
using Robots = std::vector<Robot>;

struct CvColor {
    static inline cv::Scalar Blue = cv::Scalar(255, 0, 0);
    static inline cv::Scalar Red = cv::Scalar(0, 0, 255);
    static inline cv::Scalar White = cv::Scalar(255, 255, 255);
    static inline cv::Scalar Black = cv::Scalar(0, 0, 0);
    static inline cv::Scalar Grey = cv::Scalar(128, 128, 128);
    static inline cv::Scalar LightRed = cv::Scalar(119, 47, 255);
    static inline cv::Scalar LightBlue = cv::Scalar(250, 200, 0);
    static inline cv::Scalar LightGrey = cv::Scalar(192, 192, 192);
};

std::unordered_map<int, std::string> kArmorNameMap{
    {Label::NoneType, "None"},       {Label::BlueHero, "B1"},
    {Label::BlueEngineer, "B2"},     {Label::BlueInfantryThree, "B3"},
    {Label::BlueInfantryFour, "B4"}, {Label::BlueInfantryFive, "B5"},
    {Label::RedHero, "R1"},          {Label::RedEngineer, "R2"},
    {Label::RedInfantryThree, "R3"}, {Label::RedInfantryFour, "R4"},
    {Label::RedInfantryFive, "R5"},  {Label::BlueSentry, "Bs"},
    {Label::RedSentry, "Rs"}};

Robots detect(const cv::Mat& image, std::shared_ptr<Detector> car_detector,
              std::shared_ptr<Detector> armor_detector);

void drawDetection(cv::Mat& image, std::string_view text,
                   const cv::Rect2f& rect, const cv::Scalar& color,
                   int line_width = 2, float font_scale = 1,
                   int font_thickness = 4);

cv::Rect2f cv_rect(const Detection& detection);

cv::Scalar cv_color(int label);

int main() {
    std::shared_ptr<Detector> car_detector{nullptr}, armor_detector{nullptr};
    car_detector = std::make_shared<Detector>("../models/car.engine", 1, 1);
    armor_detector =
        std::make_shared<Detector>("../models/armor.engine", 12, 20, 4);

    cv::Mat image = cv::imread("../assets/field_2.jpg", cv::IMREAD_COLOR);

    cv::Matx33f intrinsic(1685.51538398561, 0, 1278.99324114319, 0,
                          1685.26471848220, 1037.21273138299, 0, 0, 1);
    cv::Matx44f lidar_to_camera(0, -1, 0, 0.85443, 0, 0, -1, -37.6845, 1, 0, 0,
                                12.2631, 0.0, 0.0, 0.0, 1.0);
    // use angle directly from the camera
    cv::Matx44f world_to_camera(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    std::shared_ptr<Locator> locator(std::make_shared<Locator>(
        image.cols, image.rows, intrinsic, lidar_to_camera, world_to_camera));

    pcl::PointCloud<pcl::PointXYZ> cloud_1, cloud_2, cloud, cloud_background;
    pcl::io::loadPCDFile("../assets/field_2.pcd", cloud);
    pcl::io::loadPCDFile("../assets/field_2_background.pcd", cloud_background);
    pcl::io::loadPCDFile("../assets/field_2_1.pcd", cloud_1);
    pcl::io::loadPCDFile("../assets/field_2_2.pcd", cloud_2);

    // padding background and queue
    locator->update(cloud_background);
    locator->update(cloud_1);
    locator->update(cloud_2);

    // kernel function
    auto detect_result_future = std::async(
        std::launch::async,
        std::bind(::detect, std::cref(image), car_detector, armor_detector));
    auto update_cluster_future = std::async(std::launch::async, [&] {
        locator->update(cloud);
        locator->cluster();
    });
    auto robots = detect_result_future.get();
    update_cluster_future.get();

    locator->search(robots);

    for (const Robot& robot : robots) {
        if (!robot.isDetected()) {
            continue;
        }

        auto label_text = kArmorNameMap[robot.label().value()].c_str();
        auto pos_text = robot.location().has_value()
                            ? cv::format("(%.2fm, %.2fm, %.2fm)",
                                         robot.location().value().x * 1e-3,
                                         robot.location().value().y * 1e-3,
                                         robot.location().value().z * 1e-3)
                            : "unknown";
        auto text = cv::format("%s %s", label_text, pos_text.c_str());
        drawDetection(image, text, robot.rect().value(),
                      cv_color(robot.label().value()));
    }

    auto window_name = "result";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, cv::Size(1296, 1024));
    cv::imshow(window_name, image);
    cv::waitKey(0);
    cv::destroyWindow(window_name);

    return EXIT_SUCCESS;
}

Robots detect(const cv::Mat& image, std::shared_ptr<Detector> car_detector,
              std::shared_ptr<Detector> armor_detector) {
    std::vector<cv::Mat> car_images;
    std::vector<Detection> car_detections;
    std::vector<std::vector<Detection>> armor_detections_batch;

    car_detections = car_detector->detect(image);
    car_images.reserve(car_detections.size());
    std::for_each(car_detections.begin(), car_detections.end(),
                  [&](const Detection& detection) {
                      car_images.push_back(image(cv_rect(detection)).clone());
                  });
    armor_detections_batch = armor_detector->detect(car_images);

    Robots robots;
    robots.reserve(car_detections.size());

    assert(car_detections.size() == armor_detections_batch.size());
    for (size_t i = 0; i < car_detections.size(); ++i) {
        if (armor_detections_batch[i].empty()) {
            continue;
        }
        robots.emplace_back(car_detections[i], armor_detections_batch[i]);
    }

    return robots;
}

void drawDetection(cv::Mat& image, std::string_view text,
                   const cv::Rect2f& rect, const cv::Scalar& color,
                   int line_width, float font_scale, int font_thickness) {
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

cv::Rect2f cv_rect(const Detection& detection) {
    return cv::Rect2f(detection.x, detection.y, detection.width,
                      detection.height);
}

inline cv::Scalar cv_color(int label) {
    switch (label) {
        case Label::BlueHero:
        case Label::BlueEngineer:
        case Label::BlueInfantryThree:
        case Label::BlueInfantryFour:
        case Label::BlueInfantryFive:
        case Label::BlueSentry:
            return CvColor::Blue;
        case Label::RedHero:
        case Label::RedEngineer:
        case Label::RedInfantryThree:
        case Label::RedInfantryFour:
        case Label::RedInfantryFive:
        case Label::RedSentry:
            return CvColor::Red;
        default:
            return CvColor::Grey;
    }
}