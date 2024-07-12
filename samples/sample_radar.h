/**
 * @file sample_radar.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file is a sample using the provided libraries for detecting,
 * locating and tracking.
 * @date 2024-04-13
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <future>
#include <map>
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

/**
 * @brief A sample class of using provided libraries for detecting, locating
 * and tracking.
 *
 */
class SampleRadar {
   public:
    /**
     * @brief Constructor of the class
     *
     * @param car_path Filename of the car TensorRT engine.
     * @param armor_path Filename of the armor TensorRT engine.
     * @param image_size Size of input images.
     * @param intrinsic The intrinsic matrix.
     * @param lidar_to_camera The transform matrix(4x4) from lidar coordinate to
     * camera coordinate.
     * @param world_to_camera The transform matrix(4x4) from world coordinate to
     * camera coordinate.
     * @param lidar_noise The uncertainty of points provided by the lidar(m).
     * are the same.
     */
    SampleRadar(std::string_view car_path, std::string_view armor_path,
                cv::Size image_size, const cv::Matx33f& intrinsic,
                const cv::Matx44f lidar_to_camera,
                const cv::Matx44f& world_to_camera,
                const cv::Point3f& lidar_noise)
        : detector_(std::make_unique<RobotDetector>(
              car_path, armor_path, kClassNum, kMaxBatchSize, kOptBatchSize)),
          locator_(std::make_unique<Locator>(image_size.width,
                                             image_size.height, intrinsic,
                                             lidar_to_camera, world_to_camera)),
          tracker_(std::make_unique<Tracker>(lidar_noise, kClassNum)) {}

    std::vector<Robot> runOnce(const Frame& frame);

    void visualize(const Frame& frame, const std::vector<Robot>& robots);

    void updateBackgroundCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

   private:
    SampleRadar() = delete;

    std::unique_ptr<RobotDetector> detector_;
    std::unique_ptr<Locator> locator_;
    std::unique_ptr<Tracker> tracker_;
};

/**
 * @brief Updates the background depth map using the input cloud.
 *
 * @note This function is only used for demonstration using pre-processed
 * background point cloud stored on disk. Actually the acquisition of background
 * is accumulated by the input clouds.
 *
 * @param cloud The pointer of a point cloud containing the background.
 */
void SampleRadar::updateBackgroundCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    locator_->update(cloud);
}

/**
 * @brief The complete task of detecting, locating and tracking.
 *
 * @param frame A frame object including an image, a point cloud and a time
 * stamp.
 * @return The vector of robots detected.
 */
std::vector<Robot> SampleRadar::runOnce(const Frame& frame) {
    auto future_locate = std::async(std::launch::async, [&] {
        locator_->update(frame.point_cloud().value_or(nullptr));
        locator_->cluster();
    });

    auto future_detect = std::async(std::launch::async, [&] {
        return detector_->detect(frame.image().value_or(cv::Mat()));
    });

    future_locate.get();

    auto robots = future_detect.get();
    locator_->search(robots);

    tracker_->update(robots, frame.timestamp().value_or(
                                 std::chrono::high_resolution_clock::now()));

    visualize(frame, robots);

    return robots;
}

/**
 * @brief Gets the `cv::Scalar` color by the label of the input `Robot` object.
 *
 * @param robot The robot class.
 * @return the `cv::Scalar` color
 */
static cv::Scalar cvColor(const Robot& robot) {
    auto label = robot.label().value_or(-1);
    switch (label) {
        case Label::BlueHero:
        case Label::BlueEngineer:
        case Label::BlueInfantryThree:
        case Label::BlueInfantryFour:
        case Label::BlueInfantryFive:
            return cv::Scalar(255, 0, 0);
        case Label::RedHero:
        case Label::RedEngineer:
        case Label::RedInfantryThree:
        case Label::RedInfantryFour:
        case Label::RedInfantryFive:
            return cv::Scalar(0, 0, 255);
        default:
            return cv::Scalar(128, 128, 128);
    }
}

/**
 * @brief Visualize the result of detecting, locating and tracking on an image.
 * @param frame The input frame including an image, a point cloud and a time
 stamp.
 * @param robots The result of one cycle.
 */
void SampleRadar::visualize(const Frame& frame,
                            const std::vector<Robot>& robots) {
    static const std::unordered_map<int, std::string> robot_names{
        {Label::BlueHero, "B1"},          {Label::BlueEngineer, "B2"},
        {Label::BlueInfantryThree, "B3"}, {Label::BlueInfantryFour, "B4"},
        {Label::BlueInfantryFive, "B5"},  {Label::BlueSentry, "Bs"},
        {Label::RedHero, "R1"},           {Label::RedEngineer, "R2"},
        {Label::RedInfantryThree, "R3"},  {Label::RedInfantryFour, "R4"},
        {Label::RedInfantryFive, "R5"},   {Label::RedSentry, "Rs"}};
    const auto font = cv::FONT_HERSHEY_SIMPLEX;
    constexpr int thickness = 2;
    constexpr double font_size = 1.0;
    if (!frame.image().has_value()) {
        std::cerr << "frame image is empty" << std::endl;
        return;
    }
    cv::Mat image = frame.image().value().clone();
    for (const auto& robot : robots) {
        auto color = cvColor(robot);
        if (robot.rect().has_value()) {
            auto rect = robot.rect().value();
            cv::rectangle(image, rect, color, thickness);

            // label and confidence
            int base_line;
            auto text = cv::format(
                "%s %.2f",
                robot.label().has_value()
                    ? robot_names.at(robot.label().value()).c_str()
                    : "None",
                robot.confidence().has_value() ? robot.confidence().value()
                                               : 0.0f);
            auto label_size =
                cv::getTextSize(text, font, font_size, thickness, &base_line);
            cv::rectangle(
                image,
                cv::Rect(rect.x, rect.y - label_size.height - base_line,
                         label_size.width, label_size.height),
                color, -1);
            cv::putText(image, text, cv::Point(rect.x, rect.y - base_line),
                        font, font_size, cv::Scalar(255, 255, 255), thickness,
                        cv::LINE_AA);

            // location
            text = robot.location().has_value()
                       ? cv::format("%.1fm,%.1fm,%.1fm",
                                    robot.location().value().x,
                                    robot.location().value().y,
                                    robot.location().value().z)
                       : "None";
            label_size =
                cv::getTextSize(text, font, font_size, thickness, &base_line);
            cv::rectangle(image,
                          cv::Rect(rect.x, rect.y + rect.height + base_line,
                                   label_size.width, label_size.height),
                          color, -1);
            cv::putText(image, text,
                        cv::Point(rect.x, rect.y + rect.height +
                                              label_size.height + base_line),
                        font, font_size, cv::Scalar(255, 255, 255), thickness,
                        cv::LINE_AA);

            // track state
            text = robot.track_state().has_value()
                       ? robot.track_state().value() == TrackState::Confirmed
                             ? "Confirmed"
                         : robot.track_state().value() == TrackState::Tentative
                             ? "Tentative"
                             : "Deleted"
                       : "Not tracked";
            label_size =
                cv::getTextSize(text, font, font_size, thickness, &base_line);
            cv::rectangle(image,
                          cv::Rect(rect.x,
                                   rect.y + rect.height + label_size.height +
                                       base_line * 2,
                                   label_size.width, label_size.height),
                          color, -1);
            cv::putText(
                image, text,
                cv::Point(rect.x, rect.y + rect.height + label_size.height * 2 +
                                      base_line * 2),
                font, font_size, cv::Scalar(255, 255, 255), thickness,
                cv::LINE_AA);
        }
    }

    cv::namedWindow("image", cv::WINDOW_NORMAL);
    cv::resizeWindow("image", 1920, 1080);
    cv::imshow("image", image);
    cv::waitKey(0);
}