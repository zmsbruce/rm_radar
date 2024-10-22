/**
 * @file robot.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file is the header of robot class and its functions.
 * @date 2024-03-22
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "detect/detection.h"
#include "track/track.h"

namespace radar {

/**
 * @brief Enumeration representing the labels for robot types.
 *
 * The Label enum defines the labels for different types of robots. Each label
 * corresponds to a specific robot type.
 */
enum Label {
    BlueHero = 0,
    BlueEngineer = 1,
    BlueInfantryThree = 2,
    BlueInfantryFour = 3,
    BlueInfantryFive = 4,
    RedHero = 5,
    RedEngineer = 6,
    RedInfantryThree = 7,
    RedInfantryFour = 8,
    RedInfantryFive = 9,
    BlueSentry = 10,
    RedSentry = 11
};

/**
 * @brief Class representing a robot.
 *
 * The Robot class encapsulates information about a robot, including its
 * detections, tracking state, location, and other attributes.
 */
class Robot {
   public:
    Robot(const Detection& car, const std::vector<Detection>& armors);

    Robot() = default;

    /**
     * @brief Returns whether the robot is detected depending on
     * whether the armors are set.
     *
     * @return `true` if he robot is detected, otherwise `false`.
     */
    inline bool isDetected() const noexcept { return armors_.has_value(); }

    /**
     * @brief Returns whether the robot is tracked depending on whether the
     * location is set.
     *
     * @return `true` if he robot is located, otherwise `false`.
     */
    inline bool isLocated() const noexcept { return location_.has_value(); }

    /**
     * @brief Returns whether the robot is tracked depending on whether the
     * track state is set.
     *
     * @return `true` if he robot is tracked, otherwise `false`.
     */
    inline bool isTracked() const noexcept { return track_state_.has_value(); }

    void setDetection(const Detection& car,
                      const std::vector<Detection>& armors) noexcept;

    void setTrack(const Track& track) noexcept;

    /**
     * @brief Sets the location of the robot.
     *
     * @param location The location of the robot.
     */
    inline void setLocation(const cv::Point3f& location) noexcept {
        location_ = location * 1e-3;  // from millimeters to meters
    }

    /**
     * @brief Gets the label of the robot. If the robot has not been detected,
     * returns `std::nullopt`.
     *
     * @return The `std::optional` value of the label.
     */
    inline std::optional<int> label() const noexcept { return label_; }

    /**
     * @brief Gets the detection bbox of the robot. If the robot has not been
     * detected, returns `std::nullopt`.
     *
     * @return The `std::optional` value of the detection bbox.
     */
    inline std::optional<cv::Rect> rect() const noexcept { return rect_; }

    /**
     * @brief Gets the detection confidence of the robot. If the robot has not
     * been detected, returns `std::nullopt`.
     *
     * @return The `std::optional` value of the detection confidence.
     */
    inline std::optional<float> confidence() const noexcept {
        return confidence_;
    }

    /**
     * @brief Gets the armor detections of the robot. If the robot has not been
     * detected, returns `std::nullopt`.
     *
     * @return The `std::optional` value of the armor detections.
     */
    inline std::optional<std::vector<Detection>> armors() const noexcept {
        return armors_;
    }

    /**
     * @brief Gets the track state of the robot. If the robot has not been
     * tracked, returns `std::nullopt`.
     *
     * @return The `std::optional` value of the track state.
     */
    inline std::optional<TrackState> track_state() const noexcept {
        return track_state_;
    }

    /**
     * @brief Gets the detection bbox of the robot. If the robot has not been
     * located, returns `std::nullopt`.
     *
     * @return The `std::optional` value of the location.
     */
    inline std::optional<cv::Point3f> location() const noexcept {
        return location_;
    }

    Eigen::VectorXf feature(int class_num) const noexcept;

    friend std::ostream& operator<<(std::ostream& os, const Robot& robot);

   private:
    std::optional<std::vector<Detection>> armors_ = std::nullopt;
    std::optional<TrackState> track_state_ = std::nullopt;
    std::optional<cv::Point3f> location_ = std::nullopt;
    std::optional<cv::Rect2f> rect_ = std::nullopt;
    std::optional<int> label_ = std::nullopt;
    std::optional<float> confidence_ = std::nullopt;
};

}  // namespace radar