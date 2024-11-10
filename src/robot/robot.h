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
 * @brief Class representing a robot.
 *
 * The Robot class encapsulates information about a robot, including its
 * detections, tracking state, location, and other attributes.
 */
class Robot {
   public:
    /**
     * @brief Enumeration representing the labels for robot types.
     *
     * The Label enum defines the labels for different types of robots. Each
     * label corresponds to a specific robot type.
     */
    enum class Label {
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
     * @brief Default robot constructor.
     */
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

    /**
     * @brief Set the detection variables of the robot, which include the
     * vector of armor detections as well as the label, confidence and bbox of
     * the robot.
     *
     * @param car The detected car information.
     * @param armors Vector of detected armor infomation.
     */
    void setDetection(const Detection& car,
                      const std::vector<Detection>& armors) noexcept;

    /**
     * @brief Sets the track state and the filtered location of the robot.
     *
     * @param track The track of the robot.
     */
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
    inline std::optional<Robot::Label> label() const noexcept {
        return label_.has_value()
                   ? std::make_optional<Robot::Label>(label_.value())
                   : std::nullopt;
    }

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

    /**
     * @brief Gets the feature of the robot, which is a vector containing
     * confidence of each class.
     *
     * @return The feature vector.
     */
    Eigen::VectorXf feature(int class_num) const noexcept;

   private:
    /**
     * @brief The detected armor information for the robot.
     *
     * This stores a vector of armor detections if the robot has been detected.
     * If the robot has not been detected, this value is `std::nullopt`.
     */
    std::optional<std::vector<Detection>> armors_ = std::nullopt;

    /**
     * @brief The tracking state of the robot.
     *
     * This stores the current tracking state of the robot, which includes its
     * predicted location and velocity. If the robot is not being tracked, this
     * value is `std::nullopt`.
     */
    std::optional<TrackState> track_state_ = std::nullopt;

    /**
     * @brief The 3D location of the robot in the world coordinate system.
     *
     * This stores the 3D location of the robot (in meters) if it has been
     * located. If the robot has not been located, this value is `std::nullopt`.
     */
    std::optional<cv::Point3f> location_ = std::nullopt;

    /**
     * @brief The bounding box of the robot in the image.
     *
     * This stores the 2D bounding box of the robot in the image if it has been
     * detected. If the robot has not been detected, this value is
     * `std::nullopt`.
     */
    std::optional<cv::Rect2f> rect_ = std::nullopt;

    /**
     * @brief The label of the robot.
     *
     * This stores the label of the robot, which indicates its type (e.g., hero,
     * engineer, infantry). If the robot has not been detected, this value is
     * `std::nullopt`.
     */
    std::optional<Label> label_ = std::nullopt;

    /**
     * @brief The detection confidence for the robot.
     *
     * This stores the confidence score for the robot's detection. If the robot
     * has not been detected, this value is `std::nullopt`.
     */
    std::optional<float> confidence_ = std::nullopt;
};

}  // namespace radar