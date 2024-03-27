/**
 * @file robot.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the definition and part of declaration of robot
 * class and its functions.
 * @date 2024-03-22
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <Eigen/Dense>
#include <map>
#include <opencv2/core/core.hpp>
#include <optional>
#include <ostream>
#include <ranges>
#include <utility>
#include <vector>

#include "detect/detection.h"

namespace radar {

class Track;

enum class TrackState;

/**
 * @brief Enumeration representing the labels for robot types.
 *
 * The Label enum defines the labels for different types of robots. Each label
 * corresponds to a specific robot type.
 */
enum Label {
    NoneType = -1,
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
    Robot() = default;

    Robot(const Detection& car, const std::vector<Detection>& armors);

    Robot(const Track& track);

    inline bool isDetected() const { return armors_.has_value(); }

    inline bool isTracked() const { return track_state_.has_value(); }

    inline bool isLocated() const { return location_.has_value(); }

    inline void setLocation(const cv::Point3f& location) noexcept {
        location_ = location;
    }

    inline std::optional<int> label() const {
        return isDetected() || isTracked() ? std::optional<int>(label_)
                                           : std::nullopt;
    }

    inline std::optional<cv::Rect> rect() const {
        return isDetected() || isTracked() ? std::optional<cv::Rect>(rect_)
                                           : std::nullopt;
    }

    inline std::optional<float> confidence() const {
        return isDetected() || isTracked() ? std::optional<float>(confidence_)
                                           : std::nullopt;
    }

    inline std::optional<std::vector<Detection>> armors() const {
        return isDetected() ? std::optional<std::vector<Detection>>(armors_)
                            : std::nullopt;
    }

    inline std::optional<TrackState> track_state() const {
        return isTracked() ? std::optional<TrackState>(track_state_)
                           : std::nullopt;
    }

    inline std::optional<cv::Point3f> location() const {
        return isLocated() ? std::optional<cv::Point3f>(location_)
                           : std::nullopt;
    }

    friend std::ostream& operator<<(std::ostream& os, const Robot& robot);

   private:
    std::optional<std::vector<Detection>> armors_ = std::nullopt;
    std::optional<TrackState> track_state_ = std::nullopt;
    std::optional<cv::Point3f> location_ = std::nullopt;
    cv::Rect2f rect_;
    int label_;
    float confidence_;
};

}  // namespace radar