/**
 * @file robot.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the declaration of robot class and its functions.
 * @date 2024-03-22
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <map>
#include <opencv2/core/core.hpp>
#include <optional>
#include <ostream>
#include <ranges>
#include <utility>
#include <vector>

#include "detection.h"

namespace radar {

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

enum class TrackState { Tentative, Confirmed, Deleted };

class Robot {
   public:
    Robot() = default;

    /**
     * @brief Constructor of the `Robot` class, initializes the object based on
     * detection results.
     *
     * @param car The detected car information.
     * @param armors Vector of detected armor information.
     */
    Robot(const Detection& car, const std::vector<Detection>& armors) {
        setDetection(car, armors);
    }

    /**
     * @brief Set detection variables of `Robot` class based on detection
     * results.
     *
     * @param car The detected car information.
     * @param armors Vector of detected armor information.
     */
    void setDetection(const Detection& car,
                      const std::vector<Detection>& armors) {
        armors_ = armors;
        std::map<int, float> score_map;
        for (const auto& armor : armors) {
            score_map[armor.label] += armor.confidence;
        }
        std::tie(label_, confidence_) = *std::ranges::max_element(
            score_map, [&score_map](auto&& pair_a, auto&& pair_b) {
                return pair_a.second > pair_b.second;
            });
        rect_ = cv::Rect(car.x, car.y, car.width, car.height);
    }

    inline bool isDetected() const { return armors_.has_value(); }

    inline bool isTracked() const { return track_state_.has_value(); }

    inline bool isLocated() const { return location_.has_value(); }

    inline std::optional<int> label() const {
        return isDetected() ? std::optional<int>(label_) : std::nullopt;
    }

    inline std::optional<cv::Rect> rect() const {
        return isDetected() ? std::optional<cv::Rect>(rect_) : std::nullopt;
    }

    inline std::optional<float> confidence() const {
        return isDetected() ? std::optional<float>(confidence_) : std::nullopt;
    }

    inline std::optional<std::vector<Detection>> armors() const {
        return isDetected() ? std::optional<std::vector<Detection>>(armors_)
                            : std::nullopt;
    }

    friend std::ostream& operator<<(std::ostream& os, const Robot& robot) {
        os << "Robot: {\n";
        os << "    Label: "
           << (robot.isDetected() ? std::to_string(robot.label_) : "None")
           << ";\n";
        os << "    Rect: "
           << (robot.isDetected()
                   ? cv::format("[%d, %d, %d, %d]", robot.rect_.x,
                                robot.rect_.y, robot.rect_.width,
                                robot.rect_.height)
                   : "None")
           << ";\n";
        os << "    Confidence: "
           << (robot.isDetected() ? std::to_string(robot.confidence_) : "None")
           << ";\n";
        os << "    State: "
           << (!robot.isTracked()                            ? "None"
               : robot.track_state_ == TrackState::Confirmed ? "Confirmed"
               : robot.track_state_ == TrackState::Tentative ? "Tentative"
                                                             : "Deleted")
           << "\n";
        os << "}";
        return os;
    }

   private:
    std::optional<std::vector<Detection>> armors_ = std::nullopt;
    std::optional<TrackState> track_state_ = std::nullopt;
    std::optional<cv::Point3f> location_ = std::nullopt;
    cv::Rect rect_;
    int label_;
    float confidence_;
};

}  // namespace radar