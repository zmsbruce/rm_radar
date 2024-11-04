/**
 * @file robot.cpp
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the declaration of some robot functions.
 * @date 2024-03-27
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#include "robot.h"

#include <cmath>
#include <map>

#include "track/track.h"

namespace radar {

Robot::Robot(const Detection& car, const std::vector<Detection>& armors) {
    setDetection(car, armors);
}

void Robot::setDetection(const Detection& car,
                         const std::vector<Detection>& armors) noexcept {
    // Sets the bbox of car
    rect_ = cv::Rect2f(car.x, car.y, car.width, car.height);

    // If the armors are empty, sets empty label, confidence and return
    if (armors.empty()) {
        return;
    }

    // Calculates the highest score and its label, calculating the confidence of
    // the robot
    std::map<int, float> score_map;
    for (const auto& armor : armors) {
        score_map[armor.label] += armor.confidence;
    }
    auto [label, confidence] = *std::ranges::max_element(
        score_map, [&score_map](auto&& pair_a, auto&& pair_b) {
            return pair_a.second < pair_b.second;
        });
    confidence /= std::count_if(
        armors.begin(), armors.end(),
        [&](const Detection& armor) { return armor.label == label; });
    label_ = static_cast<Label>(label);
    confidence_ = confidence;

    // Sets the armor bboxes, adjusting their positions based on the position of
    // the car
    armors_ = armors;
    for (auto& armor : armors_.value()) {
        armor.x += car.x;
        armor.y += car.y;
    }
}

void Robot::setTrack(const Track& track) noexcept {
    track_state_ = track.state();
    if (track.isConfirmed()) {
        label_ = static_cast<Label>(track.label());
        location_ = track.location();
    } else {  // track is tentative
        if (!label_.has_value()) {
            label_ = static_cast<Label>(track.label());
        }
        if (!location_.has_value()) {
            location_ = track.location();
        }
    }
}

Eigen::VectorXf Robot::feature(int class_num) const noexcept {
    // Returns average if it is not detected
    Eigen::VectorXf feature(class_num);
    feature.setZero();
    if (!isDetected()) {
        return feature;
    }

    // Iterates through all armors and sets the feature vector based on the
    // label and confidence of each armor
    for (const auto& armor : armors_.value()) {
        feature(static_cast<int>(armor.label)) += armor.confidence;
    }

    // Normalize the vector
    float sum = feature.sum();
    if (iszero(sum)) {  // avoid division by zero
        return feature;
    }
    return feature / sum;
}

}  // namespace radar