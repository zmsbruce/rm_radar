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

using Track = track::Track;

/**
 * @brief Set the detection variables of the robot, which includes the
 * vector of armor detections as well as the label, confidence and bbox of
 * the robot.
 *
 * @param car The detected car information.
 * @param armors Vector of detected armor infomation.
 */
void Robot::setDetection(const Detection& car,
                         const std::vector<Detection>& armors) noexcept {
    // Sets armor detections
    armors_ = armors;

    // Calculates the highest score and its label, calculating the confidence of
    // the robot
    std::map<int, float> score_map;
    for (const auto& armor : armors) {
        score_map[armor.label] += armor.confidence;
    }
    std::tie(label_, confidence_) = *std::ranges::max_element(
        score_map, [&score_map](auto&& pair_a, auto&& pair_b) {
            return pair_a.second > pair_b.second;
        });
    confidence_ /= std::count_if(
        armors.begin(), armors.end(),
        [this](const Detection& armor) { return armor.label == label_; });

    // Sets the bbox of car
    rect_ = cv::Rect2f(car.x, car.y, car.width, car.height);

    // Sets the armor bboxes, adjusting their positions based on the position of
    // the car
    armors_ = armors;
    for (auto& armor : armors_.value()) {
        armor.x += rect_.x;
        armor.y += rect_.y;
    }
}

/**
 * @brief Sets the track state and the filtered location of the robot.
 *
 * @param track The track of the robot.
 */
void Robot::setTrack(const Track& track) noexcept {
    track_state_ = track.state();
    label_ = track.label();
    location_ = track.location();
}

/**
 * @brief Gets the feature of the robot, which is a vector containing
 * confidence of each class. If the robot has not been detected, return
 * `std::nullopt`.
 *
 * @return The `std::optional` value of the feature vector.
 */
std::optional<Eigen::VectorXf> Robot::feature() const noexcept {
    // Returns `std::nullopt` if it is not detected
    if (!isDetected()) {
        return std::nullopt;
    }

    // Iterates through all armors and sets the feature vector based on the
    // label and confidence of each armor
    Eigen::VectorXf feature(class_num_);
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

/**
 * @brief Overloaded output stream operator for the Robot class, printing
 * label, rect, confidence, state and location.
 *
 * @param os The output stream.
 * @param robot The Robot object to be printed.
 *
 * @return The output stream.
 */
std::ostream& operator<<(std::ostream& os, const Robot& robot) {
    os << "Robot: {\n";
    os << "    Label: "
       << (robot.isDetected() ? std::to_string(robot.label_) : "None") << "\n";
    os << "    Rect: "
       << (robot.isDetected()
               ? cv::format("[%f, %f, %f, %f]", robot.rect_.x, robot.rect_.y,
                            robot.rect_.width, robot.rect_.height)
               : "None")
       << "\n";
    os << "    Confidence: "
       << (robot.isDetected() ? std::to_string(robot.confidence_) : "None")
       << "\n";
    os << "    State: "
       << (!robot.isTracked()                                    ? "None"
           : robot.track_state_.value() == TrackState::Confirmed ? "Confirmed"
           : robot.track_state_.value() == TrackState::Tentative ? "Tentative"
                                                                 : "Deleted")
       << "\n";
    os << "    Location: "
       << (robot.isLocated()
               ? cv::format("[%f, %f, %f]", robot.location_.value().x,
                            robot.location_.value().y,
                            robot.location_.value().z)
               : "None")
       << "\n";
    os << "}";
    return os;
}

}  // namespace radar