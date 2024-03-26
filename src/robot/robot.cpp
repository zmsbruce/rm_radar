#include "robot.h"

#include "track/track.h"

namespace radar {

/**
 * @brief Constructor of the `Robot` class, initializes the object based on
 * detection results.
 *
 * @param car The detected car information.
 * @param armors Vector of detected armor information.
 */
Robot::Robot(const Detection& car, const std::vector<Detection>& armors)
    : armors_{armors} {
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

    rect_ = cv::Rect2f(car.x, car.y, car.width, car.height);

    assert(armors_.has_value());
    for (auto& armor : armors_.value()) {
        armor.x += rect_.x;
        armor.y += rect_.y;
    }
}

/**
 * @brief Constructor of the `Robot` class, initializes the object based on
 * track information.
 * @param track The tracked information.
 */
Robot::Robot(const Track& track) : track_state_{track.state} {
    Eigen::Matrix<float, 1, 12, Eigen::RowMajor> sum =
        track.features.colwise().sum();
    sum.maxCoeff(&label_);

    track::DETECTBOX detect_box = track.mean.leftCols(4);
    detect_box(2) *= detect_box(3);
    detect_box.leftCols(2) -= (detect_box.rightCols(2) / 2);
    rect_ =
        cv::Rect2f(detect_box(0), detect_box(1), detect_box(2), detect_box(3));
}

/**
 * @brief Overloaded output stream operator for the Robot class, printing label,
 * rect, confidence, state and location.
 * @param os The output stream.
 * @param robot The Robot object to be printed.
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
       << (!robot.isTracked()                            ? "None"
           : robot.track_state_ == TrackState::Confirmed ? "Confirmed"
           : robot.track_state_ == TrackState::Tentative ? "Tentative"
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