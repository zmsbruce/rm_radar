/**
 * @file tracker.cpp
 * @author zmsbruce (zmsbruce@163.com)
 * @brief The file implements the functions of class Tracker, which is
 * responsible for managing and updating a set of tracks based on observations
 * of robots.
 * @date 2024-04-10
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#include "tracker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

#include "auction.h"

namespace radar::track {

/**
 * @brief The Tracker class is responsible for managing and updating a set of
 * tracks based on observations of robots.
 */
Tracker::Tracker(int init_thresh, int miss_thresh, float max_acceleration,
                 float acceleration_correlation_time,
                 const cv::Point3f& observation_noise, float distance_weight,
                 float feature_weight, int max_iter)
    : init_thresh_{init_thresh},
      miss_thresh_{miss_thresh},
      max_acc_{max_acceleration},
      tau_{acceleration_correlation_time},
      measurement_noise_{observation_noise},
      feature_weight_{feature_weight},
      distance_weight_{distance_weight},
      max_iter_{max_iter},
      init_filter_{initializeFilter()} {}

/**
 * @brief Calculate the Euclidean distance between two points in 3D space.
 *
 * @param p1 The first point.
 * @param p2 The second point.
 * @return The Euclidean distance.
 */
float Tracker::calculateDistance(const cv::Point3f& p1, const cv::Point3f& p2) {
    float x1 = p1.x, y1 = p1.y, z1 = p1.z;
    float x2 = p2.x, y2 = p2.y, z2 = p2.z;
    return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                     (z1 - z2) * (z1 - z2));
}

/**
 * @brief Calculate the cost associated with matching a track to a robot
 * observation.
 *
 * @param track The track for which to calculate the cost.
 * @param robot The robot observation to be matched to the track.
 * @return The calculated cost.
 */
float Tracker::calculateCost(const Track& track, const Robot& robot) {
    if (!robot.isLocated() || !robot.isDetected()) {
        return 0.0f;
    }

    // calculate distance score
    constexpr float acc_dis_thresh = 0.8f;
    float distance =
        calculateDistance(robot.location().value(), track.location());
    float distance_score = distance < acc_dis_thresh ? 1.0f
                           : distance < 2 * acc_dis_thresh
                               ? -1.25f * distance + 2.0f
                               : 0.0f;

    // calculate feature score
    auto feature_robot = robot.feature().value();
    auto feature_track = track.feature();
    assert(feature_robot.size() == feature_track.size());

    float feature_score = feature_robot.dot(feature_track) /
                          (feature_robot.norm() * feature_track.norm());
    feature_score = (feature_score + 1.0f) / 2.0f;

    return distance_score * distance_weight_ + feature_score * feature_weight_;
}

/**
 * @brief Initialize the filter used for state estimation.
 *
 * @return SingerEKF The initialized extended Kalman filter.
 */
SingerEKF Tracker::initializeFilter() const noexcept {
    const Eigen::Matrix<float, kStateSize, 1> initial_state =
        Eigen::Matrix<float, kStateSize, 1>::Zero();
    const Eigen::Matrix<float, kStateSize, kStateSize> initial_covariance =
        Eigen::Matrix<float, kStateSize, kStateSize>::Identity() * 0.1f;
    Eigen::Matrix<float, kMeasurementSize, kMeasurementSize>
        measurement_noise_mat;
    measurement_noise_mat << measurement_noise_.x, measurement_noise_.y,
        measurement_noise_.z;
    return SingerEKF(initial_state, initial_covariance, max_acc_, tau_,
                     measurement_noise_mat);
}

/**
 * @brief Update all tracks based on a new set of robot observations.
 *
 * @param robots The new robot observations.
 * @param timestamp The timestamp of the observations.
 */
void Tracker::update(
    std::vector<Robot>& robots,
    const std::chrono::high_resolution_clock::time_point& timestamp) {
    // Predicts tracks
    std::for_each(tracks_.begin(), tracks_.end(),
                  [&](Track& track) { track.predict(timestamp); });

    // Sets the cost matrix and calculates min-cost matching
    Eigen::MatrixXf cost_matrix(tracks_.size(), robots.size());
    for (size_t track_id = 0; track_id < tracks_.size(); ++track_id) {
        for (size_t robot_id = 0; robot_id < robots.size(); ++robot_id) {
            cost_matrix(track_id, robot_id) =
                calculateCost(tracks_[track_id], robots[robot_id]);
        }
    }
    auto match_result = auction(cost_matrix, max_iter_);

    // Sets the tracks based on the match result
    std::vector<int> matched_robot_indices;
    for (size_t track_id = 0; track_id < match_result.size(); ++track_id) {
        auto& track = tracks_[track_id];

        int robot_id = match_result[track_id];
        if (robot_id == kNotMatched) {
            if (track.isTentative()) {
                track.setState(TrackState::Deleted);
            } else if (track.isConfirmed()) {
                track.miss_count_ += 1;
                if (track.miss_count_ >= miss_thresh_) {
                    track.setState(TrackState::Deleted);
                }
            }
        } else {
            auto& robot = robots[robot_id];
            // Updates track
            track.update(robot.location().value(), robot.feature().value());
            if (track.isTentative()) {
                track.init_count_ += 1;
                if (track.init_count_ >= init_thresh_) {
                    track.setState(TrackState::Confirmed);
                }
                track.miss_count_ = 0;
            }
            // Updates robot
            robot.setTrack(track);
            // Appends to matched robot indices vector
            matched_robot_indices.push_back(robot_id);
        }
    }

    // Calculates unmatched robot indices
    std::vector<int> unmatched_robot_indices(robots.size());
    std::iota(unmatched_robot_indices.begin(), unmatched_robot_indices.end(),
              0);
    std::sort(matched_robot_indices.rbegin(), matched_robot_indices.rend());
    for (int index : matched_robot_indices) {
        unmatched_robot_indices.erase(unmatched_robot_indices.begin() + index);
    }

    // Initializes tracks from unmatched robots
    for (const auto& index : unmatched_robot_indices) {
        auto& robot = robots[index];
        // Ignore robots that are not detected or located
        if (robot.isDetected() && robot.isLocated()) {
            Track track(robot.location().value(), robot.feature().value(),
                        timestamp, latest_id_++, init_filter_);
            // Emplaces new track
            tracks_.emplace_back(track);
            // Updates robot
            robot.setTrack(track);
        }
    }

    // Removes deleted tracks
    auto iter_end =
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const Track& track) { return track.isDeleted(); });
    tracks_.erase(iter_end);
}

}  // namespace radar::track