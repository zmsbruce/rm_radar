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
#include <ranges>

#include "auction.h"

namespace radar {

using namespace track;

/**
 * @brief Construct the Tracker class
 *
 * @param observation_noise The observation noise(m).
 * @param class_num The number of classes.
 * @param init_thresh Times needed to convert a track from tentative to
 * confirmed.
 * @param miss_thresh Times needed to mark a confirmed track to deleted.
 * @param max_acceleration Max acceleration(m/s^2) needed for the Singer-EKF
 * model.
 * @param acceleration_correlation_time Acceleration correlation time
 * constant(tau) of the Singer-EKF model.
 * @param distance_weight The weight of distance which is needed in min-cost
 * matching.
 * @param feature_weight The weight of feature which is needed in min-cost
 * matching.
 * @param max_iter The maximum iteration time of the auction algorithm.
 * @param distance_thresh The distance threshold(m) for scoring.
 */
Tracker::Tracker(const cv::Point3f& observation_noise, int class_num,
                 int init_thresh, int miss_thresh, float max_acceleration,
                 float acceleration_correlation_time, float distance_weight,
                 float feature_weight, int max_iter, float distance_thresh)
    : class_num_{class_num},
      init_thresh_{init_thresh},
      miss_thresh_{miss_thresh},
      max_acc_{max_acceleration},
      tau_{acceleration_correlation_time},
      distance_weight_{distance_weight},
      feature_weight_{feature_weight},
      measurement_noise_{observation_noise},
      max_iter_{max_iter},
      distance_thresh_{distance_thresh} {}

/**
 * @brief Calculate the weighted Euclidean distance between two points in 3D
 * space.
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
    if (!robot.isLocated() && !robot.isDetected()) {
        return 0.0f;
    }

    // calculate distance score
    float distance_score;
    if (!robot.isLocated()) {
        distance_score = 0.0f;
    } else {
        float distance =
            calculateDistance(robot.location().value(), track.location());
        distance_score = distance < distance_thresh_ ? 1.0f
                         : distance < 2 * distance_thresh_
                             ? -distance / distance_thresh_ + 2.0f
                             : 0.0f;
    }

    // calculate feature score
    auto feature_robot = robot.feature(class_num_);
    auto feature_track = track.feature();
    assert(feature_robot.size() == feature_track.size());

    float feature_score;
    float denom = feature_robot.norm() * feature_track.norm();
    if (iszero(denom)) {
        feature_score = 0.0f;
    } else {
        feature_score = feature_robot.dot(feature_track) / denom;
        feature_score = (feature_score + 1.0f) / 2.0f;
    }

    return distance_score * distance_weight_ + feature_score * feature_weight_;
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
    Eigen::MatrixXf cost_matrix(robots.size(), tracks_.size());
    for (size_t robot_id = 0; robot_id < robots.size(); ++robot_id) {
        for (size_t track_id = 0; track_id < tracks_.size(); ++track_id) {
            cost_matrix(robot_id, track_id) =
                calculateCost(tracks_[track_id], robots[robot_id]);
        }
    }

    std::vector<int> unmatched_robot_indices;
    std::vector<int> matched_track_indices;
    auto match_result = auction(cost_matrix, max_iter_);
    for (size_t robot_id = 0; robot_id < match_result.size(); ++robot_id) {
        auto& robot = robots[robot_id];
        if (robot.isLocated()) {
            unmatched_robot_indices.emplace_back(robot_id);
            continue;
        }

        int track_id = match_result[robot_id];
        if (track_id == kNotMatched) {
            // Updates unmatched robot indices
            unmatched_robot_indices.emplace_back(robot_id);
            continue;
        }

        auto& track = tracks_[track_id];
        //! The auction algorithm operates by ensuring that each agent is
        //! assigned a task, even if it means choosing tasks of very low
        //! value. So it is necessary to confirm whether the agent and task
        //! are appropriate to be assigned, otherwise the agent and task
        //! should be seen unassociated.
        if (calculateDistance(robot.location().value(), track.location()) >
                2 * distance_thresh_ &&
            robot.label().value_or(-1) != track.label()) {
            unmatched_robot_indices.emplace_back(robot_id);
            continue;
        }

        // Updates track
        track.update(robot.location().value(), robot.feature(class_num_));
        if (track.isTentative()) {
            track.init_count_ += 1;
            if (track.init_count_ >= init_thresh_) {
                track.setState(TrackState::Confirmed);
            }
            track.miss_count_ = 0;
        }
        // Updates robot
        robot.setTrack(track);
        // Marks the index of matched tracks
        matched_track_indices.emplace_back(track_id);
    }

    // Marks missed of unmatched tracks
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (std::ranges::find(matched_track_indices, i) ==
            matched_track_indices.end()) {
            auto& track = tracks_[i];
            if (track.isTentative()) {
                track.setState(TrackState::Deleted);
            } else if (track.isConfirmed()) {
                track.miss_count_ += 1;
                if (track.miss_count_ >= miss_thresh_) {
                    track.setState(TrackState::Deleted);
                }
            }
        }
    }

    // Erases deleted tracks
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const Track& track) { return track.isDeleted(); }),
        tracks_.end());

    // Appends new tracks
    std::ranges::for_each(unmatched_robot_indices, [&](int index) {
        auto& robot = robots[index];
        if (robot.isDetected() && robot.isLocated()) {
            Track track(robot.location().value(), robot.feature(class_num_),
                        timestamp, latest_id_++, max_acc_, tau_,
                        measurement_noise_);
            robot.setTrack(track);
            tracks_.emplace_back(std::move(track));
        }
    });
}

}  // namespace radar