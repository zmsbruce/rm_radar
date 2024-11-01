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

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <ranges>
#include <stdexcept>

#include "auction.h"

namespace radar {

using namespace track;

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
      distance_thresh_{distance_thresh} {
    spdlog::trace("Tracker constructor called");
    spdlog::debug(
        "Parameters: class_num = {}, init_thresh = {}, miss_thresh = {}, "
        "max_acceleration = {}, acceleration_correlation_time = {}, "
        "distance_weight = {}, feature_weight = {}, max_iter = {}, "
        "distance_thresh = {}, observation noise: [{}, {}, {}]",
        class_num, init_thresh, miss_thresh, max_acceleration,
        acceleration_correlation_time, distance_weight, feature_weight,
        max_iter, distance_thresh, observation_noise.x, observation_noise.y,
        observation_noise.z);
}

float Tracker::calculateDistance(const cv::Point3f& p1,
                                 const cv::Point3f& p2) noexcept {
    float x1 = p1.x, y1 = p1.y, z1 = p1.z;
    float x2 = p2.x, y2 = p2.y, z2 = p2.z;
    return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
                     (z1 - z2) * (z1 - z2));
}

float Tracker::calculateCost(const Track& track,
                             const Robot& robot) const noexcept {
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

void Tracker::update(
    std::vector<Robot>& robots,
    const std::chrono::high_resolution_clock::time_point& timestamp) noexcept {
    spdlog::trace("Tracker::update started with {} robots and {} tracks",
                  robots.size(), tracks_.size());

    // Predicts tracks
    spdlog::debug("Predicting tracks with the current timestamp");
    std::for_each(tracks_.begin(), tracks_.end(),
                  [&](Track& track) { track.predict(timestamp); });

    // Sets the cost matrix and calculates min-cost matching
    Eigen::MatrixXf cost_matrix(robots.size(), tracks_.size());
    spdlog::debug("Initialized cost matrix of size {}x{}", robots.size(),
                  tracks_.size());

    for (size_t robot_id = 0; robot_id < robots.size(); ++robot_id) {
        for (size_t track_id = 0; track_id < tracks_.size(); ++track_id) {
            cost_matrix(robot_id, track_id) =
                calculateCost(tracks_[track_id], robots[robot_id]);
            spdlog::trace("Cost between robot {} and track {}: {}", robot_id,
                          track_id, cost_matrix(robot_id, track_id));
        }
    }

    spdlog::debug("Starting auction-based matching with {} iterations",
                  max_iter_);
    std::vector<int> unmatched_robot_indices;
    std::vector<int> matched_track_indices;
    auto match_result = auction(cost_matrix, max_iter_);
    for (size_t robot_id = 0; robot_id < match_result.size(); ++robot_id) {
        auto& robot = robots[robot_id];
        if (!robot.isLocated()) {
            spdlog::debug("Robot {} is not located, adding to unmatched robots",
                          robot_id);
            unmatched_robot_indices.emplace_back(robot_id);
            continue;
        }

        int track_id = match_result[robot_id];
        if (track_id == kNotMatched) {
            spdlog::debug(
                "Robot {} did not match any track, adding to unmatched robots",
                robot_id);
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
        float distance =
            calculateDistance(robot.location().value(), track.location());
        spdlog::trace("Distance between robot {} and track {}: {}", robot_id,
                      track_id, distance);
        if (distance > 2 * distance_thresh_ &&
            robot.label().value_or(-1) != track.label()) {
            spdlog::debug(
                "Robot {} and track {} are not sufficiently close or have "
                "mismatched labels, adding to unmatched robots",
                robot_id, track_id);
            unmatched_robot_indices.emplace_back(robot_id);
            continue;
        }

        // Updates track
        spdlog::debug("Updating track {} with robot {} location and features",
                      track_id, robot_id);
        track.update(robot.location().value(), robot.feature(class_num_));
        if (track.isTentative()) {
            track.init_count_ += 1;
            if (track.init_count_ >= init_thresh_) {
                spdlog::trace("Track {} is now confirmed", track_id);
                track.setState(TrackState::Confirmed);
            }
        }
        track.miss_count_ = 0;
        // Updates robot
        robot.setTrack(track);
        // Marks the index of matched tracks
        matched_track_indices.emplace_back(track_id);
    }

    spdlog::debug("Processing unmatched tracks");
    // Marks missed of unmatched tracks
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (std::ranges::find(matched_track_indices, i) ==
            matched_track_indices.end()) {
            auto& track = tracks_[i];
            if (track.isTentative()) {
                spdlog::trace("Track {} is tentative and will be deleted", i);
                track.setState(TrackState::Deleted);
            } else if (track.isConfirmed()) {
                track.miss_count_ += 1;
                spdlog::trace("Track {} miss count incremented to {}", i,
                              track.miss_count_);
                if (track.miss_count_ >= miss_thresh_) {
                    spdlog::trace(
                        "Track {} is confirmed and has missed too many times, "
                        "setting state to Deleted",
                        i);
                    track.setState(TrackState::Deleted);
                }
            }
        }
    }

    // Erases deleted tracks
    size_t tracks_before = tracks_.size();
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const Track& track) { return track.isDeleted(); }),
        tracks_.end());
    size_t tracks_after = tracks_.size();
    spdlog::debug("Deleted {} tracks", tracks_before - tracks_after);

    // Appends new tracks
    spdlog::debug("Appending new tracks for unmatched robots");
    std::ranges::for_each(unmatched_robot_indices, [&](int index) {
        auto& robot = robots[index];
        if (robot.isDetected() && robot.isLocated()) {
            spdlog::trace("Creating new track for robot {}", index);
            Track track(robot.location().value(), robot.feature(class_num_),
                        timestamp, latest_id_++, max_acc_, tau_,
                        measurement_noise_);
            robot.setTrack(track);
            tracks_.emplace_back(std::move(track));
        }
    });

    spdlog::trace("Tracker::update completed");
}

}  // namespace radar