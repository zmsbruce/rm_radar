/**
 * @file track.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file implements a Track class and its functions.
 * @date 2024-04-09
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "features.h"
#include "singer.h"

namespace radar::track {

/**
 * @brief Enum representing the state of the track.
 *
 */
enum class TrackState { Tentative, Confirmed, Deleted };

/**
 * @class Track
 * @brief Represents a track in a tracking system.
 *
 * This class encapsulates a track which is defined by its state (tentative,
 * confirmed, deleted), a Kalman filter for state estimation, and various
 * counters and thresholds for track management.
 */
class Track {
   public:
    /**
     * @brief Constructs a Track object with the given initial parameters.
     *
     * @param location Initial location of the track.
     * @param feature Initial feature vector associated with the track.
     * @param time Initial timestamp of the track.
     * @param track_id Unique identifier for the track.
     * @param init_thresh Threshold for the number of consecutive detections to
     * confirm the track.
     * @param miss_thresh Threshold for the number of missed detections to
     * delete the track.
     * @param max_accelaration Maximum acceleration used in the motion model of
     * the Kalman filter.
     * @param accelaration_correlation_time Correlation time of the acceleration
     * used in the motion model.
     * @param observation_noise Observation noise for the Kalman filter
     * measurements.
     */
    Track(const cv::Point3f& location, const Eigen::VectorXf& feature,
          const std::chrono::high_resolution_clock::time_point& time,
          int track_id, int init_thresh, int miss_thresh,
          float max_accelaration, float accelaration_correlation_time,
          const cv::Point3f& observation_noise)
        : features_{feature},
          timestamp_{time},
          track_id_{track_id},
          init_count_{0},
          miss_count_{0},
          init_thresh_{init_thresh},
          miss_thresh_{miss_thresh},
          state_{TrackState::Tentative} {
        const Eigen::Matrix<float, kStateSize, 1> initial_state =
            Eigen::Matrix<float, kStateSize, 1>::Zero();
        const Eigen::Matrix<float, kStateSize, kStateSize> initial_covariance =
            Eigen::Matrix<float, kStateSize, kStateSize>::Identity() * 0.1f;
        Eigen::Matrix<float, kMeasurementSize, kMeasurementSize>
            observation_noise_mat;
        observation_noise_mat << observation_noise.x, observation_noise.y,
            observation_noise.z;
        ekf_ = std::make_unique<SingerEKF>(
            initial_state, initial_covariance, max_accelaration,
            accelaration_correlation_time, observation_noise_mat);
    }

    /**
     * @brief Determines if the track is confirmed.
     * @return True if the track is confirmed, false otherwise.
     */
    inline bool isConfirmed() const noexcept {
        return state_ == TrackState::Confirmed;
    }

    /**
     * @brief Determines if the track is tentative.
     * @return True if the track is tentative, false otherwise.
     */
    inline bool isTentative() const noexcept {
        return state_ == TrackState::Tentative;
    }

    /**
     * @brief Determines if the track is deleted.
     * @return True if the track is deleted, false otherwise.
     */
    inline bool isDeleted() const noexcept {
        return state_ == TrackState::Deleted;
    }

    /**
     * @brief Sets the current state of the track.
     * @param state The new state to set for the track.
     */
    inline void setTrackState(TrackState state) noexcept { state_ = state; }

    /**
     * @brief Predicts the next state of the track using the Kalman filter.
     * @param current_timestamp The current timestamp to which the prediction is
     * made.
     */
    void predict(const std::chrono::high_resolution_clock::time_point&
                     current_timestamp) {
        // update ekf
        float dt = static_cast<float>(
                       std::chrono::duration_cast<std::chrono::nanoseconds>(
                           current_timestamp - timestamp_)
                           .count()) /
                   1e9;
        ekf_->predict(dt);

        // update counter and state
        if (isConfirmed()) {
            miss_count_ += 1;
            if (miss_count_ >= miss_thresh_) {
                state_ = TrackState::Deleted;
            }
        }

        // update timestamp
        timestamp_ = current_timestamp;
    }

    /**
     * @brief Updates the track with a new observation.
     * @param location The new observed location of the track.
     * @param feature The new observed feature associated with the track.
     */
    void update(const cv::Point3f& location, const Eigen::VectorXf& feature) {
        // update feature
        features_.push_back(feature);

        Eigen::Matrix<float, kMeasurementSize, 1> measurement;
        measurement << location.x, location.y, location.z;
        ekf_->update(measurement);

        // update counter and state
        if (isTentative()) {
            init_count_ += 1;
            if (init_count_ >= init_thresh_) {
                state_ = TrackState::Confirmed;
            }
        }
        miss_count_ = 0;
    }

   private:
    Track() = delete;

    Features features_;
    std::chrono::high_resolution_clock::time_point timestamp_;
    int track_id_;
    int init_count_;
    int miss_count_;
    const int init_thresh_;
    const int miss_thresh_;
    TrackState state_;
    std::unique_ptr<SingerEKF> ekf_;
};

}  // namespace radar::track