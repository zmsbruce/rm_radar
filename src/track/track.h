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

namespace radar {

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
    friend class Tracker;
    /**
     * @brief Constructs a Track object with the given initial parameters.
     *
     * @param location Initial location of the track.
     * @param feature Initial feature vector associated with the track.
     * @param time Initial timestamp of the track.
     * @param track_id Unique identifier for the track.
     * @param filter Singer EKF model for the track.
     */
    Track(const cv::Point3f& location, const Eigen::VectorXf& feature,
          const std::chrono::high_resolution_clock::time_point& time,
          int track_id, float max_acc, float tau,
          const cv::Point3f observe_noise)
        : features_{feature},
          timestamp_{time},
          track_id_{track_id},
          init_count_{0},
          miss_count_{0},
          state_{TrackState::Tentative},
          filter_{nullptr} {
        Eigen::Matrix<float, track::kStateSize, 1> initial_state;
        initial_state << location.x, 0, 0, location.y, 0, 0, location.z, 0, 0;
        const Eigen::Matrix<float, track::kStateSize, track::kStateSize>
            initial_covariance = Eigen::Matrix<float, track::kStateSize,
                                               track::kStateSize>::Identity() *
                                 0.1f;
        Eigen::Matrix<float, track::kMeasurementSize, track::kMeasurementSize>
            observe_noise_mat;
        // clang-format off
        observe_noise_mat << observe_noise.x, 0, 0, 
                             0, observe_noise.y, 0, 
                             0, 0, observe_noise.z;
        // clang-format on 
        filter_ = std::make_unique<track::SingerEKF>(initial_state, initial_covariance,
                                              max_acc, tau, observe_noise_mat);
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
     * @brief Gets the current state of the track.
     *
     * @return The state of the track.
     */
    inline TrackState state() const noexcept { return state_; }

    /**
     * @brief Sets the current state of the track.
     * @param state The new state to set for the track.
     */
    inline void setState(TrackState state) noexcept { state_ = state; }

    /**
     * @brief Predicts the next state of the track using the Kalman filter.
     * @param current_timestamp The current timestamp to which the
     * prediction is made.
     */
    void predict(const std::chrono::high_resolution_clock::time_point&
                     current_timestamp) {
        // update ekf
        float dt = static_cast<float>(
                       std::chrono::duration_cast<std::chrono::nanoseconds>(
                           current_timestamp - timestamp_)
                           .count()) *
                   1e-9;
        filter_->predict(dt);

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

        Eigen::Matrix<float, track::kMeasurementSize, 1> measurement;
        measurement << location.x, location.y, location.z;
        filter_->update(measurement);
    }

    /**
     * @brief Gets the label of the track based on the maximum coefficient
     * in the feature sums.
     *
     * @return The label (index of the maximum coefficient) of the track.
     */
    inline int label() const noexcept { return features_.label(); }

    /**
     * @brief Gets the normalized feature of the track.
     *
     * @return The normalized feature of the track.
     */
    inline Eigen::VectorXf feature() const noexcept {
        return features_.feature();
    }

    /**
     * @brief Gets the location(x, y, z) of the track.
     *
     * @return the location of the track.
     */
    cv::Point3f location() const noexcept {
        auto state = filter_->state();
        return cv::Point3f(state(0), state(3), state(6));
    }

    friend std::ostream& operator<<(std::ostream& os, const Track& track) {
        std::cout << "Track: { ";
        std::cout << "id: " << track.track_id_ << ", ";
        std::cout << "label: " << track.label() << ", ";
        std::cout << "state: " << (track.state_ == TrackState::Confirmed ? "confirmed" : track.state_ == TrackState::Tentative ? "tentative" : "deleted") << ", ";
        std::cout << "init count: " << track.init_count_ << ", ";
        std::cout << "miss count: " << track.miss_count_;
        std::cout << " }";
        return os;
    }

   private:
    Track() = delete;

    track::Features features_;
    std::chrono::high_resolution_clock::time_point timestamp_;
    int track_id_;
    int init_count_;
    int miss_count_;
    TrackState state_;
    std::unique_ptr<track::SingerEKF> filter_;
};

}  // namespace radar