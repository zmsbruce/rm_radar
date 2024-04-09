#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "features.h"
#include "singer.h"

namespace radar::track {

class Track {
   public:
    enum class TrackState { Tentative, Confirmed, Deleted };
    using TimePoint = std::chrono::high_resolution_clock::time_point;

    Track(const cv::Point3f& location, const Eigen::VectorXf& feature,
          const TimePoint& time, int track_id, int init_thresh, int miss_thresh,
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

    inline bool isConfirmed() const noexcept {
        return state_ == TrackState::Confirmed;
    }

    inline bool isTentative() const noexcept {
        return state_ == TrackState::Tentative;
    }

    inline bool isDeleted() const noexcept {
        return state_ == TrackState::Deleted;
    }

    inline void setTrackState(TrackState state) noexcept { state_ = state; }

    void predict(const TimePoint& current_timestamp) {
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
    TimePoint timestamp_;
    int track_id_;
    int init_count_;
    int miss_count_;
    const int init_thresh_;
    const int miss_thresh_;
    TrackState state_;
    std::unique_ptr<SingerEKF> ekf_;
};

}  // namespace radar::track