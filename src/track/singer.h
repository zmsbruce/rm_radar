/**
 * @file singer.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief A file implementing an Extended Kalman Filter for the Singer model.
 * @date 2024-04-09
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <cmath>

#include "kalman_filter.h"

namespace radar::track {

constexpr int kStateSize = 9;
constexpr int kMeasurementSize = 3;

/**
 * @class SingerEKF
 * @brief Implementation of an Extended Kalman Filter for the Singer model.
 *
 * This class provides an implementation of an Extended Kalman Filter (EKF)
 * designed for tracking 3d coordinates using the Singer model, which assumes
 * that target acceleration is a random process. In this model, state is defined
 * as [x, vx, ax, y, vy, ay, z, vz, az], and observation is defined as [x, y,
 * z].
 */
class SingerEKF {
   public:
    using EKF = ExtendedKalmanFilter<kStateSize, kMeasurementSize, float>;

    /**
     * @brief Constructor to initialize the Extended Kalman Filter.
     *
     * @param initial_state Initial state vector.
     * @param initial_covariance Initial covariance matrix.
     * @param max_a Maximum expected acceleration of the target.
     * @param tau Correlation time constant.
     * @param observation_noise Observation noise covariance matrix.
     */
    SingerEKF(
        const Eigen::Matrix<float, kStateSize, 1>& initial_state,
        const Eigen::Matrix<float, kStateSize, kStateSize>& initial_covariance,
        float max_a, float tau,
        const Eigen::Matrix<float, kMeasurementSize, kMeasurementSize>&
            observation_noise)
        : ekf_(initial_state, initial_covariance, observation_noise),
          max_a_(max_a),
          tau_(tau) {}

    /**
     * @brief Predict the state of the filter forward by a time increment.
     *
     * @param dt Time increment for prediction step.
     */
    inline void predict(float dt) {
        ekf_.predict(state_transition_, process_noise_, dt);
    }

    /**
     * @brief Update the filter state with a new measurement.
     *
     * @param measurement New measurement vector.
     */
    inline void update(
        const Eigen::Matrix<float, kMeasurementSize, 1>& measurement) {
        ekf_.update(measurement, observation_);
    }

    /**
     * @brief Get the state of the filter.
     *
     * @return The state of the filter.
     */
    inline auto state() const { return ekf_.state(); }

   private:
    SingerEKF() = delete;

    EKF ekf_;
    float max_a_;
    float tau_;
    EKF::StateTransitionFunction state_transition_ =
        [this](
            [[maybe_unused]] const Eigen::Matrix<float, kStateSize, 1>& state,
            float dt) {
            Eigen::Matrix<float, kStateSize, kStateSize> transition_matrix =
                Eigen::Matrix<float, kStateSize, kStateSize>::Identity();
            for (int i = 0; i < 3; ++i) {
                transition_matrix(i * 3, i * 3 + 1) = dt;
                transition_matrix(i * 3, i * 3 + 2) = dt * dt / 2;
                transition_matrix(i * 3 + 1, i * 3 + 2) = dt;
                transition_matrix(i * 3 + 2, i * 3 + 2) = std::exp(-dt / tau_);
            }
            return transition_matrix;
        };
    EKF::ProcessNoiseFunction process_noise_ = [this](float dt) {
        Eigen::Matrix<float, kStateSize, kStateSize> process_noise =
            Eigen::Matrix<float, kStateSize, kStateSize>::Zero();
        for (int i = 0; i < 3; i++) {
            process_noise(3 * i, 3 * i) = std::pow(dt, 3) / 3;
            process_noise(3 * i + 1, 3 * i) = std::pow(dt, 2) / 2;
            process_noise(3 * i + 2, 3 * i) = dt / 2;
            process_noise(3 * i, 3 * i + 1) = std::pow(dt, 2) / 2;
            process_noise(3 * i + 1, 3 * i + 1) = dt;
            process_noise(3 * i + 2, 3 * i + 1) = 1 - std::exp(-dt / tau_);
            process_noise(3 * i, 3 * i + 2) = dt / 2;
            process_noise(3 * i + 1, 3 * i + 2) = 1 - std::exp(-dt / tau_);
            process_noise(3 * i + 2, 3 * i + 2) =
                (1 - std::exp(-2 * dt / tau_)) / 2;
        }
        process_noise *= std::pow(max_a_, 2);
        return process_noise;
    };
    EKF::ObservationFunction observation_ =
        [](const Eigen::Matrix<float, kStateSize, 1>& state) {
            Eigen::Matrix<float, kMeasurementSize, 1> measurement;
            Eigen::Matrix<float, kMeasurementSize, kStateSize>
                measurement_jacobian =
                    Eigen::Matrix<float, kMeasurementSize, kStateSize>::Zero();
            for (int i = 0; i < kMeasurementSize; ++i) {
                measurement(i) = state[i * 3];
                measurement_jacobian(i, i * 3) = 1;
            }
            return std::make_pair(measurement, measurement_jacobian);
        };
};

}  // namespace radar::track