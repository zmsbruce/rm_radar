/**
 * @file kalman_filter.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief The file defines and implements algorithms of Kalman Filter(KF) and
 * Extended Kalman Filter(EKF).
 *
 * @date 2024-04-08
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <Eigen/Dense>
#include <functional>
#include <stdexcept>
#include <utility>

namespace radar::track {

/**
 * @brief Base class for Kalman filters with fixed state and measurement sizes.
 *
 * This class defines the interface for Kalman filters. It uses template
 * parameters to set the size of the state and measurement vectors at compile
 * time, thus allowing for optimized performance. Classes that derive from this
 * base must implement the predict and update methods.
 *
 * @tparam StateSize The size of the state vector.
 * @tparam MeasurementSize The size of the measurement vector.
 */
template <int StateSize, int MeasurementSize>
class Kalman {
   public:
    virtual void predict() = 0;

    virtual void update(
        const Eigen::Matrix<float, MeasurementSize, 1>& measurement) = 0;

    virtual ~Kalman() = default;

    inline auto state() { return state_; }

    inline auto covariance() { return covariance_; }

   protected:
    Kalman(const Eigen::Matrix<float, StateSize, 1>& initial_state,
           const Eigen::Matrix<float, StateSize, StateSize>& initial_covariance)
        : state_(initial_state), covariance_(initial_covariance) {}

    Eigen::Matrix<float, StateSize, 1> state_;
    Eigen::Matrix<float, StateSize, StateSize> covariance_;
};

/**
 * @brief Kalman Filter implementation for tracking applications.
 *
 * This class implements a Kalman filter for systems with a fixed state size and
 * measurement size. It uses template parameters to set the size of the state
 * and measurement vectors at compile time, thus allowing for optimized
 * performance.
 *
 * @tparam StateSize The size of the state vector.
 * @tparam MeasurementSize The size of the measurement vector.
 */
template <int StateSize, int MeasurementSize>
class KalmanFilter final : public Kalman<StateSize, MeasurementSize> {
   public:
    /**
     * @brief Construct a new Kalman Filter object.
     *
     * @param initial_state Initial state estimate vector.
     * @param initial_covariance Initial covariance matrix of the state
     * estimate.
     * @param transition_matrix State transition matrix.
     * @param process_noise Covariance matrix of the process noise.
     * @param observation_matrix Observation matrix.
     * @param observation_noise Covariance matrix of the observation noise.
     */
    KalmanFilter(
        const Eigen::Matrix<float, StateSize, 1>& initial_state,
        const Eigen::Matrix<float, StateSize, StateSize>& initial_covariance,
        const Eigen::Matrix<float, StateSize, StateSize>& transition_matrix,
        const Eigen::Matrix<float, StateSize, StateSize>& process_noise,
        const Eigen::Matrix<float, MeasurementSize, StateSize>&
            observation_matrix,
        const Eigen::Matrix<float, MeasurementSize, MeasurementSize>&
            observation_noise)
        : Kalman<StateSize, MeasurementSize>(initial_state, initial_covariance),
          transition_matrix_(transition_matrix),
          process_noise_(process_noise),
          observation_matrix_(observation_matrix),
          observation_noise_(observation_noise) {}

    /**
     * @brief Predict the state estimate and covariance to the next time step.
     */
    void predict() override {
        this->state_ = transition_matrix_ * this->state_;
        this->covariance_ = transition_matrix_ * this->covariance_ *
                                transition_matrix_.transpose() +
                            process_noise_;
    }

    /**
     * @brief Update the state estimate and covariance with a new measurement.
     *
     * @param measurement The measurement vector to incorporate in the state
     * estimate.
     */
    virtual void update(
        const Eigen::Matrix<float, MeasurementSize, 1>& measurement) override {
        const Eigen::Matrix<float, MeasurementSize, 1> predicted_measurement =
            observation_matrix_ * this->state_;
        const Eigen::Matrix<float, MeasurementSize, 1> measurement_residual =
            measurement - predicted_measurement;
        const Eigen::Matrix<float, MeasurementSize, MeasurementSize>
            innovation_covariance = observation_matrix_ * this->covariance_ *
                                        observation_matrix_.transpose() +
                                    observation_noise_;

        // Here we use a solver for numerical stability instead of calculating
        // the inverse
        const Eigen::Matrix<float, StateSize, MeasurementSize> kalman_gain =
            this->covariance_ * observation_matrix_.transpose() *
            innovation_covariance.ldlt().solve(
                Eigen::Matrix<float, MeasurementSize,
                              MeasurementSize>::Identity());

        this->state_ += kalman_gain * measurement_residual;

        const Eigen::Matrix<float, StateSize, StateSize> identity_matrix =
            Eigen::Matrix<float, StateSize, StateSize>::Identity();
        this->covariance_ =
            (identity_matrix - kalman_gain * observation_matrix_) *
            this->covariance_;
    }

   private:
    KalmanFilter() = delete;  // Prevents creating an empty KalmanFilter object.

    Eigen::Matrix<float, StateSize, StateSize> transition_matrix_;
    Eigen::Matrix<float, StateSize, StateSize> process_noise_;
    Eigen::Matrix<float, MeasurementSize, StateSize> observation_matrix_;
    Eigen::Matrix<float, MeasurementSize, MeasurementSize> observation_noise_;
};

/**
 * @brief Extended Kalman Filter class
 *
 * This class implements an Extended Kalman Filter for non-linear state
 * estimation. It uses non-linear state transition and observation functions to
 * perform the predict and update steps of the Kalman filter algorithm.
 *
 * @tparam StateSize The size of the state vector (the number of state
 * variables).
 * @tparam MeasurementSize The size of the measurement vector (the number of
 * measurements).
 */
template <int StateSize, int MeasurementSize>
class ExtendedKalmanFilter final : public Kalman<StateSize, MeasurementSize> {
   public:
    using StateTransitionFunction =
        std::function<Eigen::Matrix<float, StateSize, StateSize>(
            const Eigen::Matrix<float, StateSize, 1>&, float)>;
    using ObservationFunction = std::function<
        std::pair<Eigen::Matrix<float, MeasurementSize, 1>,
                  Eigen::Matrix<float, MeasurementSize, StateSize>>(
            const Eigen::Matrix<float, StateSize, 1>&)>;

    /**
     * @brief Constructs a new Extended Kalman Filter object
     *
     * @param initial_state Initial state estimate vector.
     * @param initial_covariance Initial covariance matrix of the state
     * estimate.
     * @param state_transition_function State transition matrix.
     * @param process_noise Covariance matrix of the process noise.
     * @param observation_function Function for the observation.
     * @param observation_noise Covariance matrix of the observation noise.
     */
    ExtendedKalmanFilter(
        const Eigen::Matrix<float, StateSize, 1>& initial_state,
        const Eigen::Matrix<float, StateSize, StateSize>& initial_covariance,
        const StateTransitionFunction& state_transition_function,
        const Eigen::Matrix<float, StateSize, StateSize>& process_noise,
        const ObservationFunction& observation_function,
        const Eigen::Matrix<float, MeasurementSize, MeasurementSize>&
            observation_noise)
        : Kalman<StateSize, MeasurementSize>(initial_state, initial_covariance),
          state_transition_function_(state_transition_function),
          process_noise_(process_noise),
          observation_function_(observation_function),
          observation_noise_(observation_noise) {}

    /**
     * @brief Sets time duration and predicts the next state estimate and
     * covariance.
     *
     * Uses the state transition function and process noise to predict the next
     * state and covariance.
     *
     * @param dt time duration used in state transition function.
     */
    void predict(float dt) {
        dt_ = dt;
        predict();
    }

    /**
     * @brief Updates the state estimate and covariance with a new measurement
     *
     * Uses the observation function and observation noise to incorporate a new
     * measurement into the state estimate.
     *
     * @param measurement The new measurement vector.
     */
    void update(
        const Eigen::Matrix<float, MeasurementSize, 1>& measurement) override {
        const auto [predicted_measurement, measurement_jacobian] =
            observation_function_(this->state_);
        const Eigen::Matrix<float, MeasurementSize, 1> measurement_residual =
            measurement - predicted_measurement;
        const Eigen::Matrix<float, MeasurementSize, MeasurementSize>
            innovation_covariance = measurement_jacobian * this->covariance_ *
                                        measurement_jacobian.transpose() +
                                    observation_noise_;

        // Here we use a solver for numerical stability instead of calculating
        // the inverse
        const Eigen::Matrix<float, StateSize, MeasurementSize> kalman_gain =
            this->covariance_ * measurement_jacobian.transpose() *
            innovation_covariance.ldlt().solve(
                Eigen::Matrix<float, MeasurementSize,
                              MeasurementSize>::Identity());

        this->state_ += kalman_gain * measurement_residual;

        const Eigen::Matrix<float, StateSize, StateSize> identity_matrix =
            Eigen::Matrix<float, StateSize, StateSize>::Identity();
        this->covariance_ =
            (identity_matrix - kalman_gain * measurement_jacobian) *
            this->covariance_;
    }

   private:
    /**
     * @brief Predicts the next state estimate and covariance
     *
     * Uses the state transition function and process noise to predict the next
     * state and covariance.
     */
    void predict() override {
        auto transition_matrix = state_transition_function_(this->state_, dt_);
        this->state_ = transition_matrix * this->state_;
        this->covariance_ = transition_matrix * this->covariance_ *
                                transition_matrix.transpose() +
                            process_noise_;
    }

    StateTransitionFunction state_transition_function_;
    Eigen::Matrix<float, StateSize, StateSize> process_noise_;
    ObservationFunction observation_function_;
    Eigen::Matrix<float, MeasurementSize, MeasurementSize> observation_noise_;
    float dt_;
};

}  // namespace radar::track
