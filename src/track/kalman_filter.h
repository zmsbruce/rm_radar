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
#include <tuple>
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
 * @class ExtendedKalmanFilter
 * @brief Implementation of an extended Kalman filter.
 *
 * This class implements an extended Kalman filter (EKF) for nonlinear state
 * estimation.
 *
 * @tparam StateSize The size of the state vector.
 * @tparam MeasurementSize The size of the measurement vector.
 * @tparam Args Variadic template for additional arguments for transition
 * functions.
 */
template <int StateSize, int MeasurementSize, typename... Args>
class ExtendedKalmanFilter final : public Kalman<StateSize, MeasurementSize> {
   public:
    using StateTransitionFunction =
        std::function<Eigen::Matrix<float, StateSize, StateSize>(
            const Eigen::Matrix<float, StateSize, 1>&, Args...)>;

    using ObservationFunction = std::function<
        std::pair<Eigen::Matrix<float, MeasurementSize, 1>,
                  Eigen::Matrix<float, MeasurementSize, StateSize>>(
            const Eigen::Matrix<float, StateSize, 1>&)>;

    /**
     * @brief Constructor for the extended Kalman filter.
     *
     * Initializes the filter with the initial state, covariance, process noise,
     * and observation noise matrices.
     *
     * @param initial_state The initial state vector.
     * @param initial_covariance The initial state covariance matrix.
     * @param process_noise The process noise covariance matrix.
     * @param observation_noise The observation noise covariance matrix.
     */
    ExtendedKalmanFilter(
        const Eigen::Matrix<float, StateSize, 1>& initial_state,
        const Eigen::Matrix<float, StateSize, StateSize>& initial_covariance,
        const Eigen::Matrix<float, StateSize, StateSize>& process_noise,
        const Eigen::Matrix<float, MeasurementSize, MeasurementSize>&
            observation_noise)
        : Kalman<StateSize, MeasurementSize>(initial_state, initial_covariance),
          process_noise_(process_noise),
          observation_noise_(observation_noise) {}

    /**
     * @brief Predicts the next state of the filter.
     *
     * This method predicts the next state based on the current state and the
     * state transition function.
     *
     * @param state_transition_function The state transition function.
     * @param args Additional arguments for the state transition function.
     */
    void predict(const StateTransitionFunction& state_transition_function,
                 Args... args) {
        transition_matrix_ = std::apply(state_transition_function,
                                        std::make_tuple(this->state_, args...));
        this->state_ = transition_matrix_ * this->state_;
        this->covariance_ = transition_matrix_ * this->covariance_ *
                                transition_matrix_.transpose() +
                            process_noise_;
    }

    /**
     * @brief Updates the filter with a new measurement.
     *
     * This method updates the filter state based on a new measurement and the
     observation function.
     *
     * @param measurement The new measurement vector.
     * @param observation_function The observation function.
     */
    void update(const Eigen::Matrix<float, MeasurementSize, 1>& measurement,
                const ObservationFunction& observation_function) {
        std::tie(predicted_measurement_, observation_matrix_) =
            observation_function(this->state_);
        update(measurement);
    }

   private:
    /**
     * @brief Internal predict method that performs the Kalman filter predict
     * step.
     *
     */
    void predict() override {
        this->state_ = transition_matrix_ * this->state_;
        this->covariance_ = transition_matrix_ * this->covariance_ *
                                transition_matrix_.transpose() +
                            process_noise_;
    }

    /**
     * @brief Internal update method that performs the Kalman filter update
     * step.
     *
     * This method updates the filter based on the predicted measurement and
     * measurement residual.
     *
     * @param measurement The new measurement vector.
     */
    void update(
        const Eigen::Matrix<float, MeasurementSize, 1>& measurement) override {
        const Eigen::Matrix<float, MeasurementSize, 1> measurement_residual =
            measurement - predicted_measurement_;
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

    Eigen::Matrix<float, StateSize, StateSize> transition_matrix_;
    Eigen::Matrix<float, StateSize, StateSize> process_noise_;
    Eigen::Matrix<float, MeasurementSize, StateSize> observation_matrix_;
    Eigen::Matrix<float, MeasurementSize, MeasurementSize> observation_noise_;
    Eigen::Matrix<float, MeasurementSize, 1> predicted_measurement_;
};

}  // namespace radar::track
