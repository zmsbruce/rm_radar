#include <gtest/gtest.h>

#include "track/kalman_filter.h"

using namespace radar::track;

class ExtendedKalmanFilterTest : public ::testing::Test {
   protected:
    static constexpr int kStateSize = 4;
    static constexpr int kMeasurementSize = 2;

    Eigen::Matrix<float, kStateSize, 1> initial_state;
    Eigen::Matrix<float, kStateSize, kStateSize> initial_covariance;
    ExtendedKalmanFilter<kStateSize, kMeasurementSize,
                         float>::ProcessNoiseFunction process_noise_function;
    Eigen::Matrix<float, kMeasurementSize, kMeasurementSize> observation_noise;
    ExtendedKalmanFilter<kStateSize, kMeasurementSize,
                         float>::StateTransitionFunction
        state_transition_function;
    ExtendedKalmanFilter<kStateSize, kMeasurementSize,
                         float>::ObservationFunction observation_function;

    void SetUp() override {
        initial_state << 0, 0, 0, 0;
        initial_covariance << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
        state_transition_function =
            []([[maybe_unused]] const Eigen::Matrix<float, kStateSize, 1>&
                   state,
               float dt) {
                Eigen::Matrix<float, kStateSize, kStateSize> state_jacobian;
                state_jacobian << 1, 0, dt, 0, 0, 1, 0, dt, 0, 0, 1, 0, 0, 0, 0,
                    1;
                return state_jacobian;
            };
        process_noise_function = []([[maybe_unused]] float dt) {
            Eigen::Matrix<float, kStateSize, kStateSize> process_noise;
            process_noise << 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0,
                0.1;
            return process_noise;
        };
        observation_noise << 0.1, 0, 0, 0.1;
        observation_function =
            [](const Eigen::Matrix<float, kStateSize, 1>& state) {
                Eigen::Matrix<float, kMeasurementSize, 1> measurement;
                Eigen::Matrix<float, kMeasurementSize, kStateSize>
                    measurement_jacobian;
                measurement << state[0], state[1];
                measurement_jacobian << 1, 0, 0, 0, 0, 1, 0, 0;
                return std::make_pair(measurement, measurement_jacobian);
            };
    }
};

TEST_F(ExtendedKalmanFilterTest, PredictionStep) {
    ExtendedKalmanFilter<kStateSize, kMeasurementSize, float> filter(
        initial_state, initial_covariance, observation_noise);

    Eigen::Vector2f input;
    input << 0.5, 0.5;

    float dt = 1.0f;
    filter.predict(state_transition_function, process_noise_function, dt);
    filter.update(input, observation_function);

    Eigen::Matrix<float, kStateSize, 1> expected_state;
    Eigen::Matrix<float, kStateSize, kStateSize> expected_covariance;

    /**
     * The result is from filterpy. Source code:
     *
     * ```python
     * from filterpy.kalman import KalmanFilter
     * import numpy as np
     *
     * def HJacobian(x):
     *     return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
     *
     * def hx(x):
     *     return np.array([x[0], x[1]])
     *
     * ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
     * ekf.x = np.array([0, 0, 0, 0])
     * kf.P = np.array([[1, 0, 0, 0],
     *                  [0, 1, 0, 0],
     *                  [0, 0, 1, 0],
     *                  [0, 0, 0, 1]])
     * dt = 1.
     * ekf.F = np.array([[1, 0, dt, 0],
     *                   [0, 1, 0, dt],
     *                   [0, 0, 1, 0],
     *                   [0, 0, 0, 1]])
     * ekf.Q = np.array([[0.1, 0, 0, 0],
     *                   [0, 0.1, 0, 0],
     *                   [0, 0, 0.1, 0],
     *                   [0, 0, 0, 0.1]])
     * ekf.H = np.array([[1, 0, 0, 0],
     *                   [0, 1, 0, 0]])
     * ekf.R = np.array([[0.1, 0],
     *                   [0, 0.1]])
     *
     * z = np.array([0.5, 0.5])
     * ekf.predict()
     * ekf.update(z, HJacobian, hx)
     *
     * ekf.x, ekf.P
     */

    // clang-format off
    expected_state << 0.47727273, 0.47727273, 0.22727273, 0.22727273;
    expected_covariance << 0.09545455, 0.,         0.04545455,        0.,
                           0.,         0.09545455, 0.,                0.04545455,
                           0.04545455, 0.,         0.64545455,        0.,
                           0.,         0.04545455, 0.,                0.64545455;
    // clang-format on
    EXPECT_TRUE(filter.state().isApprox(expected_state));
    EXPECT_TRUE(filter.covariance().isApprox(expected_covariance));
}