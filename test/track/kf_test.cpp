#include <gtest/gtest.h>

#include "track/kalman_filter.h"

using namespace radar::track;

class KalmanFilterTest : public ::testing::Test {
   protected:
    static constexpr int kStateSize = 4;
    static constexpr int kMeasurementSize = 2;

    Eigen::Matrix<float, kStateSize, 1> initial_state;
    Eigen::Matrix<float, kStateSize, kStateSize> initial_covariance;
    Eigen::Matrix<float, kStateSize, kStateSize> transition_matrix;
    Eigen::Matrix<float, kStateSize, kStateSize> process_noise;
    Eigen::Matrix<float, kMeasurementSize, kStateSize> observation_matrix;
    Eigen::Matrix<float, kMeasurementSize, kMeasurementSize> observation_noise;

    void SetUp() override {
        initial_state << 0, 0, 0, 0;
        initial_covariance << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
        transition_matrix << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1;
        process_noise << 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.1;
        observation_matrix << 1, 0, 0, 0, 0, 1, 0, 0;
        observation_noise << 0.1, 0, 0, 0.1;
    }
};

TEST_F(KalmanFilterTest, PredictionStep) {
    KalmanFilter<kStateSize, kMeasurementSize> filter(
        initial_state, initial_covariance, transition_matrix, process_noise,
        observation_matrix, observation_noise);

    Eigen::Vector2f input;
    input << 0.5, 0.5;

    filter.predict();
    filter.update(input);

    Eigen::Matrix<float, kStateSize, 1> expected_state;
    Eigen::Matrix<float, kStateSize, kStateSize> expected_covariance;

    /**
     * The result is from filterpy. Source code:
     *
     * ```python
     * from filterpy.kalman import KalmanFilter
     * import numpy as np
     *
     * kf = KalmanFilter(dim_x=4, dim_z=2)
     * kf.x = np.array([0, 0, 0, 0])
     * kf.P = np.array([[1, 0, 0, 0],
     *                  [0, 1, 0, 0],
     *                  [0, 0, 1, 0],
     *                  [0, 0, 0, 1]])
     * kf.F = np.array([[1, 0, 1, 0],
     *                  [0, 1, 0, 1],
     *                  [0, 0, 1, 0],
     *                  [0, 0, 0, 1]])
     * kf.Q = np.array([[0.1, 0, 0, 0],
     *                  [0, 0.1, 0, 0],
     *                  [0, 0, 0.1, 0],
     *                  [0, 0, 0, 0.1]])
     * kf.H = np.array([[1, 0, 0, 0],
     *                  [0, 1, 0, 0]])
     * kf.R = np.array([[0.1, 0],
     *                  [0, 0.1]])
     * kf.predict()
     * z = np.array([0.5, 0.5])
     * kf.update(z)
     *
     * kf.x, kf.P
     * ```
     */

    // clang-format off
    expected_state << 0.47727273, 0.47727273, 0.22727273, 0.22727273;
    expected_covariance << 0.09545455, 0.,         0.04545455,        0.,
                           0.,         0.09545455, 0.,                0.04545455,
                           0.04545455, 0.,         0.64545455,        0.,
                           0.,         0.04545455, 0.,                0.64545455;
    // clang-format on

    EXPECT_TRUE(filter.state().isApprox(expected_state, 1e-5));
    EXPECT_TRUE(filter.covariance().isApprox(expected_covariance, 1e-5));
}
