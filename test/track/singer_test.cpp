#include "track/singer.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>

using namespace radar::track;

class SingerTest : public ::testing::Test {
   protected:
    Eigen::Matrix<float, kStateSize, 1> initial_state;
    Eigen::Matrix<float, kStateSize, kStateSize> initial_covariance;
    Eigen::Matrix<float, kMeasurementSize, kMeasurementSize> observation_noise;
    static constexpr float max_a = 2;  // m/s^2
    static constexpr float tau = 1;    // s

    std::unique_ptr<SingerEKF> filter = nullptr;

    void SetUp() override {
        initial_state.setZero();
        initial_covariance = decltype(initial_covariance)::Identity() * 0.5f;
        observation_noise << 0.2f, 0, 0, 0, 0.2f, 0, 0, 0, 0.2f;
        filter = std::make_unique<SingerEKF>(initial_state, initial_covariance,
                                             max_a, tau, observation_noise);
    }
};

TEST_F(SingerTest, TestStable) {
    Eigen::Matrix<float, kMeasurementSize, 1> measurement;
    measurement << 10.0f, 20.0f, 30.0f;  // m
    const float dt = 1;                  // s

    for (int i = 0; i < 10; ++i) {
        filter->predict(dt);
        filter->update(measurement);
    }

    auto state = filter->state();
    Eigen::Matrix<float, kMeasurementSize, 1> state_xyz;
    state_xyz << state(0), state(3), state(6);
    EXPECT_TRUE(state_xyz.isApprox(measurement, 1e-1));
}

TEST_F(SingerTest, TestUniformMotion) {
    constexpr int times = 10;
    constexpr float x_init = 10.0f, y_init = 20.0f, z_init = 30.0f;
    constexpr float x_vel = 2.0f, y_vel = 4.0f, z_vel = 6.0f;
    constexpr float dt = 1.0f;

    for (int i = 0; i < times; ++i) {
        Eigen::Matrix<float, kMeasurementSize, 1> measurement;
        measurement << x_init + i * x_vel, y_init + i * y_vel,
            z_init + i * z_vel;

        filter->predict(dt);
        filter->update(measurement);
    }

    auto state = filter->state();
    Eigen::Matrix<float, kMeasurementSize, 1> state_pos, state_vel, state_acc;
    state_pos << state(0), state(3), state(6);
    state_vel << state(1), state(4), state(7);
    state_acc << state(2), state(5), state(8);

    Eigen::Matrix<float, kMeasurementSize, 1> state_pos_gt, state_vel_gt,
        state_acc_gt;
    state_pos_gt << x_init + x_vel * (times - 1), y_init + y_vel * (times - 1),
        z_init + z_vel * (times - 1);
    state_vel_gt << x_vel, y_vel, z_vel;
    state_acc_gt << 0.0f, 0.0f, 0.0f;

    EXPECT_TRUE(state_pos.isApprox(state_pos_gt, 1e-1));
    EXPECT_TRUE(state_vel.isApprox(state_vel_gt, 1e-1));
    for (int i = 0; i < state_acc.size(); ++i) {
        EXPECT_TRUE(std::abs(state_acc(i) - state_acc_gt(i)) < 1e-1);
    }
}

TEST_F(SingerTest, TestAcceleratedMotion) {
    constexpr int times = 10;
    constexpr float x_pos_init = 10.0f, y_pos_init = 20.0f, z_pos_init = 30.0f;
    constexpr float x_vel = 2.0f, y_vel = 4.0f, z_vel = 6.0f;
    constexpr float x_acc = 0.0f, y_acc = 0.5f, z_acc = 1.0f;
    constexpr float dt = 1.0f;

    for (int i = 0; i < times; ++i) {
        Eigen::Matrix<float, kMeasurementSize, 1> measurement;
        float x_pos = x_pos_init + x_vel * i + 0.5 * x_acc * i * i;
        float y_pos = y_pos_init + y_vel * i + 0.5 * y_acc * i * i;
        float z_pos = z_pos_init + z_vel * i + 0.5 * z_acc * i * i;

        measurement << x_pos, y_pos, z_pos;

        filter->predict(dt);
        filter->update(measurement);
    }

    auto state = filter->state();
    Eigen::Matrix<float, kMeasurementSize, 1> state_pos, state_vel, state_acc;
    state_pos << state(0), state(3), state(6);
    state_vel << state(1), state(4), state(7);
    state_acc << state(2), state(5), state(8);

    Eigen::Matrix<float, kMeasurementSize, 1> state_pos_gt, state_vel_gt,
        state_acc_gt;
    float x_pos = x_pos_init + x_vel * (times - 1) +
                  0.5 * x_acc * (times - 1) * (times - 1);
    float y_pos = y_pos_init + y_vel * (times - 1) +
                  0.5 * y_acc * (times - 1) * (times - 1);
    float z_pos = z_pos_init + z_vel * (times - 1) +
                  0.5 * z_acc * (times - 1) * (times - 1);
    state_pos_gt << x_pos, y_pos, z_pos;

    float x_vel_gt = x_vel + x_acc * (times - 1);
    float y_vel_gt = y_vel + y_acc * (times - 1);
    float z_vel_gt = z_vel + z_acc * (times - 1);
    state_vel_gt << x_vel_gt, y_vel_gt, z_vel_gt;

    EXPECT_TRUE(state_pos.isApprox(state_pos_gt, 1e-1));
    EXPECT_TRUE(state_vel.isApprox(state_vel_gt, 1e-1));
}