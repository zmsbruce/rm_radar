/**
 * @file tracker.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This is a header file containing the definition of the Tracker class,
 * which is responsible for managing and updating a set of tracks based on
 * observations of robots, and its functions.
 * @date 2024-04-10
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "robot/robot.h"
#include "track.h"

namespace radar {

class Tracker {
   public:
    Tracker(const cv::Point3f& observation_noise, int init_thresh = 4,
            int miss_thresh = 10, float max_acceleration = 2.0f,
            float acceleration_correlation_time = 1.0f,
            float distance_weight = 0.35f, float feature_weight = 0.65f,
            int max_iter = 100);

    void update(
        std::vector<Robot>& robots,
        const std::chrono::high_resolution_clock::time_point& timestamp);

   private:
    float calculateCost(const Track& track, const Robot& robot);

    static float calculateDistance(const cv::Point3f& p1,
                                   const cv::Point3f& p2);

    const int init_thresh_;
    const int miss_thresh_;
    const float max_acc_;
    const float tau_;
    float distance_weight_;
    float feature_weight_;
    const cv::Point3f measurement_noise_;
    std::vector<Track> tracks_;
    const int max_iter_;
    int latest_id_ = 0;
};

}  // namespace radar