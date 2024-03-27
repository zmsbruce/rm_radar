#pragma once

#include <vector>

#include "detect/detection.h"
#include "kalman_filter.h"
#include "nn_matching.h"
#include "track.h"

namespace radar {

class Robot;

class Tracker {
   public:
    Tracker(float max_cosine_distance, int nn_budget,
            float max_iou_distance = 0.7, int max_age = 30, int n_init = 3);

    void predict() noexcept;

    void update(const std::vector<Robot>& robots) noexcept;

    inline std::vector<Track> tracks() { return tracks_; }

    typedef track::DYNAMICM (Tracker::*GATED_METRIC_FUNC)(
        std::vector<Track>& tracks, const std::vector<Robot>& robots,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices);

   private:
    void match(const std::vector<Robot>& robots, track::TRACHER_MATCHD& res);

    void initiate_track(const Robot& robot);

    track::DYNAMICM gated_matric(std::vector<Track>& tracks,
                                 const std::vector<Robot>& robots,
                                 const std::vector<int>& track_indices,
                                 const std::vector<int>& detection_indices);

    track::DYNAMICM iou_cost(std::vector<Track>& tracks,
                             const std::vector<Robot>& robots,
                             const std::vector<int>& track_indices,
                             const std::vector<int>& detection_indices);

    Eigen::VectorXf iou(track::DETECTBOX& bbox, track::DETECTBOXSS& candidates);

    std::vector<Track> tracks_;
    track::NearNeighborDisMetric* metric;
    float max_iou_distance;
    int max_age;
    int n_init;
    float process_min_iou;
    track::KalmanFilter* kf;
    int _next_idx;
};

}  // namespace radar
