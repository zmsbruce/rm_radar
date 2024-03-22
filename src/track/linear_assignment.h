#pragma once

#include "data_type.h"
#include "tracker.h"

#define INFTY_COST 1e5

namespace radar::track {

// for matching;
class linear_assignment {
    linear_assignment();

    linear_assignment(const linear_assignment&);

    linear_assignment& operator=(const linear_assignment&);

    static linear_assignment* instance;

   public:
    static linear_assignment* getInstance();

    TRACHER_MATCHD matching_cascade(
        Tracker* distance_metric,
        Tracker::GATED_METRIC_FUNC distance_metric_func, float max_distance,
        int cascade_depth, std::vector<Track>& tracks,
        const std::vector<Detection>& detections,
        std::vector<int>& track_indices,
        std::vector<int> detection_indices = std::vector<int>());

    TRACHER_MATCHD min_cost_matching(
        Tracker* distance_metric,
        Tracker::GATED_METRIC_FUNC distance_metric_func, float max_distance,
        std::vector<Track>& tracks, const std::vector<Detection>& detections,
        std::vector<int>& track_indices, std::vector<int>& detection_indices);

    DYNAMICM gate_cost_matrix(KalmanFilter* kf, DYNAMICM& cost_matrix,
                              std::vector<Track>& tracks,
                              const std::vector<Detection>& detections,
                              const std::vector<int>& track_indices,
                              const std::vector<int>& detection_indices,
                              float gated_cost = INFTY_COST,
                              bool only_position = false);
};

}  // namespace radar::track
