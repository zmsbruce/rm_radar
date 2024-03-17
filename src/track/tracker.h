#pragma once
#include <vector>

#include "detection.h"
#include "kalman_filter.h"
#include "track.h"

namespace radar {

class NearNeighborDisMetric;

class Tracker {
   public:
    NearNeighborDisMetric* metric;
    float max_iou_distance;
    int max_age;
    int n_init;
    float process_min_iou;
    KalmanFilter* kf;
    int _next_idx;

   public:
    std::vector<Track> tracks;

    Tracker(float max_cosine_distance, int nn_budget,
            float max_iou_distance = 0.7, int max_age = 30, int n_init = 3);

    void predict() noexcept;

    void update(const std::vector<Detection>& detections) noexcept;

    typedef DYNAMICM (Tracker::*GATED_METRIC_FUNC)(
        std::vector<Track>& tracks, const std::vector<Detection>& dets,
        const std::vector<int>& track_indices,
        const std::vector<int>& detection_indices);

   private:
    void _match(const std::vector<Detection>& detections, TRACHER_MATCHD& res);

    void _initiate_track(const Detection& detection);

   public:
    DYNAMICM gated_matric(std::vector<Track>& tracks,
                          const std::vector<Detection>& dets,
                          const std::vector<int>& track_indices,
                          const std::vector<int>& detection_indices);

    DYNAMICM iou_cost(std::vector<Track>& tracks,
                      const std::vector<Detection>& dets,
                      const std::vector<int>& track_indices,
                      const std::vector<int>& detection_indices);

    Eigen::VectorXf iou(DETECTBOX& bbox, DETECTBOXSS& candidates);
};

}  // namespace radar
