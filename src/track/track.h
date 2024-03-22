#pragma once

#include <numeric>

#include "data_type.h"
#include "detection.h"
#include "kalman_filter.h"
#include "robot.h"

namespace radar::track {

inline DETECTBOX xyah(float x, float y, float width, float height) {
    DETECTBOX ret;
    ret << x, y, width, height;
    ret(0, 0) += ret(0, 2) * 0.5f;
    ret(0, 1) += ret(0, 3) * 0.5f;
    ret(0, 2) /= ret(0, 3);
    return ret;
}

inline DETECTBOX xyah(const cv::Rect& rect) {
    return xyah(rect.x, rect.y, rect.width, rect.height);
}

inline DETECTBOX xyah(const Detection& detection) {
    return xyah(detection.x, detection.y, detection.width, detection.height);
}

inline DETECTBOX xyah(const Robot& robot) {
    if (!robot.isDetected()) {
        throw std::logic_error("robot is not detected.");
    }
    return xyah(robot.rect().value());
}

inline DETECTBOX tlwh(float x, float y, float width, float height) {
    DETECTBOX ret;
    ret << x, y, width, height;
    return ret;
}

inline DETECTBOX tlwh(const cv::Rect& rect) {
    return tlwh(rect.x, rect.y, rect.width, rect.height);
}

inline DETECTBOX tlwh(const Detection& detection) {
    return tlwh(detection.x, detection.y, detection.width, detection.height);
}

inline DETECTBOX tlwh(const Robot& robot) {
    if (!robot.isDetected()) {
        throw std::logic_error("robot is not detected.");
    }
    return tlwh(robot.rect().value());
}

inline FEATURE feature(const std::vector<Detection>& detections) {
    FEATURE ret;
    ret.setZero();
    float total_confidence =
        std::accumulate(detections.begin(), detections.end(), 0.,
                        [](float partial, const Detection& detection) {
                            return partial + detection.confidence;
                        });
    for (auto&& detection : detections) {
        ret[static_cast<int>(detection.label)] +=
            detection.confidence / total_confidence;
    }
    return ret;
}

inline FEATURE feature(const Robot& robot) {
    if (!robot.isDetected()) {
        throw std::logic_error("robot is not detected.");
    }
    return feature(robot.armors().value());
}

/**
 * @brief A single target track with state space `(x, y, a, h)` and associated
 * velocities, where `(x, y)` is the center of the bounding box, `a` is the
 * aspect ratio and `h` is the height.
 *
 */
class Track {
   public:
    Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id, int n_init,
          int max_age, const FEATURE& feature);
    void predit(KalmanFilter* kf);
    void update(KalmanFilter* const kf, const Robot& robot);
    void mark_missed();
    bool is_confirmed();
    bool is_deleted();
    bool is_tentative();
    DETECTBOX to_tlwh();
    int time_since_update;
    int track_id;
    FEATURESS features;
    KAL_MEAN mean;
    KAL_COVA covariance;

    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;

   private:
    void featuresAppendOne(const FEATURE& f);
};

}  // namespace radar::track