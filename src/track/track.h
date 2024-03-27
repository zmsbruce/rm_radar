#pragma once

#include <numeric>
#include <opencv2/opencv.hpp>

#include "data_type.h"
#include "detect/detection.h"
#include "kalman_filter.h"

namespace radar {

class Robot;

}  // namespace radar

namespace radar::track {

DETECTBOX xyah(float x, float y, float width, float height);
DETECTBOX xyah(const cv::Rect& rect);
DETECTBOX xyah(const Detection& detection);
DETECTBOX xyah(const Robot& robot);

DETECTBOX tlwh(float x, float y, float width, float height);
DETECTBOX tlwh(const cv::Rect& rect);
DETECTBOX tlwh(const Detection& detection);
DETECTBOX tlwh(const Robot& robot);

FEATURE feature(const std::vector<Detection>& detections);
FEATURE feature(const Robot& robot);

}  // namespace radar::track

namespace radar {

class Robot;

enum class TrackState { Tentative, Confirmed, Deleted };

/**
 * @brief A single target track with state space `(x, y, a, h)` and associated
 * velocities, where `(x, y)` is the center of the bounding box, `a` is the
 * aspect ratio and `h` is the height.
 *
 */
class Track {
   public:
    Track(track::KAL_MEAN& mean, track::KAL_COVA& covariance, int track_id,
          int n_init, int max_age, const track::FEATURE& feature);
    void predit(track::KalmanFilter* kf);
    void update(track::KalmanFilter* const kf, const Robot& robot);
    void mark_missed();
    bool is_confirmed() const;
    bool is_deleted() const;
    bool is_tentative() const;
    track::DETECTBOX to_tlwh() const;
    int time_since_update;
    int track_id;
    track::FEATURESS features;
    track::KAL_MEAN mean;
    track::KAL_COVA covariance;

    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;

   private:
    void featuresAppendOne(const track::FEATURE& f);
};

}  // namespace radar