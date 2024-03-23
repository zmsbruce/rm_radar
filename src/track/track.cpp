#include "track.h"

#include "robot/robot.h"

namespace radar::track {

DETECTBOX xyah(float x, float y, float width, float height) {
    DETECTBOX ret;
    ret << x, y, width, height;
    ret(0, 0) += ret(0, 2) * 0.5f;
    ret(0, 1) += ret(0, 3) * 0.5f;
    ret(0, 2) /= ret(0, 3);
    return ret;
}

DETECTBOX xyah(const cv::Rect& rect) {
    return xyah(rect.x, rect.y, rect.width, rect.height);
}

DETECTBOX xyah(const Detection& detection) {
    return xyah(detection.x, detection.y, detection.width, detection.height);
}

DETECTBOX xyah(const Robot& robot) {
    if (!robot.isDetected()) {
        throw std::logic_error("robot is not detected.");
    }
    return xyah(robot.rect().value());
}

DETECTBOX tlwh(float x, float y, float width, float height) {
    DETECTBOX ret;
    ret << x, y, width, height;
    return ret;
}

DETECTBOX tlwh(const cv::Rect& rect) {
    return tlwh(rect.x, rect.y, rect.width, rect.height);
}

DETECTBOX tlwh(const Detection& detection) {
    return tlwh(detection.x, detection.y, detection.width, detection.height);
}

DETECTBOX tlwh(const Robot& robot) {
    if (!robot.isDetected()) {
        throw std::logic_error("robot is not detected.");
    }
    return tlwh(robot.rect().value());
}

FEATURE feature(const std::vector<Detection>& detections) {
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

FEATURE feature(const Robot& robot) {
    if (!robot.isDetected()) {
        throw std::logic_error("robot is not detected.");
    }
    return feature(robot.armors().value());
}

}  // namespace radar::track

namespace radar {

using namespace radar::track;

float label_trust;
float label_score_min;
float label_trust_no_detect;

Track::Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id, int n_init,
             int max_age, const FEATURE& feature) {
    this->mean = mean;
    this->covariance = covariance;
    this->track_id = track_id;
    this->hits = 1;
    this->age = 1;
    this->time_since_update = 0;
    this->state = TrackState::Tentative;
    features = FEATURESS(1, k_feature_dim);
    features.row(0) = feature;  // features.rows() must = 0;

    this->_n_init = n_init;
    this->_max_age = max_age;
}

/**
 * @brief Propagate the state distribution to the current time step using a
 * Kalman filter prediction step.
 *
 * @param kf The Kalman filter
 */
void Track::predit(KalmanFilter* kf) {
    kf->predict(this->mean, this->covariance);
    this->age += 1;
    this->time_since_update += 1;
}

void Track::update(KalmanFilter* const kf, const Robot& robot) {
    KAL_DATA pa = kf->update(this->mean, this->covariance, xyah(robot));
    this->mean = pa.first;
    this->covariance = pa.second;

    featuresAppendOne(feature(robot));

    this->hits += 1;
    this->time_since_update = 0;
    if (this->state == TrackState::Tentative && this->hits >= this->_n_init) {
        this->state = TrackState::Confirmed;
    }
}

void Track::mark_missed() {
    if (this->state == TrackState::Tentative) {
        this->state = TrackState::Deleted;
    } else if (this->time_since_update > this->_max_age) {
        this->state = TrackState::Deleted;
    }
}

bool Track::is_confirmed() const {
    return this->state == TrackState::Confirmed;
}

bool Track::is_deleted() const { return this->state == TrackState::Deleted; }

bool Track::is_tentative() const {
    return this->state == TrackState::Tentative;
}

DETECTBOX Track::to_tlwh() const {
    DETECTBOX ret = mean.leftCols(4);
    ret(2) *= ret(3);
    ret.leftCols(2) -= (ret.rightCols(2) / 2);
    return ret;
}

void Track::featuresAppendOne(const FEATURE& f) {
    int size = this->features.rows();
    FEATURESS newfeatures = FEATURESS(size + 1, k_feature_dim);
    newfeatures.block(0, 0, size, k_feature_dim) = this->features;
    newfeatures.row(size) = f;
    features = newfeatures;
}

}  // namespace radar