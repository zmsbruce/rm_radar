#include "tracker.h"

#include "linear_assignment.h"
#include "nn_matching.h"

using namespace std;

namespace radar {

using namespace track;

Tracker::Tracker(float max_cosine_distance, int nn_budget,
                 float max_iou_distance, int max_age, int n_init) {
    this->metric =
        new NearNeighborDisMetric(NearNeighborDisMetric::METRIC_TYPE::cosine,
                                  max_cosine_distance, nn_budget);
    this->max_iou_distance = max_iou_distance;
    this->max_age = max_age;
    this->n_init = n_init;

    this->kf = new KalmanFilter();
    this->tracks_.clear();
    this->_next_idx = 1;
}

void Tracker::predict() noexcept {
    for (Track &track : tracks_) {
        track.predit(kf);
    }
}

void Tracker::update(const std::vector<Robot> &robots) noexcept {
    TRACHER_MATCHD res;
    match(robots, res);

    vector<MATCH_DATA> &matches = res.matches;
    for (MATCH_DATA &data : matches) {
        int track_idx = data.first;
        int detection_idx = data.second;
        tracks_[track_idx].update(this->kf, robots[detection_idx]);
    }
    vector<int> &unmatched_tracks = res.unmatched_tracks;
    for (int &track_idx : unmatched_tracks) {
        this->tracks_[track_idx].mark_missed();
    }
    vector<int> &unmatched_detections = res.unmatched_detections;
    for (int &detection_idx : unmatched_detections) {
        this->initiate_track(robots[detection_idx]);
    }
    vector<Track>::iterator it;
    for (it = tracks_.begin(); it != tracks_.end();) {
        if ((*it).is_deleted())
            it = tracks_.erase(it);
        else
            ++it;
    }

    vector<int> active_targets;
    vector<TRACKER_DATA> tid_features;
    for (Track &track : tracks_) {
        if (track.is_confirmed() == false) continue;
        active_targets.push_back(track.track_id);
        tid_features.push_back(std::make_pair(track.track_id, track.features));
        FEATURESS t = FEATURESS(0, k_feature_dim);
        track.features = t;
    }
    this->metric->partial_fit(tid_features, active_targets);
}

void Tracker::match(const std::vector<Robot> &robots,
                    track::TRACHER_MATCHD &res) {
    vector<int> confirmed_tracks;
    vector<int> unconfirmed_tracks;
    int idx = 0;
    for (Track &t : tracks_) {
        if (t.is_confirmed())
            confirmed_tracks.push_back(idx);
        else
            unconfirmed_tracks.push_back(idx);
        idx++;
    }

    TRACHER_MATCHD matcha = linear_assignment::getInstance()->matching_cascade(
        this, &Tracker::gated_matric, this->metric->mating_threshold,
        this->max_age, this->tracks_, robots, confirmed_tracks);
    vector<int> iou_track_candidates;
    iou_track_candidates.assign(unconfirmed_tracks.begin(),
                                unconfirmed_tracks.end());
    vector<int>::iterator it;
    for (it = matcha.unmatched_tracks.begin();
         it != matcha.unmatched_tracks.end();) {
        int idx = *it;
        if (tracks_[idx].time_since_update == 1) {
            iou_track_candidates.push_back(idx);
            it = matcha.unmatched_tracks.erase(it);
            continue;
        }
        ++it;
    }
    TRACHER_MATCHD matchb = linear_assignment::getInstance()->min_cost_matching(
        this, &Tracker::iou_cost, this->max_iou_distance, this->tracks_, robots,
        iou_track_candidates, matcha.unmatched_detections);

    res.matches.assign(matcha.matches.begin(), matcha.matches.end());
    res.matches.insert(res.matches.end(), matchb.matches.begin(),
                       matchb.matches.end());

    res.unmatched_tracks.assign(matcha.unmatched_tracks.begin(),
                                matcha.unmatched_tracks.end());
    res.unmatched_tracks.insert(res.unmatched_tracks.end(),
                                matchb.unmatched_tracks.begin(),
                                matchb.unmatched_tracks.end());
    res.unmatched_detections.assign(matchb.unmatched_detections.begin(),
                                    matchb.unmatched_detections.end());
}

void Tracker::initiate_track(const Robot &robot) {
    KAL_DATA data = kf->initiate(xyah(robot));
    KAL_MEAN mean = data.first;
    KAL_COVA covariance = data.second;

    this->tracks_.push_back(Track(mean, covariance, this->_next_idx,
                                  this->n_init, this->max_age, feature(robot)));
    _next_idx += 1;
}

DYNAMICM Tracker::gated_matric(std::vector<Track> &tracks,
                               const std::vector<Robot> &robots,
                               const std::vector<int> &track_indices,
                               const std::vector<int> &detection_indices) {
    FEATURESS features(detection_indices.size(), k_feature_dim);
    int pos = 0;
    for (int i : detection_indices) {
        features.row(pos++) = feature(robots[i]);
    }
    vector<int> targets;
    for (int i : track_indices) {
        targets.push_back(tracks[i].track_id);
    }
    DYNAMICM cost_matrix = this->metric->distance(features, targets);
    DYNAMICM res = linear_assignment::getInstance()->gate_cost_matrix(
        this->kf, cost_matrix, tracks, robots, track_indices,
        detection_indices);
    return res;
}

DYNAMICM
Tracker::iou_cost(std::vector<Track> &tracks, const std::vector<Robot> &robots,
                  const std::vector<int> &track_indices,
                  const std::vector<int> &detection_indices) {
    int rows = track_indices.size();
    int cols = detection_indices.size();
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(rows, cols);
    for (int i = 0; i < rows; i++) {
        int track_idx = track_indices[i];
        if (tracks[track_idx].time_since_update > 1) {
            cost_matrix.row(i) = Eigen::RowVectorXf::Constant(cols, INFTY_COST);
            continue;
        }
        DETECTBOX bbox = tracks[track_idx].to_tlwh();
        int csize = detection_indices.size();
        DETECTBOXSS candidates(csize, 4);
        for (int k = 0; k < csize; k++)
            candidates.row(k) = tlwh(robots[detection_indices[k]]);
        Eigen::RowVectorXf rowV =
            (1. - iou(bbox, candidates).array()).matrix().transpose();
        cost_matrix.row(i) = rowV;
    }
    return cost_matrix;
}

Eigen::VectorXf Tracker::iou(DETECTBOX &bbox, DETECTBOXSS &candidates) {
    float bbox_tl_1 = bbox[0];
    float bbox_tl_2 = bbox[1];
    float bbox_br_1 = bbox[0] + bbox[2];
    float bbox_br_2 = bbox[1] + bbox[3];
    float area_bbox = bbox[2] * bbox[3];

    Eigen::Matrix<float, -1, 2> candidates_tl;
    Eigen::Matrix<float, -1, 2> candidates_br;
    candidates_tl = candidates.leftCols(2);
    candidates_br = candidates.rightCols(2) + candidates_tl;

    int size = int(candidates.rows());
    Eigen::VectorXf res(size);
    for (int i = 0; i < size; i++) {
        float tl_1 = std::max(bbox_tl_1, candidates_tl(i, 0));
        float tl_2 = std::max(bbox_tl_2, candidates_tl(i, 1));
        float br_1 = std::min(bbox_br_1, candidates_br(i, 0));
        float br_2 = std::min(bbox_br_2, candidates_br(i, 1));

        float w = br_1 - tl_1;
        w = (w < 0 ? 0 : w);
        float h = br_2 - tl_2;
        h = (h < 0 ? 0 : h);
        float area_intersection = w * h;
        float area_candidates = candidates(i, 2) * candidates(i, 3);
        res[i] = area_intersection /
                 (area_bbox + area_candidates - area_intersection);
    }
    return res;
}

}  // namespace radar