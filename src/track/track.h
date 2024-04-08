#pragma once

#include <Eigen/Dense>

namespace radar::track {

enum class TrackState { Tentative, Confirmed, Deleted };

class Track {
   public:
    Track(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance,
          const Eigen::VectorXf& feature, int track_id, int n_init,
          int max_age);

    inline bool isConfirmed() const noexcept {
        return state_ == TrackState::Confirmed;
    }

    inline bool isTentative() const noexcept {
        return state_ == TrackState::Tentative;
    }

    inline bool isDeleted() const noexcept {
        return state_ == TrackState::Deleted;
    }

    inline void setTrack(TrackState state) noexcept { state_ = state; }

    inline void updateFeature(const Eigen::VectorXf& feature) {
        features_.conservativeResize(features_.rows(), features_.cols() + 1);
        features_.col(features_.cols() - 1) = feature;
    }

   private:
    Track() = delete;
    Eigen::VectorXf mean_;
    Eigen::MatrixXf covariance_;
    Eigen::MatrixXf features_;
    int track_id_;
    int n_init_;
    int max_age_;
    int hits_;
    int age_;
    int time_since_update_;
    TrackState state_;
};

Track::Track(const Eigen::VectorXf& mean, const Eigen::MatrixXf& covariance,
             const Eigen::VectorXf& feature, int track_id, int n_init,
             int max_age)
    : mean_{mean},
      covariance_{covariance},
      features_{feature.size(), 1},
      track_id_{track_id},
      n_init_{n_init},
      max_age_{max_age},
      hits_{1},
      age_{1},
      time_since_update_{0},
      state_{TrackState::Tentative} {
    features_.col(0) = feature;
}

}  // namespace radar::track