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
    /**
     * @brief Construct the Tracker class
     *
     * @param observation_noise The observation noise(m).
     * @param class_num The number of classes.
     * @param init_thresh Times needed to convert a track from tentative to
     * confirmed.
     * @param miss_thresh Times needed to mark a confirmed track to deleted.
     * @param max_acceleration Max acceleration(m/s^2) needed for the Singer-EKF
     * model.
     * @param acceleration_correlation_time Acceleration correlation time
     * constant(tau) of the Singer-EKF model.
     * @param distance_weight The weight of distance which is needed in min-cost
     * matching.
     * @param feature_weight The weight of feature which is needed in min-cost
     * matching.
     * @param max_iter The maximum iteration time of the auction algorithm.
     * @param distance_thresh The distance threshold(m) for scoring.
     */
    Tracker(const cv::Point3f& observation_noise, int class_num,
            int init_thresh = 4, int miss_thresh = 10,
            float max_acceleration = 2.0f,
            float acceleration_correlation_time = 1.0f,
            float distance_weight = 0.40f, float feature_weight = 0.60f,
            int max_iter = 100, float distance_thresh = 0.8f);

    /**
     * @brief Update all tracks based on a new set of robot observations.
     *
     * @param robots The new robot observations.
     * @param timestamp The timestamp of the observations.
     */
    void update(std::vector<Robot>& robots,
                const std::chrono::high_resolution_clock::time_point&
                    timestamp) noexcept;

   private:
    /**
     * @brief Calculate the cost associated with matching a track to a robot
     * observation.
     *
     * @param track The track for which to calculate the cost.
     * @param robot The robot observation to be matched to the track.
     * @return The calculated cost.
     */
    float calculateCost(const Track& track, const Robot& robot) const noexcept;

    /**
     * @brief Calculate the weighted Euclidean distance between two points in 3D
     * space.
     *
     * @param p1 The first point.
     * @param p2 The second point.
     * @return The Euclidean distance.
     */
    static float calculateDistance(const cv::Point3f& p1,
                                   const cv::Point3f& p2) noexcept;

    /**
     * @brief The number of robot classes.
     *
     * This is the total number of different robot classes that the tracker
     * can recognize.
     */
    const int class_num_;

    /**
     * @brief The number of updates needed to confirm a tentative track.
     *
     * A track remains tentative until it has been updated `init_thresh_` times
     * with new observations. After that, it is marked as confirmed.
     */
    const int init_thresh_;

    /**
     * @brief The number of missed updates before deleting a confirmed track.
     *
     * A confirmed track will be deleted if it misses `miss_thresh_` consecutive
     * updates (i.e., it is not matched with any observations for `miss_thresh_`
     * updates).
     */
    const int miss_thresh_;

    /**
     * @brief The maximum acceleration (in m/s^2) used in the Singer-EKF model.
     *
     * This parameter defines the maximum expected acceleration of the robot,
     * which is used in the Kalman filter to predict the robot's future
     * position.
     */
    const float max_acc_;

    /**
     * @brief The time constant (tau) for the Singer-EKF acceleration model.
     *
     * This parameter is used to model the correlation of acceleration over time
     * in the Singer-EKF model. A smaller value represents a faster change in
     * acceleration.
     */
    const float tau_;

    /**
     * @brief The weight of the distance component in the min-cost matching
     * algorithm.
     *
     * This parameter controls the importance of the distance between the track
     * and the observation when calculating the cost of a match.
     */
    float distance_weight_;

    /**
     * @brief The weight of the feature component in the min-cost matching
     * algorithm.
     *
     * This parameter controls the importance of the feature similarity between
     * the track and the observation when calculating the cost of a match.
     */
    float feature_weight_;

    /**
     * @brief The observation noise in the measurement (in meters).
     *
     * This parameter represents the noise in the measurements (e.g., position)
     * of the robots. It is used in the Kalman filter to account for uncertainty
     * in the observations.
     */
    const cv::Point3f measurement_noise_;

    /**
     * @brief The list of active tracks.
     *
     * This vector stores all the active tracks that are being managed by the
     * tracker. Each track represents a robot and includes its state, position,
     * and other information.
     */
    std::vector<Track> tracks_;

    /**
     * @brief The maximum number of iterations for the auction algorithm.
     *
     * The auction algorithm is used to solve the assignment problem when
     * matching tracks to observations. This parameter defines the maximum
     * number of iterations that the algorithm can run.
     */
    const int max_iter_;

    /**
     * @brief The distance threshold (in meters) for scoring matches.
     *
     * This threshold is used to filter out matches between tracks and
     * observations that are too far apart. If the distance between a track and
     * an observation exceeds this threshold, they will not be matched.
     */
    const float distance_thresh_;

    /**
     * @brief The latest track ID assigned by the tracker.
     *
     * This variable keeps track of the most recent ID assigned to a new track.
     * It is incremented each time a new track is created.
     */
    int latest_id_ = 0;
};

}  // namespace radar