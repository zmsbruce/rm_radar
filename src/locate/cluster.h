/**
 * @file cluster.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief A file that implements the DBSCAN clustering algorithm.
 * @date 2024-04-06
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <cmath>
#include <concepts>
#include <span>
#include <stack>
#include <vector>

const int kNoise = 0;
const int kUnclassified = -1;

/**
 * @brief Checks if a type T can represent a 3D point with float-convertible
 * members x, y, and z.
 *
 * @tparam T Template type to check.
 */
template <typename T>
concept Point3d = requires(T point) {
    { point.x } -> std::convertible_to<float>;
    { point.y } -> std::convertible_to<float>;
    { point.z } -> std::convertible_to<float>;
};

/**
 * @brief Checks if a type T can represent a 2D point with float-convertible
 * members x and y.
 *
 * @tparam T Template type to check.
 */
template <typename T>
concept Point2d = requires(T point) {
    { point.x } -> std::convertible_to<float>;
    { point.y } -> std::convertible_to<float>;
};

/**
 * @brief Point concept requiring Point to be either Point3d or Point2d.
 *
 * @tparam T Template type to ensure it matches either Point3d or Point2d
 * concept.
 */
template <typename T>
concept Point = Point3d<T> || Point2d<T>;

/**
 * @brief A class that implements the DBSCAN clustering algorithm.
 */
class DBSCAN {
   public:
    /**
     * @brief Constructs a new DBSCAN object with the specified minimum number
     * of points and epsilon value.
     *
     * @param min_point_size Minimum number of points to form a dense region.
     * @param epsilon Maximum distance between two points for one to be
     * considered as in the neighborhood of the other.
     */
    DBSCAN(size_t min_point_size, float epsilon)
        : min_point_size_{min_point_size}, epsilon_{epsilon} {}

    /**
     * @brief Performs the clustering algorithm on a span of points and returns
     * the cluster indices.
     *
     * @tparam T The type of the points which must satisfy the Point concept.
     * @param points A span of points to cluster.
     * @return A `std::vector<int>` containing the cluster index for each point.
     */
    template <Point T>
    std::vector<int> cluster(const std::span<T>& points) {
        std::vector<int> cluster_indices(points.size(), kUnclassified);

        int cluster_id = 1;

        for (size_t i = 0; i < points.size(); ++i) {
            if (cluster_indices[i] != kUnclassified) {
                continue;
            }
            auto neighbor_indices = calculateNeighbors(i, points);
            if (neighbor_indices.size() < min_point_size_) {
                cluster_indices[i] = kNoise;
                continue;
            }

            // Mark the point as part of the cluster
            cluster_indices[i] = cluster_id;

            // Expand the cluster from the neighbors
            std::stack<size_t> stack;
            for (size_t neighbor_index : neighbor_indices) {
                if (cluster_indices[neighbor_index] == kUnclassified) {
                    stack.push(neighbor_index);
                    cluster_indices[neighbor_index] = cluster_id;
                }
            }

            // Iteratively expand the cluster
            while (!stack.empty()) {
                size_t current_index = stack.top();
                stack.pop();

                auto current_neighbor_indices =
                    calculateNeighbors(current_index, points);
                if (current_neighbor_indices.size() >= min_point_size_) {
                    for (size_t neighbor_index : current_neighbor_indices) {
                        if (cluster_indices[neighbor_index] == kUnclassified) {
                            stack.push(neighbor_index);
                            cluster_indices[neighbor_index] = cluster_id;
                        }
                    }
                }
            }

            // Move to the next cluster
            cluster_id++;
        }

        return cluster_indices;
    }

   private:
    DBSCAN() = delete;

    /**
     * @brief Calculates the neighboring indices of a point at a given index.
     *
     * @tparam T The type of the points which must satisfy the Point concept.
     * @param index The index of the point to calculate neighbors for.
     * @param points A span of points among which to calculate neighbors.
     * @return A std::vector<size_t> containing the indices of neighboring
     * points.
     */
    template <Point T>
    std::vector<size_t> calculateNeighbors(size_t index,
                                           const std::span<T>& points) {
        std::vector<size_t> indices;
        for (size_t i = 0; i < points.size(); ++i) {
            if (i != index &&
                calculateDistance(points[index], points[i]) <= epsilon_) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    /**
     * @brief Calculates the Euclidean distance between two points.
     *
     * @tparam T The type of the points which must satisfy the Point concept.
     * @param p1 The first point.
     * @param p2 The second point.
     * @return The Euclidean distance between p1 and p2.
     */
    template <Point T>
    static inline float calculateDistance(const T& p1, const T& p2) {
        if constexpr (Point2d<T>) {
            return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                             (p1.y - p2.y) * (p1.y - p2.y));
        } else {
            return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) +
                             (p1.y - p2.y) * (p1.y - p2.y) +
                             (p1.z - p2.z) * (p1.z - p2.z));
        }
    }
    size_t min_point_size_;
    float epsilon_;
};