/**
 * @file locator.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief The file defines the `Locator` class, which performs robot
 * localization by processing point cloud data. It integrates depth images,
 * performs clustering, and searches for robots within the analyzed data using
 * sensor fusion techniques.
 * @date 2024-03-27
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>

#include <deque>
#include <functional>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <utility>

#include "robot/robot.h"

namespace radar {

/**
 * @brief Hash function for `cv::Point2i`.
 *
 * This struct defines a hash function for `cv::Point2i` objects. It combines
 * the hash values of the point's x and y using the XOR (^) operator.
 */
struct CvPoint2iHash {
    std::size_t operator()(const cv::Point2i& pt) const {
        std::size_t hx = std::hash<int>{}(pt.x);
        std::size_t hy = std::hash<int>{}(pt.y);
        return hx ^ hy;
    }
};

/**
 * @brief Class for robot localization using sensor fusion of point cloud data.
 *
 * The Locator class performs robot localization by processing point cloud data.
 * It integrates depth images from the point cloud, performs clustering to
 * identify objects, and searches for robots within the analyzed data.
 */
class Locator {
   public:
    friend class Radar;

    /**
     * @brief Deleted default constructor.
     *
     */
    Locator() = delete;

    /**
     * @brief Constructs a Locator object with the specified parameters.
     *
     * This constructor initializes a Locator object with the given parameters,
     * including image dimensions, camera intrinsic matrix, Lidar-to-camera
     * transformation matrix, world-to-camera transformation matrix, zoom
     * factor, scale factor, queue size, depth difference thresholds, cluster
     * tolerance, minimum and maximum cluster sizes, and maximum distance.
     *
     * @param image_width The width of the image.
     * @param image_height The height of the image.
     * @param intrinsic The camera intrinsic matrix.
     * @param lidar_to_camera The transformation matrix from Lidar to camera
     * coordinates.
     * @param world_to_camera The transformation matrix from world to camera
     * coordinates.
     * @param zoom_factor The zoom factor to apply to the image, which is used
     * to shrink the size of depth image in order to accelerate speed of
     * processing.
     * @param queue_size The size of the queue of depth images.
     * @param min_depth_diff The minimum depth difference for point cloud
     * differencing.
     * @param max_depth_diff The maximum depth difference for point cloud
     * differencing.
     * @param cluster_tolerance The cluster tolerance for point cloud
     * clustering.
     * @param min_cluster_size The minimum cluster size for point cloud
     * clustering.
     * @param max_cluster_size The maximum cluster size for point cloud
     * clustering.
     * @param max_distance The maximum distance threshold of x-coordinate for
     * point cloud processing.
     */
    Locator(int image_width, int image_height, const cv::Matx33f& intrinsic,
            const cv::Matx44f& lidar_to_camera,
            const cv::Matx44f& world_to_camera, float zoom_factor = 0.5f,
            size_t queue_size = 3, float min_depth_diff = 500,
            float max_depth_diff = 4000, float cluster_tolerance = 400,
            int min_cluster_size = 8, int max_cluster_size = 1000,
            float max_distance = 29300);

    /**
     * @brief Updates the Locator with a new point cloud.
     *
     * This method updates the Locator with a new point cloud. It processes the
     * input point cloud to generate depth images, perform differencing with the
     * background depth image, and updates the depth image of difference between
     * the background image and current images stored in a queue.
     *
     * @param cloud The input point cloud.
     */
    void update(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) noexcept;

    /**
     * @brief Clusters points based on depth values from a differential depth
     * image.
     *
     * This method iterates over the differential depth image stored in
     * `diff_depth_image_` and collects non-zero points into a vector. Each
     * point contains the column (x), the row (y), and the depth value (z) from
     * the depth image. These points are then clustered using the DBSCAN
     * algorithm and then stored in an image.
     */
    void cluster() noexcept;

    /**
     * @brief Searches for robots within the specified vector of robots.
     *
     * This method searches for robots within the specified vector of Robot
     * objects by invoking the `search` method for each individual robot in
     * parallel. It updates the location of each robot based on the analysis of
     * the differencing depth image.
     *
     * @param robots The vector of Robot objects to search for.
     */
    void search(std::vector<Robot>& robot) const noexcept;

   private:
    /**
     * @brief Converts a point from the camera coordinate system to the Lidar
     * coordinate system.
     *
     * This function takes a point in the camera coordinate system and applies
     * the necessary transformations to convert it to the Lidar coordinate
     * system.
     *
     * @param point The point in the camera coordinate system to be converted.
     * @return The corresponding point in the Lidar coordinate system.
     */
    cv::Point3f cameraToLidar(const cv::Point3f& point) const noexcept;

    /**
     * @brief Converts a point from the Lidar coordinate system to the world
     * coordinate system.
     *
     * This function takes a point in the Lidar coordinate system and applies
     * the appropriate transformations to convert it to the world coordinate
     * system.
     *
     * @param point The point in the Lidar coordinate system to be converted.
     * @return The corresponding point in the world coordinate system.
     */
    cv::Point3f lidarToWorld(const cv::Point3f& point) const noexcept;

    /**
     * @brief Converts a point from the Lidar coordinate system to the camera
     * coordinate system.
     *
     * This function takes a point in the Lidar coordinate system and applies
     * the necessary transformations to convert it to the camera coordinate
     * system.
     *
     * @param point The point in the Lidar coordinate system to be converted.
     * @return The corresponding point in the camera coordinate system.
     */
    cv::Point3f lidarToCamera(const cv::Point3f& point) const noexcept;

    /**
     * @brief Searches for the robot within the specified region of interest.
     *
     * This method searches for the robot within the specified region of
     * interest (ROI) by analyzing the differencing depth image. It identifies
     * candidate points within the ROI, associates them with clusters, and
     * determines the location of the robot based on the largest cluster of
     * points.
     *
     * @param robot The robot object.
     */
    void search(Robot& robot) const noexcept;

    /**
     * @brief Applies zoom transformations to the input rectangle.
     *
     * @param rect The input rectangle to be transformed.
     * @return The transformed rectangle after applying zoom transform.
     *
     * @note The `zoom` factor is used to shrink the size of depth image in
     * order to accelerate processing.
     */
    cv::Rect zoom(const cv::Rect& rect) const noexcept;

    /**
     * @brief The width of the input image.
     */
    int image_width_;

    /**
     * @brief The height of the input image.
     */
    int image_height_;

    /**
     * @brief The zoom factor applied to the image to accelerate processing.
     *
     * The zoom factor is used to shrink the size of the depth image to
     * accelerate processing.
     */
    float zoom_factor_;

    /**
     * @brief The width of the image after applying the zoom factor.
     */
    int image_width_zoomed_;

    /**
     * @brief The height of the image after applying the zoom factor.
     */
    int image_height_zoomed_;

    /**
     * @brief The maximum size of the depth image queue.
     */
    size_t queue_size_;

    /**
     * @brief The camera's intrinsic matrix.
     *
     * This matrix contains the intrinsic parameters of the camera, such as
     * focal length and principal point.
     */
    cv::Matx33f intrinsic_;

    /**
     * @brief The inverse of the camera's intrinsic matrix.
     */
    cv::Matx33f intrinsic_inv_;

    /**
     * @brief The transformation matrix from Lidar coordinates to camera
     * coordinates.
     */
    cv::Matx44f lidar_to_camera_transform_;

    /**
     * @brief The translation vector from camera coordinates to Lidar
     * coordinates.
     */
    cv::Matx31f camera_to_lidar_translate_;

    /**
     * @brief The rotation matrix from camera coordinates to Lidar coordinates.
     */
    cv::Matx33f camera_to_lidar_rotate_;

    /**
     * @brief The transformation matrix from camera coordinates to world
     * coordinates.
     */
    cv::Matx44f camera_to_world_transform_;

    /**
     * @brief The minimum depth difference threshold for point cloud
     * differencing.
     */
    float min_depth_diff_;

    /**
     * @brief The maximum depth difference threshold for point cloud
     * differencing.
     */
    float max_depth_diff_;

    /**
     * @brief The maximum distance threshold for point cloud processing.
     *
     * This threshold defines the maximum x-coordinate distance for processing
     * points in the point cloud.
     */
    float max_distance_;

    /**
     * @brief The current depth image.
     */
    cv::Mat depth_image_;

    /**
     * @brief The background depth image used for point cloud differencing.
     */
    cv::Mat background_depth_image_;

    /**
     * @brief The differential depth image showing differences between the
     * current and background depth images.
     */
    cv::Mat diff_depth_image_;

    /**
     * @brief A queue of depth images for temporal processing.
     *
     * This deque stores a rolling history of depth images for comparison and
     * analysis.
     */
    std::deque<cv::Mat> depth_images_;

    /**
     * @brief The point cloud cluster extractor.
     *
     * This object is responsible for extracting clusters from the point cloud
     * using the EuclideanClusterExtraction algorithm.
     */
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> cluster_extractor_;

    /**
     * @brief A KD-Tree search structure for efficient point cloud queries.
     *
     * This KD-Tree is used for efficient nearest-neighbor searches within the
     * point cloud.
     */
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_;

    /**
     * @brief A map from 2D points to their corresponding indices in the point
     * cloud.
     *
     * This unordered map associates 2D image points (in pixel coordinates) with
     * their corresponding indices in the 3D point cloud data.
     */
    std::unordered_map<cv::Point2i, int, CvPoint2iHash> point_index_map_;

    /**
     * @brief A map from point indices to cluster indices.
     *
     * This map associates 3D point indices with their corresponding cluster
     * indices after clustering is performed.
     */
    std::map<int, int> index_cluster_map_;

    /**
     * @brief A vector of point cloud clusters.
     *
     * This vector contains the indices of points belonging to each cluster
     * after clustering is performed on the point cloud.
     */
    std::vector<pcl::PointIndices> clusters_;

    /**
     * @brief A point cloud containing foreground objects.
     *
     * This point cloud contains the points that have been identified as
     * foreground objects after differencing the depth images.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_foreground_;
};

}  // namespace radar