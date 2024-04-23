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

    Locator() = delete;

    Locator(int image_width, int image_height, const cv::Matx33f& intrinsic,
            const cv::Matx44f& lidar_to_camera,
            const cv::Matx44f& world_to_camera, float zoom_factor = 0.5f,
            size_t queue_size = 3, float min_depth_diff = 500,
            float max_depth_diff = 4000, float cluster_tolerance = 40,
            int min_cluster_size = 8, int max_cluster_size = 1000,
            float max_distance = 29300);

    void update(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) noexcept;

    void cluster() noexcept;

    void search(std::vector<Robot>& robot) const noexcept;

   private:
    cv::Point3f cameraToLidar(const cv::Point3f& point) const noexcept;
    cv::Point3f lidarToWorld(const cv::Point3f& point) const noexcept;
    cv::Point3f lidarToCamera(const cv::Point3f& point) const noexcept;
    void search(Robot& robot) const noexcept;
    cv::Rect zoom(const cv::Rect& rect) const noexcept;
    int image_width_, image_height_;
    float zoom_factor_;
    int image_width_zoomed_, image_height_zoomed_;
    size_t queue_size_;
    cv::Matx33f intrinsic_, intrinsic_inv_;
    cv::Matx44f lidar_to_camera_transform_;
    cv::Matx31f camera_to_lidar_translate_;
    cv::Matx33f camera_to_lidar_rotate_;
    cv::Matx44f camera_to_world_transform_;
    float min_depth_diff_, max_depth_diff_;
    float max_distance_;
    cv::Mat depth_image_, background_depth_image_, diff_depth_image_;
    std::deque<cv::Mat> depth_images_;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> cluster_extractor_;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_;
    std::unordered_map<cv::Point2i, int, CvPoint2iHash> point_index_map_;
    std::map<int, int> index_cluster_map_;
    std::vector<pcl::PointIndices> clusters_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_foreground_;
};

}  // namespace radar