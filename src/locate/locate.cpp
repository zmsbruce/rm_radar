#include <algorithm>
#include <cmath>
#include <execution>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <ranges>

#include "locator.h"

namespace radar {

cv::Point3f Locator::lidarToWorld(const cv::Point3f& point) const noexcept {
    cv::Matx41f lidar_coor{point.x, point.y, point.z, 1.0f};
    cv::Matx41f world_coor =
        camera_to_world_transform_ * lidar_to_camera_transform_ * lidar_coor;
    return cv::Point3f(world_coor(0), world_coor(1), world_coor(2));
}

cv::Point3f Locator::cameraToLidar(const cv::Point3f& point) const noexcept {
    cv::Matx31f camera_coor{point.x, point.y, 1.0f};
    cv::Matx31f lidar_coor =
        camera_to_lidar_rotate_ *
        (intrinsic_inv_ * point.z * camera_coor + camera_to_lidar_translate_);
    return cv::Point3f(lidar_coor(0), lidar_coor(1), lidar_coor(2));
}

cv::Point3f Locator::lidarToCamera(const cv::Point3f& point) const noexcept {
    cv::Matx41f lidar_coor{point.x, point.y, point.z, 1.0f};
    cv::Matx31f camera_coor =
        intrinsic_ *
        (lidar_to_camera_transform_ * lidar_coor).get_minor<3, 1>(0, 0);
    return cv::Point3f(camera_coor(0) / camera_coor(2),
                       camera_coor(1) / camera_coor(2), camera_coor(2));
}

Locator::Locator(int image_width, int image_height,
                 const cv::Matx33f& intrinsic,
                 const cv::Matx44f& lidar_to_camera,
                 const cv::Matx44f& world_to_camera, float min_depth_diff,
                 float max_depth_diff, float cluster_tolerance,
                 float min_cluster_size, float max_cluster_size,
                 float max_distance)
    : image_width_{image_width},
      image_height_{image_height},
      intrinsic_{intrinsic},
      lidar_to_camera_transform_{lidar_to_camera},
      min_depth_diff_{min_depth_diff},
      max_depth_diff_{max_depth_diff},
      max_distance_{max_distance},
      diff_cloud_{new pcl::PointCloud<pcl::PointXYZ>()},
      search_tree_{new pcl::search::KdTree<pcl::PointXYZ>()} {
    intrinsic_inv_ = intrinsic.inv();
    cv::Matx44f camera_to_lidar = lidar_to_camera.inv();
    camera_to_lidar_rotate_ = camera_to_lidar.get_minor<3, 3>(0, 0);
    camera_to_lidar_translate_ = camera_to_lidar.get_minor<3, 1>(0, 3);
    camera_to_world_transform_ = world_to_camera.inv();
    depth_image_ = cv::Mat::zeros(image_height, image_width, CV_32F);
    background_depth_image_ = cv::Mat::zeros(image_height, image_width, CV_32F);
    diff_depth_image_ = cv::Mat::zeros(image_height, image_width, CV_32F);
    extraction_.setClusterTolerance(cluster_tolerance);
    extraction_.setMinClusterSize(min_cluster_size);
    extraction_.setMaxClusterSize(max_cluster_size);
}

void Locator::update(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr) noexcept {
    depth_image_.setTo(0);
    diff_depth_image_.setTo(0);

    if (!cloud_ptr) {
        std::cerr << "cloud pointer is nullptr." << std::endl;
        return;
    }

    if (cloud_ptr->empty()) {
        std::cerr << "no points in cloud." << std::endl;
        return;
    }

    std::for_each(
        std::execution::par_unseq, cloud_ptr->begin(), cloud_ptr->end(),
        [this](const pcl::PointXYZ& point) {
            if (iszero(point.x) && iszero(point.y) && iszero(point.z)) {
                return;
            }
            if (point.x > max_distance_) {
                return;
            }
            auto uvd = lidarToCamera(cv::Point3f(point.x, point.y, point.z));
            float &u = uvd.x, &v = uvd.y, &d = uvd.z;
            if (u < 0 || u > image_width_ || v < 0 || v > image_height_) {
                return;
            }
            float& background_depth = background_depth_image_.at<float>(v, u);
            if (d > background_depth) {
                background_depth = d;
            }
            depth_image_.at<float>(v, u) = d;
        });

    for (int i = 0; i < depth_image_.rows; ++i) {
        const float* depth_row = depth_image_.ptr<float>(i);
        const float* background_row = background_depth_image_.ptr<float>(i);
        float* diff_row = diff_depth_image_.ptr<float>(i);
        for (int j = 0; j < depth_image_.cols; ++j) {
            const float value = depth_row[j];
            if (iszero(value)) {
                continue;
            }
            float diff = background_row[j] - value;
            if (diff >= min_depth_diff_ && diff <= max_depth_diff_) {
                diff_row[j] = value;
            }
        }
    }
}

void Locator::cluster() noexcept {
    diff_cloud_->clear();
    cluster_indices_.clear();
    pixel_index_map_.clear();
    index_cluster_map_.clear();

    for (int i = 0; i < diff_depth_image_.rows; ++i) {
        const float* image_row = diff_depth_image_.ptr<float>(i);
        for (int j = 0; j < diff_depth_image_.cols; ++j) {
            float value = image_row[j];
            if (iszero(value)) {
                continue;
            }
            auto point = cameraToLidar(cv::Point3f(j, i, value));
            diff_cloud_->emplace_back(point.x, point.y, point.z);
            pixel_index_map_[std::make_pair(i, j)] = diff_cloud_->size() - 1;
        }
    }

    if (diff_cloud_->empty()) {
        return;
    }

    search_tree_->setInputCloud(diff_cloud_);
    extraction_.setSearchMethod(search_tree_);
    extraction_.setInputCloud(diff_cloud_);
    extraction_.extract(cluster_indices_);

    for (size_t i = 0; i < cluster_indices_.size(); ++i) {
        const auto& indices{cluster_indices_[i]};
        for (int index : indices.indices) {
            index_cluster_map_[index] = i + 1;
        }
    }
}

void Locator::search(Robot& robot) noexcept {
    if (!robot.rect().has_value()) {
        return;
    }

    std::map<int, std::vector<cv::Point3f>> candidates;

    const auto& rect = robot.rect().value();
    for (int v = rect.y; v < rect.y + rect.height; ++v) {
        const float* image_row = diff_depth_image_.ptr<float>(v);
        for (int u = rect.x; u < rect.x + rect.width; ++u) {
            float depth = image_row[u];
            if (iszero(depth)) {
                continue;
            }
            int index = pixel_index_map_.at(std::make_pair(v, u));
            int cluster_id = index_cluster_map_.at(index);
            candidates[cluster_id].emplace_back(
                cameraToLidar(cv::Point3f(u, v, depth)));
        }
    }

    if (candidates.empty()) {
        return;
    }
    auto& [_, points] =
        *std::ranges::max_element(candidates, [](auto&& pair_a, auto&& pair_b) {
            return pair_a.second.size() < pair_b.second.size();
        });
    auto location = std::accumulate(points.begin(), points.end(),
                                    cv::Point3f(0.0f, 0.0f, 0.0f)) /
                    static_cast<float>(points.size());
    robot.setLocation(lidarToWorld(location));
}

void Locator::search(std::vector<Robot>& robots) noexcept {
    std::for_each(std::execution::par_unseq, robots.begin(), robots.end(),
                  [this](Robot& robot) { search(robot); });
}

}  // namespace radar