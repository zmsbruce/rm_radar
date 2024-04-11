/**
 * @file frame.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file defines a Frame class which is a collection of image, point
 * cloud and time stamp.
 * @date 2024-04-11
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <optional>

namespace radar {

/**
 * @brief Class representing a frame with image, point cloud, and timestamp.
 */
class Frame {
   public:
    Frame() = default;

    /**
     * @brief Construct a new Frame object with image, point cloud, and optional
     * timestamp.
     *
     * @param image A cv::Mat image associated with the frame.
     * @param point_cloud A shared pointer to a pcl::PointCloud of pcl::PointXYZ
     * points.
     * @param timestamp A time_point object representing the time the frame was
     * captured.
     */
    Frame(const cv::Mat& image, pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud,
          std::chrono::high_resolution_clock::time_point timestamp =
              std::chrono::high_resolution_clock::now())
        : image_(image), point_cloud_(point_cloud), timestamp_(timestamp) {}

    virtual ~Frame() = default;

    /**
     * @brief Get the image associated with the frame.
     *
     * @return An optional containing the image if it is valid, std::nullopt
     * otherwise.
     */
    inline std::optional<cv::Mat> image() const noexcept {
        return isImageValid() ? std::make_optional(image_) : std::nullopt;
    }

    /**
     * @brief Get the point cloud associated with the frame.
     *
     * @return An optional containing the point cloud if it is valid,
     * std::nullopt otherwise.
     */
    inline auto point_cloud() const noexcept {
        return isPointCloudValid() ? std::make_optional(point_cloud_)
                                   : std::nullopt;
    }

    /**
     * @brief Get the timestamp of the frame.
     *
     * @return An optional containing the timestamp if it is valid, std::nullopt
     * otherwise.
     */
    inline auto timestamp() const noexcept {
        return isTimeStampValid() ? std::make_optional(timestamp_)
                                  : std::nullopt;
    }

    /**
     * @brief Set the timestamp of the frame.
     *
     * @param timestamp A time_point object to set as the frame's timestamp.
     */
    inline void setTime(
        std::chrono::high_resolution_clock::time_point timestamp) noexcept {
        timestamp_ = timestamp;
    }

    /**
     * @brief Set the image of the frame.
     *
     * @param image A cv::Mat object to set as the frame's image.
     */
    inline void setImage(const cv::Mat& image) { image_ = image; }

    /**
     * @brief Set the point cloud of the frame.
     *
     * @param point_cloud A shared pointer to a pcl::PointCloud to set as the
     * frame's point cloud.
     */
    inline void setPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud) {
        point_cloud_ = point_cloud;
    }

   protected:
    /**
     * @brief Check if the image is valid (non-empty).
     *
     * @return true If the image is not empty.
     */
    inline bool isImageValid() const noexcept { return !image_.empty(); }

    /**
     * @brief Check if the point cloud is valid (non-null).
     *
     * @return If the point cloud is not null.
     */
    inline bool isPointCloudValid() const noexcept {
        return point_cloud_ != nullptr;
    }

    /**
     * @brief Check if the timestamp is valid (non-zero).
     *
     * @return If the timestamp is valid.
     */
    inline bool isTimeStampValid() const noexcept {
        return timestamp_.time_since_epoch().count() != 0;
    }

    cv::Mat image_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_;
    std::chrono::high_resolution_clock::time_point timestamp_;
};

}  // namespace radar