/**
 * @file lidar.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the definition of the Lidar class, which provides
 * an abstract interface for interacting with generic LiDAR devices. It includes
 * methods for connecting, disconnecting, starting, stopping, and managing
 * point cloud data acquisition.
 * @date 2024-10-31
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <atomic>
#include <span>
#include <string>

namespace radar::lidar {

/**
 * @brief The Lidar class is an abstract base class representing a generic LiDAR
 * device. It provides an interface for connecting, disconnecting, starting, and
 * stopping the device, as well as managing point cloud data.
 */
class Lidar {
   public:
    /**
     * @brief Virtual destructor for proper cleanup in derived classes.
     */
    virtual ~Lidar() = default;

    /**
     * @brief Connects to the LiDAR device.
     *
     * @return true if the connection is successful, false otherwise.
     */
    virtual bool connect() = 0;

    /**
     * @brief Disconnects from the LiDAR device.
     *
     * @return true if the disconnection is successful, false otherwise.
     */
    virtual bool disconnect() = 0;

    /**
     * @brief Starts the LiDAR device, enabling data acquisition.
     *
     * @return true if the device starts successfully, false otherwise.
     */
    virtual bool start() = 0;

    /**
     * @brief Stops the LiDAR device, halting data acquisition.
     *
     * @return true if the device stops successfully, false otherwise.
     */
    virtual bool stop() = 0;

    /**
     * @brief Retrieves model information about the LiDAR device.
     *
     * @return A string containing the model information.
     */
    virtual std::string getModelInfo() const = 0;

    /**
     * @brief Retrieves the current point cloud data from the LiDAR device.
     *
     * @param cloud A pointer to a PCL point cloud container to store the
     * retrieved point cloud data.
     */
    virtual void getPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) = 0;

    /**
     * @brief Clears the current point cloud data.
     */
    virtual void clearPointCloud() = 0;

    /**
     * @brief Checks if the LiDAR device is currently connected.
     *
     * @return true if the device is connected, false otherwise.
     */
    bool isConnected() { return is_connected_; }

    /**
     * @brief Checks if the LiDAR device is currently started.
     *
     * @return true if the device is started, false otherwise.
     */
    bool isStarted() { return is_started_; }

    /**
     * @brief Checks if the LiDAR device is currently stopped.
     *
     * @return true if the device is stopped, false otherwise.
     */
    bool isStopped() { return !is_started_; }

   protected:
    /**
     * @brief Atomic boolean indicating whether the device is connected.
     */
    std::atomic_bool is_connected_ = false;

    /**
     * @brief Atomic boolean indicating whether the device is started.
     */
    std::atomic_bool is_started_ = false;
};

}  // namespace radar::lidar