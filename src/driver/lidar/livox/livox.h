/**
 * @file livox.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief Header file for the LivoxLidar class, providing an interface for
 * interacting with Livox LiDAR devices.
 *
 * This file contains the definition of the `LivoxLidar` class and related
 * enumerations, which allow for configuring and controlling Livox LiDAR
 * devices. It includes methods for connecting to devices, starting and stopping
 * data sampling, retrieving point cloud data, and handling various callbacks
 * from the Livox SDK. Additionally, it defines enums for configuring point
 * cloud return modes and IMU status.
 *
 * @date 2024-10-31
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <livox_def.h>
#include <livox_sdk.h>

#include <memory>
#include <shared_mutex>
#include <string_view>

#include "driver/lidar/lidar.h"

/**
 * @brief Executes a function and checks its return status. If the function
 * returns an error, logs the error message and returns from the current
 * function (used for void-returning functions).
 *
 * @param func The function to be called.
 * @param msg A message describing the operation for error logging.
 * @param ... Any additional parameters to be passed to the function.
 *
 * @note This macro is designed for functions that return `void`. If the
 * function does not succeed, it logs the error and immediately returns from the
 * caller.
 */
#define LIVOX_CHECK_RETURN_VOID(func, msg, ...)                   \
    do {                                                          \
        livox_status ret = func(__VA_ARGS__);                     \
        if (kStatusSuccess != ret) {                              \
            spdlog::error("Failed to {}, error: {}", msg,         \
                          LivoxLidar::getLivoxStatusString(ret)); \
            return;                                               \
        }                                                         \
    } while (0)

/**
 * @brief Executes a function and checks its return status. If the function
 * returns an error, logs the error message and returns `false` from the calling
 * function.
 *
 * @param func The function to be called.
 * @param msg A message describing the operation for error logging.
 * @param ... Any additional parameters to be passed to the function.
 *
 * @return `false` if the function returns an error.
 *
 * @note This macro is designed for functions that return `bool`. If the
 * function fails, it logs the error and returns `false`.
 */
#define LIVOX_CHECK_RETURN_BOOL(func, msg, ...)                   \
    do {                                                          \
        livox_status ret = func(__VA_ARGS__);                     \
        if (kStatusSuccess != ret) {                              \
            spdlog::error("Failed to {}, error: {}", msg,         \
                          LivoxLidar::getLivoxStatusString(ret)); \
            return false;                                         \
        }                                                         \
    } while (0)

/**
 * @brief Executes a function and checks its return status. If the function
 * returns an error, logs the error message but does not return from the calling
 * function.
 *
 * @param func The function to be called.
 * @param msg A message describing the operation for error logging.
 * @param ... Any additional parameters to be passed to the function.
 *
 * @note This macro is designed for functions where failure does not require
 * returning from the caller. It logs the error but allows the caller to
 * continue executing.
 */
#define LIVOX_CHECK_NORETURN(func, msg, ...)                      \
    do {                                                          \
        livox_status ret = func(__VA_ARGS__);                     \
        if (kStatusSuccess != ret) {                              \
            spdlog::error("Failed to {}, error: {}", msg,         \
                          LivoxLidar::getLivoxStatusString(ret)); \
        }                                                         \
    } while (0)

namespace radar::lidar {

/**
 * @brief Enumeration representing the point cloud return modes for Livox LiDAR
 * devices.
 *
 * This enum defines the different modes for point cloud return data that the
 * Livox LiDAR device can be configured to use.
 */
enum class LivoxCloudMode {
    FirstReturn,
    DualReturn,
    StrongestReturn,
    TripleReturn,
};

/**
 * @brief Enumeration representing the IMU (Inertial Measurement Unit) status
 * for Livox LiDAR devices.
 *
 * This enum defines whether the Livox LiDAR device's IMU is enabled or
 * disabled.
 */
enum class LivoxImuEnable {
    Off,
    On,
};

/**
 * @brief The LivoxLidar class provides an interface for interacting with Livox
 * LiDAR devices.
 */
class LivoxLidar final : public Lidar {
   public:
    /**
     * @brief Constructs a LivoxLidar object.
     *
     * Initializes an instance of the LivoxLidar class with the provided
     * parameters.
     *
     * @param[in] broadcast_code The broadcast code of the LiDAR device, used
     * for identification.
     * @param[in] max_point_num The maximum number of points in the point cloud.
     * @param[in] cloud_mode The mode of point cloud data (e.g., first return,
     * dual return, etc.).
     * @param[in] imu_enable Specifies whether the IMU (Inertial Measurement
     * Unit) is enabled.
     */
    LivoxLidar(std::string_view broadcast_code, size_t max_point_num,
               LivoxCloudMode cloud_mode = LivoxCloudMode::DualReturn,
               LivoxImuEnable imu_enable = LivoxImuEnable::Off);

    /**
     * @brief Destructor for the LivoxLidar object.
     *
     * Cleans up resources and disconnects from the LiDAR device.
     */
    ~LivoxLidar();

    /**
     * @brief Connects to the LiDAR device.
     *
     * Establishes a connection with the LiDAR device using the broadcast code.
     *
     * @return Returns true if the connection is successful, otherwise false.
     */
    bool connect() override;

    /**
     * @brief Disconnects from the LiDAR device.
     *
     * Closes the connection to the LiDAR device.
     *
     * @return Returns true if the disconnection is successful, otherwise false.
     */
    bool disconnect() override;

    /**
     * @brief Retrieves the latest point cloud data.
     *
     * Retrieves the current point cloud stored within the LivoxLidar instance.
     *
     * @param[out] cloud A pointer to a PointCloud object where the point cloud
     * data will be copied.
     */
    void getPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) override;

    /**
     * @brief Clears the stored point cloud data.
     *
     * Removes all points from the internal point cloud storage.
     */
    void clearPointCloud() override;

    /**
     * @brief Retrieves detailed information about the LiDAR device in a
     * formatted string.
     *
     * This method provides a human-readable description of the LiDAR device,
     * including its broadcast code, handle, slot, ID, type, ports, IP address,
     * state, features, firmware version, and status. The state and features are
     * converted to string representations for better readability.
     *
     * @return A formatted string containing the device information.
     */
    std::string getModelInfo() const override;

   private:
    /**
     * @brief Handles the broadcast from a Livox device and attempts to connect
     * it.
     *
     * This callback is triggered when a Livox device broadcasts its
     * information. The function checks if the broadcast information matches any
     * existing instances based on the broadcast code. If a matching instance is
     * found, the function proceeds to connect the device and set up the
     * necessary data callbacks.
     *
     * @param info A pointer to the BroadcastDeviceInfo structure containing the
     * device's broadcast details.
     */
    static void onDeviceBroadcast(const BroadcastDeviceInfo *info);

    /**
     * @brief Handles changes in the state of a Livox device.
     *
     * This callback is invoked when a Livox device experiences an event such as
     * connection, disconnection, or a state change. The function ensures the
     * correct instance is identified and processes the event accordingly.
     *
     * @param info A pointer to the DeviceInfo structure containing the device's
     * information.
     * @param event The type of event that triggered the callback.
     *
     * @note The function uses various helper macros (`LIVOX_CHECK_RETURN_VOID`)
     * to handle SDK function calls and error checking.
     */
    static void onDeviceChange(const DeviceInfo *info, DeviceEvent type);

    /**
     * @brief Callback function for processing incoming LiDAR data packets.
     *
     * This function is triggered when a Livox device sends point cloud data. It
     * processes the incoming Ethernet packet, extracts the point cloud data,
     * and stores it in the internal buffer. The function supports various types
     * of Cartesian data formats and ensures thread-safe access to the point
     * cloud.
     *
     * @param handle The handle of the LiDAR device sending the data.
     * @param eth_packet A pointer to the LivoxEthPacket structure containing
     * the point cloud data.
     * @param data_num The number of data points in the packet.
     * @param client_data A pointer to the LivoxLidar instance associated with
     * the callback.
     *
     * @note Noise filtering is applied for extended Cartesian formats using
     * specific tag values.
     * @note The point cloud data is stored in a circular buffer, ensuring it
     * doesn't exceed the maximum number of points (`max_point_num_`).
     */
    static void onGetLidarData(uint8_t handle, LivoxEthPacket *data,
                               uint32_t data_num, void *client_data);

    /**
     * @brief Callback function for handling the result of a device information
     * query.
     *
     * This function is called in response to a query for device information
     * from a Livox LiDAR device. It checks the status of the query and
     * processes the device information (such as firmware version) if the query
     * is successful.
     *
     * @param status The status of the query operation. This is typically
     * `kStatusSuccess` if the query was successful.
     * @param handle The handle of the LiDAR device for which the query was
     * made.
     * @param ack A pointer to the DeviceInformationResponse structure
     * containing the device's information, such as firmware version.
     * @param client_data A pointer to the LivoxLidar instance associated with
     * the query.
     *
     * @note If the query fails or the acknowledgment is null, appropriate error
     * messages are logged.
     */
    static void onQueryDeviceInfo(livox_status status, uint8_t handle,
                                  DeviceInformationResponse *ack,
                                  void *clent_data);

    /**
     * @brief Callback function for handling error messages from the LiDAR
     * device.
     *
     * This function is called when the LiDAR device reports an error. It checks
     * the status and error message, interprets various error codes, and logs
     * them in a human-readable format.
     *
     * @param status The status of the error message callback, typically
     * `kStatusSuccess` if the message was received correctly.
     * @param handle The handle of the LiDAR device that reported the error.
     * @param message A pointer to the ErrorMessage structure containing
     * detailed error information from the LiDAR device.
     *
     * @note The function uses several lambda functions to decode and interpret
     * different error codes (temperature, voltage, motor, etc.).
     *
     * @warning If the `message` pointer is null, the function logs an error and
     * returns early.
     */
    static void onErrorMessage(livox_status status, uint8_t handle,
                               ErrorMessage *message);
    /**
     * @brief Callback function for handling the result of setting the Cartesian
     * coordinate system on the LiDAR device.
     *
     * This function is called in response to a request to set the Cartesian
     * coordinate system for the LiDAR device. It checks the status of the
     * operation and logs whether the operation was successful or failed.
     *
     * @param status The status of the operation. Typically, `kStatusSuccess` if
     * the request was successful.
     * @param handle The handle of the LiDAR device for which the Cartesian
     * coordinate system was set.
     * @param response A byte indicating the response from the LiDAR device.
     * @param client_data A pointer to the LivoxLidar instance associated with
     * the request.
     *
     * @note This function is typically used when configuring the LiDAR device
     * to use Cartesian coordinates for point cloud data.
     *
     * @warning If `client_data` is null, the function logs an error and exits
     * early.
     */
    static void onSetCartesianCoor(livox_status status, uint8_t handle,
                                   uint8_t response, void *client_data);

    /**
     * @brief Callback function for handling the result of setting the point
     * cloud mode on the LiDAR device.
     *
     * This function is triggered in response to a request to set the point
     * cloud return mode on the LiDAR device. It checks the status of the
     * operation and logs whether the operation was successful or failed.
     *
     * @param status The status of the operation. Typically, `kStatusSuccess` if
     * the request succeeded.
     * @param handle The handle of the LiDAR device for which the cloud mode was
     * set.
     * @param response A byte indicating the response from the LiDAR device.
     * @param client_data A pointer to the `LivoxLidar` instance associated with
     * the request.
     *
     * @note This function is used to configure the device's point cloud return
     * mode (e.g., single return, dual return).
     *
     * @warning If `client_data` is null, the function logs an error and exits
     * early.
     */
    static void onSetCloudMode(livox_status status, uint8_t handle,
                               uint8_t response, void *client_data);

    /**
     * @brief Callback function for handling the result of setting the IMU data
     * frequency on the LiDAR device.
     *
     * This function is called in response to a request to set the IMU (Inertial
     * Measurement Unit) data frequency for the LiDAR device. It checks the
     * status of the operation and logs whether the operation was successful or
     * failed.
     *
     * @param status The status of the operation, typically `kStatusSuccess` if
     * the request succeeded.
     * @param handle The handle of the LiDAR device for which the IMU frequency
     * was set.
     * @param response A byte indicating the response from the LiDAR device.
     * @param client_data A pointer to the `LivoxLidar` instance associated with
     * the request.
     *
     * @note This function is used to configure the IMU data frequency for the
     * LiDAR device, which is important for synchronizing sensor data.
     *
     * @warning If `client_data` is null, the function logs an error and exits
     * early.
     */
    static void onSetImuFreq(livox_status status, uint8_t handle,
                             uint8_t response, void *client_data);

    /**
     * @brief Callback function for handling the result of starting data
     * sampling on the LiDAR device.
     *
     * This function is triggered in response to a request to start the sampling
     * process on the LiDAR device. It checks the status of the operation and
     * logs whether the operation was successful or failed.
     *
     * @param status The status of the operation, typically `kStatusSuccess` if
     * the request succeeded.
     * @param handle The handle of the LiDAR device for which the sampling was
     * started.
     * @param response A byte indicating the response from the LiDAR device.
     * @param client_data A pointer to the `LivoxLidar` instance associated with
     * the request.
     *
     * @note This function is used to initiate data sampling on the LiDAR
     * device. The device starts collecting and sending point cloud data once
     * sampling is started.
     *
     * @warning If `client_data` is null, the function logs an error and exits
     * early.
     */
    static void onStartSampling(livox_status status, uint8_t handle,
                                uint8_t response, void *client_data);

    /**
     * @brief Callback function for handling the result of stopping data
     * sampling on the LiDAR device.
     *
     * This function is triggered in response to a request to stop the sampling
     * process on the LiDAR device. It checks the status of the operation and
     * logs whether the operation was successful or failed.
     *
     * @param status The status of the operation, typically `kStatusSuccess` if
     * the request succeeded.
     * @param handle The handle of the LiDAR device for which the sampling was
     * stopped.
     * @param response A byte indicating the response from the LiDAR device.
     * @param client_data A pointer to the `LivoxLidar` instance associated with
     * the request.
     *
     *
     * @note This function is used to stop the data sampling on the LiDAR
     * device, halting the collection and transmission of point cloud data.
     *
     * @warning If `client_data` is null, the function logs an error and exits
     * early.
     */
    static void onStopSampling(livox_status status, uint8_t handle,
                               uint8_t response, void *client_data);

    /**
     * @brief Callback function for handling the result of a device
     * disconnection request.
     *
     * This function is triggered in response to a request to disconnect the
     * LiDAR device. It checks the status of the disconnection operation and
     * logs whether the operation was successful or failed.
     *
     * @param status The status of the operation, typically `kStatusSuccess` if
     * the disconnection request succeeded.
     * @param handle The handle of the LiDAR device being disconnected.
     * @param response A byte indicating the response from the LiDAR device.
     * @param client_data A pointer to the `LivoxLidar` instance associated with
     * the device.
     *
     * @note If `client_data` is `nullptr`, a trace log is produced, indicating
     * this situation is expected during object deconstruction.
     *
     * @warning If `client_data` is null but not during deconstruction, it logs
     * a warning and does not update the connection status.
     */
    static void onDisconnectDevice(livox_status status, uint8_t handle,
                                   uint8_t response, void *client_data);

    /**
     * @brief Converts a Livox status code to a human-readable string.
     *
     * This function maps Livox SDK status codes to corresponding descriptive
     * strings, which can be used for logging and debugging.
     *
     * @param status The Livox status code (of type `livox_status`).
     *
     * @return A string that describes the status code.
     *
     * @note This function helps in converting numeric status codes into
     * readable strings for easier interpretation in logs and error messages.
     */
    static std::string getLivoxStatusString(livox_status status);

    /**
     * @brief The broadcast code of the LiDAR device, used for identifying the
     * device when connecting.
     *
     * This unique code is used to identify and connect to a specific Livox
     * LiDAR device.
     */
    std::string broadcast_code_;

    /**
     * @brief The maximum number of points that the point cloud can hold.
     *
     * This variable sets the upper limit on the number of points that can be
     * stored in the point cloud buffer.
     */
    size_t max_point_num_;

    /**
     * @brief The point cloud return mode for the LiDAR device.
     *
     * This specifies how the LiDAR device provides point cloud data, such as
     * single return, dual return, etc.
     */
    PointCloudReturnMode cloud_mode_;

    /**
     * @brief The IMU data frequency setting for the LiDAR device.
     *
     * This determines how frequently the LiDAR device provides IMU data, which
     * includes measurements of acceleration and angular velocity for sensor
     * fusion.
     */
    ImuFreq imu_freq_;

    /**
     * @brief The handle used to identify the connected LiDAR device within the
     * Livox SDK.
     *
     * This handle is assigned by the Livox SDK and is used to interact with the
     * specific LiDAR device.
     */
    uint8_t handle_ = 0;

    /**
     * @brief A shared mutex used to protect access to the point cloud data.
     *
     * The point cloud data may be accessed by multiple threads, so this mutex
     * ensures thread-safe access and modification of the point cloud.
     */
    std::shared_mutex cloud_mutex_;

    /**
     * @brief A pointer to the point cloud data structure.
     *
     * This pointer holds the point cloud data retrieved from the LiDAR. It
     * stores the 3D coordinates of points measured by the LiDAR sensor.
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ = nullptr;

    /**
     * @brief The current index in the point cloud buffer for storing new
     * points.
     *
     * This index keeps track of where the next point should be stored in the
     * point cloud buffer, ensuring that new data overwrites old data when the
     * buffer is full.
     */
    size_t point_index_ = 0;

    /**
     * @brief Structure holding detailed information about the LiDAR device.
     *
     * This structure contains various hardware and configuration details about
     * the LiDAR device, including firmware version, serial number, and other
     * characteristics.
     */
    DeviceInfo device_info_;

    /**
     * @brief A flag indicating whether the LiDAR device is currently connected.
     *
     * This atomic boolean is used to track the connection status of the device
     * in a thread-safe manner, allowing multiple threads to safely check or
     * update the connection state.
     */
    std::atomic_bool is_connected_ = false;
};

}  // namespace radar::lidar