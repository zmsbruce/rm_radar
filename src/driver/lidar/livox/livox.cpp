/**
 * @file livox.cpp
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the implementation of the LivoxLidar class, which
 * manages the connection, data acquisition, and control of Livox LiDAR devices.
 * @date 2024-10-31
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#include "livox.h"

#include <pcl/common/io.h>
#include <spdlog/spdlog.h>

#include <magic_enum/magic_enum.hpp>
#include <unordered_map>

namespace radar::lidar {

std::unordered_map<std::string, LivoxLidar*> LIDAR_INSTANCES;

std::shared_mutex LIDAR_INSTANCES_MUTEX;

LivoxLidar::LivoxLidar(std::string_view broadcast_code, size_t max_point_num,
                       LivoxCloudMode cloud_mode, LivoxImuEnable imu_enable)
    : broadcast_code_(broadcast_code),
      max_point_num_(max_point_num),
      point_cloud_(pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>()) {
    // Log the initialization values of the constructor
    spdlog::trace(
        "Constructing LivoxLidar instance with broadcast_code: {}, "
        "max_point_num: {}, cloud_mode: {}, imu_freq: {}",
        broadcast_code, max_point_num, magic_enum::enum_name(cloud_mode),
        magic_enum::enum_name(imu_enable));

    // Set cloud mode and imu frequency
    spdlog::trace("Setting cloud mode and imu frequency.");
    switch (cloud_mode) {
        case LivoxCloudMode::FirstReturn:
            cloud_mode_ = kFirstReturn;
            break;
        case LivoxCloudMode::DualReturn:
            cloud_mode_ = kDualReturn;
            break;
        case LivoxCloudMode::StrongestReturn:
            cloud_mode_ = kStrongestReturn;
            break;
        case LivoxCloudMode::TripleReturn:
            cloud_mode_ = kTripleReturn;
            break;
        default:
            assert(0 && "unreachable code in cloud mode setting.");
    }

    imu_freq_ = imu_enable == LivoxImuEnable::On ? kImuFreq200Hz : kImuFreq0Hz;
    spdlog::debug("Cloud mode is set to {}, imu frequency is set to {}",
                  static_cast<int>(cloud_mode_), static_cast<int>(imu_freq_));

    // Initialize the point cloud
    spdlog::debug("Initializing point cloud with max_point_num: {}",
                  max_point_num);
    point_cloud_->assign(max_point_num, pcl::PointXYZ(0, 0, 0));

    // Initialize the Livox SDK
    spdlog::debug("Initializing Livox SDK for broadcast_code: {}",
                  broadcast_code_);
    if (!Init()) {
        spdlog::error("Livox SDK initialization failed for broadcast_code: {}",
                      broadcast_code_);
        Uninit();  // Clean up resources
        std::string err_string = "Failed to initialize Livox-SDK.";
        spdlog::critical("{}", err_string);    // Log a critical error
        throw std::runtime_error(err_string);  // Throw an exception
    }

    // Log Livox SDK version information
    LivoxSdkVersion version;
    GetLivoxSdkVersion(&version);
    spdlog::info("Livox SDK initialized, version: {}.{}.{}", version.major,
                 version.minor, version.patch);

    // Check if an instance with the same broadcast code already exists in
    // LIDAR_INSTANCES
    spdlog::trace("Acquiring LIDAR_INSTANCES_MUTEX lock for broadcast_code: {}",
                  broadcast_code_);
    std::unique_lock lock(LIDAR_INSTANCES_MUTEX);

    if (LIDAR_INSTANCES.contains(broadcast_code_)) {
        auto err_string = fmt::format(
            "Another instance with the same broadcast code {} exists.",
            broadcast_code_);
        spdlog::critical(err_string);          // Log a critical error
        throw std::runtime_error(err_string);  // Throw an exception
    }

    // Add the current instance to LIDAR_INSTANCES
    spdlog::debug(
        "Adding LivoxLidar instance to LIDAR_INSTANCES with broadcast_code: {}",
        broadcast_code_);
    LIDAR_INSTANCES[broadcast_code_] = this;

    spdlog::info(
        "LivoxLidar instance created successfully with broadcast_code: {}",
        broadcast_code_);
}

LivoxLidar::~LivoxLidar() {
    spdlog::trace("Destroying LivoxLidar instance for broadcast_code: {}",
                  broadcast_code_);

    // Check if the device is currently connected
    if (isConnected()) {
        spdlog::debug("Lidar {} is connected, attempting to disconnect.",
                      broadcast_code_);

        // Attempt to disconnect the device
        LIVOX_CHECK_NORETURN(DisconnectDevice, "disconnect device", handle_,
                             onDisconnectDevice, nullptr);
        is_connected_ = false;
        spdlog::info("Lidar {} disconnected successfully.", broadcast_code_);
    } else {
        spdlog::debug("Lidar {} is not connected, no need to disconnect.",
                      broadcast_code_);
    }

    // Clean up and uninitialize the SDK
    spdlog::debug("Uninitializing the Livox SDK for broadcast_code: {}",
                  broadcast_code_);
    Uninit();
    spdlog::info("Livox SDK uninitialized for broadcast_code: {}",
                 broadcast_code_);

    // Remove the instance from the LIDAR_INSTANCES map
    spdlog::trace(
        "Acquiring LIDAR_INSTANCES_MUTEX lock to remove instance for "
        "broadcast_code: {}",
        broadcast_code_);
    std::unique_lock lock(LIDAR_INSTANCES_MUTEX);

    if (LIDAR_INSTANCES.erase(broadcast_code_) > 0) {
        spdlog::info(
            "LivoxLidar instance for broadcast_code {} erased from "
            "LIDAR_INSTANCES.",
            broadcast_code_);
    } else {
        spdlog::warn(
            "LivoxLidar instance for broadcast_code {} was not found in "
            "LIDAR_INSTANCES during destruction.",
            broadcast_code_);
    }

    spdlog::trace(
        "LivoxLidar instance for broadcast_code {} successfully destroyed.",
        broadcast_code_);
}

bool LivoxLidar::connect() {
    spdlog::trace("Attempting to connect LivoxLidar with broadcast_code: {}",
                  broadcast_code_);

    // Check if the LiDAR is already connected
    if (isConnected()) {
        spdlog::info("Lidar {} is already connected.", broadcast_code_);
        return false;
    }

    // Set up the broadcast and device state update callbacks
    spdlog::debug(
        "Setting broadcast and device state update callbacks for "
        "broadcast_code: {}",
        broadcast_code_);
    SetBroadcastCallback(LivoxLidar::onDeviceBroadcast);
    SetDeviceStateUpdateCallback(LivoxLidar::onDeviceChange);

    // Attempt to start the device
    spdlog::debug("Starting device for broadcast_code: {}", broadcast_code_);
    if (!Start()) {
        spdlog::error("Failed to start device for broadcast_code: {}",
                      broadcast_code_);
        Uninit();  // Clean up resources in case of failure
        spdlog::warn(
            "Device uninitialized after failed start attempt for "
            "broadcast_code: {}",
            broadcast_code_);
        return false;
    }

    spdlog::info("Lidar {} connected successfully.", broadcast_code_);
    return true;
}

bool LivoxLidar::disconnect() {
    spdlog::trace("Attempting to disconnect LivoxLidar with broadcast_code: {}",
                  broadcast_code_);

    // Check if the lidar is connected before attempting to disconnect
    if (!isConnected()) {
        spdlog::error("Lidar {} is not connected, cannot perform disconnect.",
                      broadcast_code_);
        return false;  // Early exit if not connected
    }

    // Attempt to disconnect the device using the macro
    spdlog::debug("Disconnecting Lidar {}...", broadcast_code_);
    LIVOX_CHECK_RETURN_BOOL(DisconnectDevice, "disconnect device", handle_,
                            onDisconnectDevice, this);

    // Log successful disconnection
    spdlog::info("Lidar {} disconnected successfully.", broadcast_code_);
    return true;
}

bool LivoxLidar::start() {
    spdlog::trace(
        "Attempting to start sampling for LivoxLidar with broadcast_code: {}",
        broadcast_code_);

    // Check if the LiDAR is connected before starting sampling
    if (!isConnected()) {
        spdlog::error("Lidar {} is not connected, cannot start sampling.",
                      broadcast_code_);
        return false;
    }

    // Attempt to start the sampling process
    spdlog::debug("Starting LiDAR sampling for broadcast_code: {}",
                  broadcast_code_);
    LIVOX_CHECK_RETURN_BOOL(LidarStartSampling, "start sampling", handle_,
                            onStartSampling, this);

    // Log successful start of sampling
    spdlog::info("Lidar {} started sampling successfully.", broadcast_code_);
    return true;
}

bool LivoxLidar::stop() {
    spdlog::trace(
        "Attempting to stop sampling for LivoxLidar with broadcast_code: {}",
        broadcast_code_);

    // Check if the LiDAR is connected before stopping sampling
    if (!isConnected()) {
        spdlog::error("Lidar {} is not connected, cannot stop sampling.",
                      broadcast_code_);
        return false;
    }

    // Attempt to stop the sampling process
    spdlog::debug("Stopping LiDAR sampling for broadcast_code: {}",
                  broadcast_code_);
    LIVOX_CHECK_RETURN_BOOL(LidarStopSampling, "stop sampling", handle_,
                            onStopSampling, this);

    // Log successful stop of sampling
    spdlog::info("Lidar {} stopped sampling successfully.", broadcast_code_);
    return true;
}

void LivoxLidar::getPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    spdlog::trace(
        "Attempting to retrieve point cloud for LivoxLidar with "
        "broadcast_code: {}",
        broadcast_code_);

    // Lock the point cloud data with a shared lock for thread-safe access
    spdlog::debug(
        "Acquiring shared lock on point cloud data for broadcast_code: {}",
        broadcast_code_);
    std::shared_lock lock(cloud_mutex_);

    // Check if the point cloud is valid before copying
    if (!point_cloud_ || point_cloud_->empty()) {
        spdlog::warn(
            "Point cloud is empty or not initialized for broadcast_code: {}",
            broadcast_code_);
    } else {
        // Copy the point cloud data
        pcl::copyPointCloud(*point_cloud_, *cloud);
        spdlog::debug(
            "Point cloud successfully copied for broadcast_code: {}. Point "
            "count: {}",
            broadcast_code_, point_cloud_->size());
    }
}

void LivoxLidar::clearPointCloud() {
    spdlog::trace(
        "Attempting to clear point cloud for LivoxLidar with broadcast_code: "
        "{}",
        broadcast_code_);

    // Acquire a unique lock for thread-safe point cloud modification
    spdlog::debug(
        "Acquiring unique lock to clear point cloud for broadcast_code: {}",
        broadcast_code_);
    std::unique_lock lock(cloud_mutex_);

    // Check if the point cloud is already empty
    if (point_cloud_->empty()) {
        spdlog::warn("Point cloud for broadcast_code: {} is already empty.",
                     broadcast_code_);
    } else {
        // Clear the point cloud
        point_cloud_->clear();
        spdlog::info("Point cloud cleared successfully for broadcast_code: {}.",
                     broadcast_code_);
    }
}

void LivoxLidar::onDeviceBroadcast(const BroadcastDeviceInfo* info) {
    // Check if the broadcast information is null
    if (info == nullptr) {
        spdlog::error("Received null BroadcastDeviceInfo.");
        return;
    }

    spdlog::trace("Processing broadcast from device with broadcast_code: {}",
                  info->broadcast_code);

    LivoxLidar* instance = nullptr;

    {
        // Acquire shared lock to safely access LIDAR_INSTANCES
        spdlog::debug("Acquiring shared lock to search for broadcast_code: {}",
                      info->broadcast_code);
        std::shared_lock lock(LIDAR_INSTANCES_MUTEX);

        // Find the instance matching the broadcast code
        auto iter = LIDAR_INSTANCES.find(info->broadcast_code);
        if (iter != LIDAR_INSTANCES.end()) {
            instance = iter->second;
            spdlog::debug("Found matching instance for broadcast_code: {}",
                          info->broadcast_code);
        } else {
            spdlog::warn("No matching instance found for broadcast_code: {}",
                         info->broadcast_code);
        }
    }

    // If an instance was found, proceed with the connection
    if (instance != nullptr) {
        spdlog::info(
            "Broadcast code {} matched. Attempting to connect the lidar.",
            info->broadcast_code);

        // Attempt to add the LiDAR for connection
        LIVOX_CHECK_RETURN_VOID(AddLidarToConnect, "connect lidar",
                                info->broadcast_code, &instance->handle_);

        // Set up the data callback for receiving LiDAR data
        spdlog::debug("Setting data callback for lidar with broadcast_code: {}",
                      info->broadcast_code);
        SetDataCallback(instance->handle_, LivoxLidar::onGetLidarData,
                        &instance);

        spdlog::info(
            "Lidar with broadcast_code: {} is now connected and data callback "
            "is set.",
            info->broadcast_code);
    } else {
        // If no matching instance was found, log a warning
        spdlog::warn(
            "Broadcast code {} not matched, broadcast will be ignored.",
            info->broadcast_code);
    }
}

void LivoxLidar::onDeviceChange(const DeviceInfo* info, DeviceEvent event) {
    // Check if the device info is null
    if (info == nullptr) {
        spdlog::error("Device info in device state update callback is null.");
        return;
    }

    spdlog::trace(
        "Processing device change for broadcast_code: {} with event: {}",
        info->broadcast_code, static_cast<int>(event));

    LivoxLidar* instance = nullptr;

    {
        // Acquire shared lock to safely access LIDAR_INSTANCES
        spdlog::debug("Acquiring shared lock to search for broadcast_code: {}",
                      info->broadcast_code);
        std::shared_lock lock(LIDAR_INSTANCES_MUTEX);

        // Find the instance matching the broadcast code
        auto iter = LIDAR_INSTANCES.find(info->broadcast_code);
        if (iter != LIDAR_INSTANCES.end()) {
            instance = iter->second;
            spdlog::debug("Found matching instance for broadcast_code: {}",
                          info->broadcast_code);
        } else {
            spdlog::error("No matching instance found for broadcast_code: {}",
                          info->broadcast_code);
        }
    }

    // If no instance was found, log an error and return
    if (instance == nullptr) {
        spdlog::error(
            "Lidar instance not found in device change callback for "
            "broadcast_code: {}",
            info->broadcast_code);
        return;
    }

    // Check if the device handle matches the instance handle
    if (instance->handle_ != info->handle) {
        spdlog::error(
            "Mismatch in lidar handles. Device info handle: {} does not match "
            "instance handle: {} for broadcast_code: {}.",
            info->handle, instance->handle_, info->broadcast_code);
        return;
    }

    // Handle different device events
    switch (event) {
        case kEventConnect:
            spdlog::info("Lidar {} connected.", info->broadcast_code);
            // Query device information on connection
            LIVOX_CHECK_RETURN_VOID(
                QueryDeviceInformation, "query device information",
                info->handle, onQueryDeviceInfo, static_cast<void*>(&instance));
            instance->is_connected_ = true;
            instance->device_info_ = *info;
            break;

        case kEventDisconnect:
            spdlog::info("Lidar {} disconnected.", info->broadcast_code);
            instance->is_connected_ = false;
            break;

        case kEventStateChange:
            spdlog::info("Lidar {} state changed.", info->broadcast_code);
            instance->device_info_ = *info;
            break;

        default:
            spdlog::info("Lidar {} hub connection changed.",
                         info->broadcast_code);
            break;
    }

    // If the device is connected, set additional callbacks and configurations
    if (instance->isConnected()) {
        spdlog::info("Model info for Lidar {}: {}", info->broadcast_code,
                     instance->getModelInfo());

        // Set error message callback
        LIVOX_CHECK_RETURN_VOID(SetErrorMessageCallback,
                                "set error message callback", info->handle,
                                onErrorMessage);
        // Set cartesian coordinate system
        LIVOX_CHECK_RETURN_VOID(SetCartesianCoordinate,
                                "set cartesian coordinate", info->handle,
                                onSetCartesianCoor, &instance);

        // Set the point cloud return mode and IMU frequency for specific device
        // types
        if (info->type != kDeviceTypeLidarMid40) {
            spdlog::debug("Setting point cloud return mode for Lidar {}.",
                          info->broadcast_code);
            LIVOX_CHECK_RETURN_VOID(
                LidarSetPointCloudReturnMode, "set point cloud return mode",
                info->handle, instance->cloud_mode_, onSetCloudMode, &instance);
        }
        if (info->type != kDeviceTypeLidarMid40 &&
            info->type != kDeviceTypeLidarMid70) {
            spdlog::debug("Setting IMU push frequency for Lidar {}.",
                          info->broadcast_code);
            LIVOX_CHECK_RETURN_VOID(
                LidarSetImuPushFrequency, "set imu push frequency",
                info->handle, instance->imu_freq_, onSetImuFreq, &instance);
        }
    }
}

void LivoxLidar::onGetLidarData(uint8_t handle, LivoxEthPacket* eth_packet,
                                uint32_t data_num, void* client_data) {
    // Check if client data is null
    if (client_data == nullptr) {
        spdlog::error("Client data in data callback is null.");
        return;
    }

    // Check if the Ethernet packet is null
    if (eth_packet == nullptr) {
        spdlog::error("Point cloud packet is null.");
        return;
    }

    auto lidar = static_cast<LivoxLidar*>(client_data);

    // Warn if the handle does not match the lidar instance handle
    if (handle != lidar->handle_) {
        spdlog::warn(
            "Handle mismatch: received handle {} does not match lidar instance "
            "handle {}.",
            handle, lidar->handle_);
    }

    spdlog::trace("Processing LiDAR data for handle: {}, data_num: {}", handle,
                  data_num);

    // Lambda function to push points into the point cloud with a limit
    const auto pushPointWithLimit = [lidar](int x, int y, int z) {
        lidar->point_cloud_->points[lidar->point_index_++] =
            pcl::PointXYZ(x, y, z);
        lidar->point_index_ %= lidar->max_point_num_;
    };

    // Lambda function to detect noise based on tag values
    const auto isNoise = [](uint8_t tag) -> bool {
        constexpr uint8_t spatial_noise_flag = 0b0001;
        constexpr uint8_t intensity_noise_flag = 0b0100;
        return (tag & spatial_noise_flag) || (tag & intensity_noise_flag);
    };

    // Acquire unique lock for thread-safe point cloud modification
    spdlog::debug("Acquiring unique lock to modify point cloud.");
    std::unique_lock lock(lidar->cloud_mutex_);

    // Process the point cloud data based on the data type
    switch (eth_packet->data_type) {
        case kCartesian: {
            spdlog::debug("Processing Cartesian data.");
            auto begin = reinterpret_cast<LivoxRawPoint*>(eth_packet->data);
            for (size_t i = 0; i < data_num; ++i) {
                const LivoxRawPoint& p = begin[i];
                if (p.x != 0 && p.y != 0 && p.z != 0) {
                    pushPointWithLimit(p.x, p.y, p.z);
                }
            }
            break;
        }
        case kExtendCartesian: {
            spdlog::debug("Processing Extended Cartesian data.");
            auto begin =
                reinterpret_cast<LivoxExtendRawPoint*>(eth_packet->data);
            for (size_t i = 0; i < data_num; ++i) {
                const LivoxExtendRawPoint& p = begin[i];
                if (p.x != 0 && p.y != 0 && p.z != 0 && !isNoise(p.tag)) {
                    pushPointWithLimit(p.x, p.y, p.z);
                }
            }
            break;
        }
        case kDualExtendCartesian: {
            spdlog::debug("Processing Dual Extended Cartesian data.");
            auto begin =
                reinterpret_cast<LivoxDualExtendRawPoint*>(eth_packet->data);
            for (size_t i = 0; i < data_num; ++i) {
                const LivoxDualExtendRawPoint& p = begin[i];
                if (p.x1 != 0 && p.y1 != 0 && p.z1 != 0 && !isNoise(p.tag1)) {
                    pushPointWithLimit(p.x1, p.y1, p.z1);
                }
                if (p.x2 != 0 && p.y2 != 0 && p.z2 != 0 && !isNoise(p.tag2)) {
                    pushPointWithLimit(p.x2, p.y2, p.z2);
                }
            }
            break;
        }
        case kTripleExtendCartesian: {
            spdlog::debug("Processing Triple Extended Cartesian data.");
            auto begin =
                reinterpret_cast<LivoxTripleExtendRawPoint*>(eth_packet->data);
            for (size_t i = 0; i < data_num; ++i) {
                const LivoxTripleExtendRawPoint& p = begin[i];
                if (p.x1 != 0 && p.y1 != 0 && p.z1 != 0 && !isNoise(p.tag1)) {
                    pushPointWithLimit(p.x1, p.y1, p.z1);
                }
                if (p.x2 != 0 && p.y2 != 0 && p.z2 != 0 && !isNoise(p.tag2)) {
                    pushPointWithLimit(p.x2, p.y2, p.z2);
                }
                if (p.x3 != 0 && p.y3 != 0 && p.z3 != 0 && !isNoise(p.tag3)) {
                    pushPointWithLimit(p.x3, p.y3, p.z3);
                }
            }
            break;
        }
        default:
            spdlog::error(
                "Unsupported data type: spherical points are not supported.");
            break;
    }

    spdlog::info("Processing completed for handle {} with {} points.", handle,
                 data_num);
}

void LivoxLidar::onQueryDeviceInfo(livox_status status, uint8_t handle,
                                   DeviceInformationResponse* ack,
                                   void* client_data) {
    // Check if the status indicates failure
    if (status != kStatusSuccess) {
        spdlog::error("Failed to query device information for handle {}: {}",
                      handle, getLivoxStatusString(status));
        return;
    }

    // Ensure the acknowledgment (ack) is not null
    if (ack == nullptr) {
        spdlog::error("Received null DeviceInformationResponse for handle {}.",
                      handle);
        return;
    }

    // Log the firmware version if available
    spdlog::info("Received device information for handle {}.", handle);
    spdlog::info("Firmware version: {}.{}.{}.{}", ack->firmware_version[0],
                 ack->firmware_version[1], ack->firmware_version[2],
                 ack->firmware_version[3]);

    // Optionally check client_data to log additional information
    if (client_data == nullptr) {
        spdlog::warn(
            "Client data is null for handle {} in device info response.",
            handle);
    } else {
        auto lidar = static_cast<LivoxLidar*>(client_data);
        if (lidar->handle_ == handle) {
            spdlog::info(
                "Device info queried for LiDAR with broadcast code: {}",
                lidar->device_info_.broadcast_code);
        } else {
            spdlog::warn(
                "Handle mismatch: queried handle {} does not match LiDAR "
                "instance handle {}.",
                handle, lidar->handle_);
        }
    }
}

void LivoxLidar::onErrorMessage(livox_status status, uint8_t handle,
                                ErrorMessage* message) {
    // Check if the error message is null
    if (message == nullptr) {
        spdlog::error("Received null error message for handle {}.", handle);
        return;
    }

    // Log the status if it's not successful
    if (status != kStatusSuccess) {
        spdlog::warn(
            "Non-success status received in error message callback: {} for "
            "handle {}.",
            getLivoxStatusString(status), handle);
    }

    // Extract the lidar error code from the message
    auto code = message->lidar_error_code;

    // Lambda functions to interpret different status codes
    const auto get_temp_status = [](uint32_t temp_status) {
        switch (temp_status) {
            case 0:
                return "normal";
            case 1:
                return "high or low";
            case 2:
                return "extremely high or extremely low";
            default:
                return "unknown";
        }
    };

    const auto get_volt_status = [](uint32_t volt_status) {
        switch (volt_status) {
            case 0:
                return "normal";
            case 1:
                return "high";
            case 2:
                return "extremely high";
            default:
                return "unknown";
        }
    };

    const auto get_motor_status = [](uint32_t motor_status) {
        switch (motor_status) {
            case 0:
                return "normal";
            case 1:
                return "warning";
            case 2:
                return "error";
            default:
                return "unknown";
        }
    };

    const auto get_time_sync_status = [](uint32_t time_sync_status) {
        switch (time_sync_status) {
            case 0:
                return "not started";
            case 1:
                return "ptp 1588";
            case 2:
                return "gps";
            case 3:
                return "pps";
            case 4:
                return "abnormal";
            default:
                return "unknown";
        }
    };

    const auto get_system_status = [](uint32_t system_status) {
        switch (system_status) {
            case 0:
                return "normal";
            case 1:
                return "warning";
            case 2:
                return "error";
            default:
                return "unknown";
        }
    };

    // Log the detailed error message
    spdlog::error(
        "Lidar error message for handle {}: {{ "
        "temperature: {}, voltage: {}, motor: {}, dirty status: {}, firmware "
        "status: {}, "
        "PPS status: {}, device status: {}, fan status: {}, self-heating: {}, "
        "PTP status: {}, time sync status: {}, system status: {} }}",
        handle, get_temp_status(code.temp_status),
        get_volt_status(code.volt_status), get_motor_status(code.motor_status),
        code.dirty_warn == 0 ? "not dirty or blocked" : "dirty or blocked",
        code.firmware_err == 0 ? "ok" : "upgrade needed",
        code.pps_status == 0 ? "no signal" : "signal ok",
        code.device_status == 0 ? "normal" : "end of service",
        code.fan_status == 0 ? "normal" : "warning",
        code.self_heating == 0 ? "normal" : "on",
        code.ptp_status == 0 ? "no signal" : "signal ok",
        get_time_sync_status(code.time_sync_status),
        get_system_status(code.system_status));
}

void LivoxLidar::onSetCartesianCoor(livox_status status, uint8_t handle,
                                    uint8_t response, void* client_data) {
    // Check if client data is valid
    if (client_data == nullptr) {
        spdlog::error(
            "Client data is null in SetCartesianCoor callback for handle {}.",
            handle);
        return;
    }

    // Cast client_data to the expected LivoxLidar instance
    auto lidar = static_cast<LivoxLidar*>(client_data);

    // Check if the handle matches the lidar instance's handle
    if (handle != lidar->handle_) {
        spdlog::warn(
            "Handle mismatch: received handle {} does not match lidar instance "
            "handle {}.",
            handle, lidar->handle_);
    }

    // Log the result of the operation based on the status
    if (status == kStatusSuccess) {
        spdlog::info(
            "Successfully set cartesian coordinate for handle {}. Response: "
            "{}.",
            handle, response);
    } else {
        spdlog::error(
            "Failed to set cartesian coordinate for handle {}. Status: {}, "
            "Response: {}.",
            handle, getLivoxStatusString(status), response);
    }
}

void LivoxLidar::onSetImuFreq(livox_status status, uint8_t handle,
                              uint8_t response, void* client_data) {
    // Check if client_data is valid
    if (client_data == nullptr) {
        spdlog::error(
            "Client data is null in SetImuFreq callback for handle {}.",
            handle);
        return;
    }

    // Cast client_data to the expected LivoxLidar instance
    auto lidar = static_cast<LivoxLidar*>(client_data);

    // Check if the handle matches the lidar instance's handle
    if (handle != lidar->handle_) {
        spdlog::warn(
            "Handle mismatch: received handle {} does not match lidar instance "
            "handle {}.",
            handle, lidar->handle_);
    }

    // Log the result of the operation based on the status
    if (status == kStatusSuccess) {
        spdlog::info(
            "Successfully set IMU frequency for handle {}. Response: {}.",
            handle, response);
    } else {
        spdlog::error(
            "Failed to set IMU frequency for handle {}. Status: {}, Response: "
            "{}.",
            handle, getLivoxStatusString(status), response);
    }
}

void LivoxLidar::onDisconnectDevice(livox_status status, uint8_t handle,
                                    uint8_t response, void* client_data) {
    // Check if client_data is valid
    if (client_data == nullptr) {
        spdlog::warn(
            "Client data is null in DisconnectDevice callback for handle {}. "
            "This is only permitted in deconstruction.",
            handle);
    }

    // Cast client_data to LivoxLidar instance
    auto lidar = static_cast<LivoxLidar*>(client_data);

    // Check if the handle matches the lidar instance's handle
    if (lidar != nullptr && handle != lidar->handle_) {
        spdlog::warn(
            "Handle mismatch: received handle {} does not match lidar instance "
            "handle {}.",
            handle, lidar->handle_);
    }

    // Log the result of the operation based on the status
    if (status == kStatusSuccess) {
        spdlog::info(
            "Successfully disconnected lidar with handle {}. Response: {}.",
            handle, response);

        // Update lidar connection status
        if (lidar != nullptr) {
            lidar->is_connected_ = false;
            spdlog::debug(
                "Lidar connection status updated: handle {} is now "
                "disconnected.",
                handle);
        } else {
            spdlog::trace(
                "Lidar pointer is null, will not change status. This can only "
                "occur in deconstruction.");
        }
    } else {
        spdlog::error(
            "Failed to disconnect lidar with handle {}. Status: {}, Response: "
            "{}.",
            handle, getLivoxStatusString(status), response);
    }
}

void LivoxLidar::onSetCloudMode(livox_status status, uint8_t handle,
                                uint8_t response, void* client_data) {
    // Check if client_data is valid
    if (client_data == nullptr) {
        spdlog::error(
            "Client data is null in SetCloudMode callback for handle {}.",
            handle);
        return;
    }

    // Cast client_data to the expected LivoxLidar instance
    auto lidar = static_cast<LivoxLidar*>(client_data);

    // Check if the handle matches the lidar instance's handle
    if (handle != lidar->handle_) {
        spdlog::warn(
            "Handle mismatch: received handle {} does not match lidar instance "
            "handle {}.",
            handle, lidar->handle_);
    }

    // Log the result of the operation based on the status
    if (status == kStatusSuccess) {
        spdlog::info(
            "Successfully set cloud mode for lidar with handle {}. Response: "
            "{}.",
            handle, response);
    } else {
        spdlog::error(
            "Failed to set cloud mode for lidar with handle {}. Status: {}, "
            "Response: {}.",
            handle, getLivoxStatusString(status), response);
    }
}

void LivoxLidar::onStartSampling(livox_status status, uint8_t handle,
                                 uint8_t response, void* client_data) {
    // Check if client_data is valid
    if (client_data == nullptr) {
        spdlog::error(
            "Client data is null in StartSampling callback for handle {}.",
            handle);
        return;
    }

    // Cast client_data to the expected LivoxLidar instance
    auto lidar = static_cast<LivoxLidar*>(client_data);

    // Check if the handle matches the lidar instance's handle
    if (handle != lidar->handle_) {
        spdlog::warn(
            "Handle mismatch: received handle {} does not match lidar instance "
            "handle {}.",
            handle, lidar->handle_);
    }

    // Log the result of the operation based on the status
    if (status == kStatusSuccess) {
        spdlog::info(
            "Successfully started sampling for lidar with handle {}. Response: "
            "{}.",
            handle, response);
        lidar->is_started_ = true;  // Mark the lidar as started
    } else {
        spdlog::error(
            "Failed to start sampling for lidar with handle {}. Status: {}, "
            "Response: {}.",
            handle, getLivoxStatusString(status), response);
    }
}

void LivoxLidar::onStopSampling(livox_status status, uint8_t handle,
                                uint8_t response, void* client_data) {
    // Check if client_data is valid
    if (client_data == nullptr) {
        spdlog::error(
            "Client data is null in StopSampling callback for handle {}.",
            handle);
        return;
    }

    // Cast client_data to the expected LivoxLidar instance
    auto lidar = static_cast<LivoxLidar*>(client_data);

    // Check if the handle matches the lidar instance's handle
    if (handle != lidar->handle_) {
        spdlog::warn(
            "Handle mismatch: received handle {} does not match lidar instance "
            "handle {}.",
            handle, lidar->handle_);
    }

    // Log the result of the operation based on the status
    if (status == kStatusSuccess) {
        spdlog::info(
            "Successfully stopped sampling for lidar with handle {}. Response: "
            "{}.",
            handle, response);
        lidar->is_started_ = false;  // Mark the lidar as stopped
    } else {
        spdlog::error(
            "Failed to stop sampling for lidar with handle {}. Status: {}, "
            "Response: {}.",
            handle, getLivoxStatusString(status), response);
    }
}

std::string LivoxLidar::getLivoxStatusString(livox_status status) {
    switch (status) {
        case kStatusSuccess:
            return "success";
        case kStatusFailure:
            return "failure";
        case kStatusNotConnected:
            return "not connected";
        case kStatusNotSupported:
            return "not supported";
        case kStatusTimeout:
            return "time out";
        case kStatusNotEnoughMemory:
            return "not enough memory";
        case kStatusChannelNotExist:
            return "channel not exist";
        case kStatusInvalidHandle:
            return "invalid handle";
        case kStatusHandlerImplNotExist:
            return "handler implement not exist";
        case kStatusSendFailed:
            return "send failed";
        default:
            return "unknown status";
    }
}

std::string LivoxLidar::getModelInfo() const {
    auto LidarStateToString = [](LidarState state) -> std::string {
        switch (state) {
            case kLidarStateInit:
                return "Initialization";
            case kLidarStateNormal:
                return "Normal";
            case kLidarStatePowerSaving:
                return "PowerSaving";
            case kLidarStateStandBy:
                return "StandBy";
            case kLidarStateError:
                return "Error";
            case kLidarStateUnknown:
                return "Unknown";
            default:
                return "Invalid";
        }
    };

    auto LidarFeatureToString = [](LidarFeature feature) -> std::string {
        switch (feature) {
            case kLidarFeatureNone:
                return "None";
            case kLidarFeatureRainFog:
                return "RainFog";
            default:
                return "Unknown";
        }
    };

    std::string status_description =
        (device_info_.state == kLidarStateNormal)
            ? fmt::format("Progress: {}", device_info_.status.progress)
            : fmt::format("ErrorCode: {}",
                          device_info_.status.status_code.error_code);

    std::string firmware_version = fmt::format(
        "{}.{}.{}.{}", device_info_.firmware_version[0],
        device_info_.firmware_version[1], device_info_.firmware_version[2],
        device_info_.firmware_version[3]);

    return fmt::format(
        "Livox Lidar: {{ BroadcastCode: {}, Handle: {}, Slot: {}, ID: {}, "
        "Type: {}, DataPort: {}, CmdPort: {}, SensorPort: {}, IP: {}, State: "
        "{}, Feature: {}, FirmwareVersion: {}, Status: {} }}",
        device_info_.broadcast_code, device_info_.handle, device_info_.slot,
        device_info_.id, device_info_.type, device_info_.data_port,
        device_info_.cmd_port, device_info_.sensor_port, device_info_.ip,
        LidarStateToString(device_info_.state),
        LidarFeatureToString(device_info_.feature), firmware_version,
        status_description);
}

}  // namespace radar::lidar