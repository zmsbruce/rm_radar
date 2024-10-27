/**
 * @file hik.cpp
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file defines the implementation of the HikCamera class which
 * handles the interfacing with the Hikvision camera using the SDK.
 * @date 2024-10-27
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#include "hik.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <vector>

/**
 * @brief Helper macro to call a function and check its return value.
 *
 * This macro streamlines the process of calling a function and checking
 * its return value. If the function call does not return `MV_OK`, the macro
 * logs a critical error message using `spdlog` and returns `false` from the
 * enclosing function.
 *
 * @param func The function to call. This function should return an integer
 * status code.
 * @param msg A descriptive message indicating the operation being performed.
 * This message will be included in the error log if the function fails.
 * @param ... Variadic arguments that are passed to the function `func`.
 */
#define CALL_AND_CHECK(func, msg, ...)                               \
    do {                                                             \
        int ret = func(__VA_ARGS__);                                 \
        if (MV_OK != ret) {                                          \
            spdlog::critical("Failed to {}, error code: {:#x}", msg, \
                             static_cast<unsigned int>(ret));        \
            return false;                                            \
        }                                                            \
    } while (0)

namespace radar::camera {

/**
 * @brief Constructor for HikCamera.
 * @param camera_sn Camera serial number.
 * @param width Camera resolution width.
 * @param height Camera resolution height.
 * @param exposure Camera exposure time.
 * @param gamma Gamma correction value.
 * @param gain Gain value.
 * @param grab_timeout Timeout for grabbing an image.
 * @param auto_white_balance Whether auto white balance is enabled.
 * @param balance_ratio White balance ratio for RGB channels.
 */
HikCamera::HikCamera(std::string_view camera_sn, unsigned int width,
                     unsigned int height, float exposure, float gamma,
                     float gain, unsigned int grab_timeout,
                     bool auto_white_balance,
                     std::array<unsigned int, 3>&& balance_ratio)
    : camera_sn_{camera_sn},
      width_{width},
      height_{height},
      exposure_{exposure},
      gamma_{gamma},
      gain_{gain},
      grab_timeout_{grab_timeout},
      auto_white_balance_{auto_white_balance},
      balance_ratio_{balance_ratio},
      frame_out_{std::make_unique<MV_FRAME_OUT>()},
      device_info_{std::make_unique<MV_CC_DEVICE_INFO>()} {
    spdlog::trace("Initializing HikCamera with SN: {}", camera_sn);
    spdlog::debug(
        "Camera parameters: width={}, height={}, exposure={}, gamma={}, "
        "gain={}, grab timeout: {}, auto white balance: {}, balance ratio: "
        "[{}, {}, {}]",
        width, height, exposure, gamma, gain, grab_timeout, auto_white_balance,
        balance_ratio[0], balance_ratio[1], balance_ratio[2]);

    // Initializing device info
    std::memset(device_info_.get(), 0, sizeof(MV_CC_DEVICE_INFO));
    spdlog::trace("Device info structure initialized.");

    // Starting daemon thread
    spdlog::trace("Starting daemon thread.");
    startDaemonThread();

    // Attempt to open the camera
    spdlog::trace("Attempting to open camera with SN: {}", camera_sn);
    if (!open()) {
        auto error_info =
            fmt::format("Failed to open camera with SN: {}", camera_sn);
        spdlog::error("{}", error_info);
        throw std::runtime_error(error_info);
    }

    // Attempt to start capturing
    spdlog::trace("Attempting to start capturing with camera: {}", camera_sn);
    if (!startCapture()) {
        auto error_info =
            fmt::format("Failed to start capturing with camera: {}", camera_sn);
        spdlog::error("{}", error_info);
        throw std::runtime_error(error_info);
    }
}

/**
 * @brief Destructor for HikCamera.
 */
HikCamera::~HikCamera() {
    spdlog::info("Destroying camera with serial number: {}", camera_sn_);

    // Check if the camera is open before attempting to close it
    spdlog::trace("Checking if camera with SN: {} is open.", camera_sn_);
    if (isOpen()) {
        spdlog::debug("Camera with SN: {} is open. Attempting to close it.",
                      camera_sn_);
        if (!close()) {
            spdlog::error("Camera with SN: {} failed to close.", camera_sn_);
        } else {
            spdlog::info("Camera with SN: {} successfully closed.", camera_sn_);
        }
    }

    spdlog::trace("Requesting daemon thread stop for camera with SN: {}.",
                  camera_sn_);
    daemon_thread_.request_stop();
}

/**
 * @brief Opens the camera for use.
 * @return True if the camera was opened successfully, false otherwise.
 */
bool HikCamera::open() {
    spdlog::info("Opening camera with serial number: {}", camera_sn_);

    // Acquire lock for thread-safety when opening the camera
    std::unique_lock lock(mutex_);
    spdlog::debug("Acquired lock for opening camera {}", camera_sn_);

    // Retrieve available device information
    spdlog::trace("Retrieving device information list.");
    auto device_info_list = HikCamera::getDeviceInfoList();
    spdlog::debug("Found {} devices.", device_info_list.size());

    // Search for the device in the list using the serial number
    spdlog::trace("Searching for device with serial number: {}.", camera_sn_);
    auto device_info_iter = std::ranges::find_if(
        device_info_list, [this](MV_CC_DEVICE_INFO* device_info) {
            return reinterpret_cast<const char*>(
                       device_info->SpecialInfo.stUsb3VInfo.chSerialNumber) ==
                   camera_sn_;
        });

    // Check if the device was found
    if (device_info_iter == device_info_list.end()) {
        spdlog::critical("No devices found for serial number: {}", camera_sn_);
        return false;
    }
    spdlog::info("Device with serial number {} found.", camera_sn_);

    // Create a handle for the camera device
    spdlog::trace("Creating handle for the camera.");
    CALL_AND_CHECK(MV_CC_CreateHandle, "create handle", &handle_,
                   *device_info_iter);
    spdlog::debug("Handle created successfully for camera {}.", camera_sn_);

    // Open the camera device
    spdlog::trace("Opening device for camera {}.", camera_sn_);
    CALL_AND_CHECK(MV_CC_OpenDevice, "open device", handle_);
    spdlog::info("Device for camera {} opened successfully.", camera_sn_);

    // Retrieve supported pixel formats
    MVCC_ENUMVALUE val;
    spdlog::trace("Getting supported pixel formats for camera {}.", camera_sn_);
    CALL_AND_CHECK(MV_CC_GetPixelFormat, "get pixel format", handle_, &val);
    auto supported_pixel_format =
        std::span(val.nSupportValue, val.nSupportedNum);
    spdlog::debug("Supported pixel format for camera {}: [{:#x}]", camera_sn_,
                  fmt::join(supported_pixel_format, ", "));

    // Set the pixel type based on supported formats
    spdlog::trace("Setting pixel type for camera {}.", camera_sn_);
    setPixelType(supported_pixel_format);
    spdlog::debug("Pixel type set successfully for camera {}.", camera_sn_);

    // Setting camera configurations: resolution, balance ratio, exposure,
    // gamma, gain Each function call logs success/failure internally
    spdlog::trace(
        "Setting camera configuration (resolution, balance ratio, exposure, "
        "gamma, gain) for camera {}.",
        camera_sn_);
    if (!setResolutionInner()) {
        spdlog::error("Failed to set resolution for camera {}.", camera_sn_);
        return false;
    }
    spdlog::debug("Resolution set successfully for camera {}.", camera_sn_);

    if (!setBalanceRatioInner()) {
        spdlog::error("Failed to set balance ratio for camera {}.", camera_sn_);
        return false;
    }
    spdlog::debug("Balance ratio set successfully for camera {}.", camera_sn_);

    if (!setExposureInner()) {
        spdlog::error("Failed to set exposure for camera {}.", camera_sn_);
        return false;
    }
    spdlog::debug("Exposure set successfully for camera {}.", camera_sn_);

    if (!setGammaInner()) {
        spdlog::error("Failed to set gamma for camera {}.", camera_sn_);
        return false;
    }
    spdlog::debug("Gamma set successfully for camera {}.", camera_sn_);

    if (!setGainInner()) {
        spdlog::error("Failed to set gain for camera {}.", camera_sn_);
        return false;
    }
    spdlog::debug("Gain set successfully for camera {}.", camera_sn_);

    // Register an exception callback to handle any errors during runtime
    spdlog::trace("Registering exception callback for camera {}.", camera_sn_);
    CALL_AND_CHECK(MV_CC_RegisterExceptionCallBack,
                   "register exception callback", handle_,
                   HikCamera::exceptionHandler, this);
    spdlog::info("Exception callback registered successfully for camera {}.",
                 camera_sn_);

    // Mark the camera as open
    is_open_ = true;
    spdlog::info("Camera {} opened successfully.", camera_sn_);
    return true;
}

/**
 * @brief Closes the camera.
 * @return True if the camera was closed successfully, false otherwise.
 */
bool HikCamera::close() {
    spdlog::info("Closing camera with serial number: {}", camera_sn_);

    // Check if the camera is currently capturing images
    spdlog::trace("Checking if camera {} is capturing.", camera_sn_);
    if (isCapturing()) {
        spdlog::debug("Camera {} is capturing, attempting to stop capture.",
                      camera_sn_);
        stopCapture();
        spdlog::info("Capture stopped successfully for camera {}.", camera_sn_);
    }

    // Acquire lock before modifying camera state
    spdlog::trace("Acquiring lock to close camera {}.", camera_sn_);
    {
        std::unique_lock lock(mutex_);
        spdlog::debug("Lock acquired for closing camera {}.", camera_sn_);

        // Close the camera device
        spdlog::trace("Closing device for camera {}.", camera_sn_);
        CALL_AND_CHECK(MV_CC_CloseDevice, "close device", handle_);
        spdlog::info("Device closed successfully for camera {}.", camera_sn_);

        // Destroy the camera handle
        spdlog::trace("Destroying handle for camera {}.", camera_sn_);
        CALL_AND_CHECK(MV_CC_DestroyHandle, "destroy handle", handle_);
        spdlog::info("Handle destroyed successfully for camera {}.",
                     camera_sn_);

        // Reset handle and mark as closed
        handle_ = nullptr;
        is_open_ = false;
        spdlog::debug("Camera {} handle reset and marked as closed.",
                      camera_sn_);
    }

    spdlog::info("Camera {} closed successfully.", camera_sn_);
    return true;
}

/**
 * @brief Reconnects the camera by closing and reopening it.
 * @return True if the camera was reconnected successfully, false otherwise.
 */
bool HikCamera::reconnect() {
    spdlog::warn("Reconnecting camera with serial number: {}", camera_sn_);

    // Attempt to close the camera
    spdlog::trace("Closing camera {} before attempting reconnection.",
                  camera_sn_);
    if (!close()) {
        spdlog::error("Failed to close camera {} during reconnection.",
                      camera_sn_);
    } else {
        spdlog::info("Camera {} closed successfully.", camera_sn_);
    }

    // Attempt to reopen the camera
    spdlog::trace("Reopening camera {} during reconnection.", camera_sn_);
    if (!open()) {
        spdlog::error("Failed to open camera {} during reconnection.",
                      camera_sn_);
        return false;
    } else {
        spdlog::info("Camera {} reconnected successfully.", camera_sn_);
        return true;
    }
}

/**
 * @brief Starts capturing images from the camera.
 * @return True if capturing started successfully, false otherwise.
 */
bool HikCamera::startCapture() {
    spdlog::trace(
        "Attempting to start capture on camera with serial number: {}",
        camera_sn_);

    // Acquire lock to ensure thread safety while starting the capture process
    spdlog::trace("Acquiring lock to start capture on camera {}.", camera_sn_);
    {
        std::unique_lock lock(mutex_);
        spdlog::debug("Lock acquired for starting capture on camera {}.",
                      camera_sn_);

        // Call the SDK's function to start grabbing frames
        spdlog::trace("Calling MV_CC_StartGrabbing for camera {}.", camera_sn_);
        CALL_AND_CHECK(MV_CC_StartGrabbing, "start grabbing", handle_);
        spdlog::debug("MV_CC_StartGrabbing called successfully for camera {}.",
                      camera_sn_);

        // Mark the camera status as capturing
        is_capturing_ = true;
        spdlog::debug("Camera {} marked as capturing.", camera_sn_);
    }

    spdlog::info("Capture started successfully on camera {}.", camera_sn_);
    return true;
}

/**
 * @brief Stops capturing images from the camera.
 * @return True if capturing stopped successfully, false otherwise.
 */
bool HikCamera::stopCapture() {
    spdlog::trace("Attempting to stop capture on camera with serial number: {}",
                  camera_sn_);

    // Acquire lock to ensure thread safety while stopping the capture process
    spdlog::trace("Acquiring lock to stop capture on camera {}.", camera_sn_);
    {
        std::unique_lock lock(mutex_);
        spdlog::debug("Lock acquired for stopping capture on camera {}.",
                      camera_sn_);

        // Call the SDK's function to stop grabbing frames
        spdlog::trace("Calling MV_CC_StopGrabbing for camera {}.", camera_sn_);
        CALL_AND_CHECK(MV_CC_StopGrabbing, "stop grabbing", handle_);
        spdlog::debug("MV_CC_StopGrabbing called successfully for camera {}.",
                      camera_sn_);

        // Update the camera's capturing status
        is_capturing_ = false;
        spdlog::debug("Camera {} marked as not capturing.", camera_sn_);
    }

    spdlog::info("Capture stopped successfully on camera {}.", camera_sn_);
    return true;
}

/**
 * @brief Grabs an image from the camera.
 * @param image Output cv::Mat containing the image.
 * @param pixel_format Desired pixel format for the image.
 * @return True if the image was grabbed successfully, false otherwise.
 */
bool HikCamera::grabImage(cv::Mat& image,
                          camera::PixelFormat pixel_format) noexcept {
    spdlog::debug("Attempting to grab image from camera with serial number: {}",
                  camera_sn_);

    // Check if the camera is open and capturing
    if (!isOpen() || !isCapturing()) {
        spdlog::error("Camera {} is not open or capturing.", camera_sn_);
        return false;
    }

    // Acquire lock to ensure thread-safe access to camera resources
    spdlog::trace("Acquiring lock to grab image from camera {}.", camera_sn_);
    {
        std::unique_lock lock(mutex_);
        spdlog::debug("Lock acquired for grabbing image from camera {}.",
                      camera_sn_);

        // Clear the frame output buffer before grabbing a new image
        spdlog::trace("Clearing frame output buffer for camera {}.",
                      camera_sn_);
        std::memset(frame_out_.get(), 0, sizeof(MV_FRAME_OUT));

        // Attempt to get the image buffer from the camera
        spdlog::trace("Calling MV_CC_GetImageBuffer for camera {}.",
                      camera_sn_);
        CALL_AND_CHECK(MV_CC_GetImageBuffer, "get image buffer", handle_,
                       frame_out_.get(), grab_timeout_);
        spdlog::debug("Image buffer retrieved successfully for camera {}.",
                      camera_sn_);
    }

    // Convert the buffer to a cv::Mat object
    spdlog::trace("Converting frame buffer to cv::Mat for camera {}.",
                  camera_sn_);
    image =
        cv::Mat(frame_out_->stFrameInfo.nHeight, frame_out_->stFrameInfo.nWidth,
                CV_8UC3, frame_out_->pBufAddr);

    // Check if the image is valid (not empty)
    if (image.empty()) {
        spdlog::error("Failed to grab image from camera {}: image is empty.",
                      camera_sn_);
        return false;
    }
    spdlog::debug("Image successfully converted to cv::Mat for camera {}.",
                  camera_sn_);

    // Convert the pixel format as needed
    spdlog::trace("Converting pixel format for camera {}.", camera_sn_);
    convertPixelFormat(image, pixel_format);
    spdlog::debug("Pixel format conversion completed for camera {}.",
                  camera_sn_);

    spdlog::trace("Image grabbed successfully from camera {}.", camera_sn_);
    return true;
}

/**
 * @brief Sets the pixel format for the camera.
 * @param supported_types Span of supported pixel formats.
 * @return True if the pixel format was set successfully, false otherwise.
 */
bool HikCamera::setPixelType(std::span<unsigned int> supported_types) {
    spdlog::debug("Attempting to set pixel format for camera {}.", camera_sn_);

    const std::vector<PixelType> candidate_types{
        PixelType::RGB8Packed,    PixelType::BayerBG8, PixelType::BayerGB8,
        PixelType::BayerGR8,      PixelType::BayerRG8, PixelType::YUV422_8,
        PixelType::YUV422_8_UYVY,
    };

    spdlog::trace("Supported pixel formats for camera {}: [{}]", camera_sn_,
                  fmt::join(supported_types, ", "));

    // Iterate over candidate types and check if any are supported
    for (PixelType type : candidate_types) {
        spdlog::trace("Checking if pixel format {} is supported.",
                      static_cast<unsigned int>(type));

        if (std::ranges::any_of(
                supported_types, [type](unsigned int supported_type) {
                    return static_cast<unsigned int>(type) == supported_type;
                })) {
            pixel_type_ = type;
            spdlog::debug("Pixel format {} selected for camera {}.",
                          static_cast<unsigned int>(pixel_type_), camera_sn_);
            break;
        }
    }

    // If no supported pixel type was found, log a critical error
    if (pixel_type_ == PixelType::Unknown) {
        spdlog::critical(
            "Failed to set pixel format for camera {}: no supported format "
            "found.",
            camera_sn_);
        return false;
    }

    // Set the pixel format using the selected pixel type
    spdlog::trace("Setting pixel format to {} for camera {}.",
                  static_cast<unsigned int>(pixel_type_), camera_sn_);
    CALL_AND_CHECK(MV_CC_SetPixelFormat, "set pixel format", handle_,
                   static_cast<unsigned int>(pixel_type_));

    spdlog::info("Pixel format set successfully to {} for camera {}.",
                 static_cast<unsigned int>(pixel_type_), camera_sn_);
    return true;
}

/**
 * @brief Gets the camera's resolution.
 * @return A pair of integers representing the width and height.
 */
std::pair<int, int> HikCamera::getResolution() const {
    std::shared_lock lock(mutex_);
    return std::make_pair(width_, height_);
}

/**
 * @brief Sets the camera's resolution.
 * @param width New width for the camera.
 * @param height New height for the camera.
 * @return True if the resolution was set successfully, false otherwise.
 */
bool HikCamera::setResolution(int width, int height) {
    std::unique_lock lock(mutex_);
    width_ = width;
    height_ = height;
    setBalanceRatioInner();
    return true;
}

/**
 * @brief Gets the current gain value.
 * @return The current gain value.
 */
float HikCamera::getGain() const {
    std::shared_lock lock(mutex_);
    return gain_;
}

/**
 * @brief Sets the gain value for the camera.
 * @param gain New gain value.
 * @return True if the gain was set successfully, false otherwise.
 */
bool HikCamera::setGain(float gain) {
    std::unique_lock lock(mutex_);
    gain_ = gain;
    setGainInner();
    return true;
}

/**
 * @brief Gets the current exposure time.
 * @return The current exposure time.
 */
int HikCamera::getExposureTime() const {
    std::shared_lock lock(mutex_);
    return static_cast<int>(exposure_);
}

/**
 * @brief Sets a new exposure time for the camera.
 * @param exposure New exposure time.
 * @return True if the exposure time was set successfully, false otherwise.
 */
bool HikCamera::setExposureTime(int exposure) {
    std::unique_lock lock(mutex_);
    exposure_ = exposure;
    setExposureInner();
    return true;
}

/**
 * @brief Gets the current white balance ratio.
 * @return An array representing the white balance ratio for RGB channels.
 */
std::array<unsigned int, 3> HikCamera::getBalanceRatio() const {
    std::shared_lock lock(mutex_);
    return balance_ratio_;
}

/**
 * @brief Sets the white balance ratio.
 * @param balance New white balance ratio for RGB channels.
 * @return True if the balance ratio was set successfully, false otherwise.
 */
bool HikCamera::setBalanceRatio(std::array<unsigned int, 3>&& balance) {
    std::unique_lock lock(mutex_);
    balance_ratio_ = balance;
    setBalanceRatioInner();
    return true;
}

/**
 * @brief Determines if auto white balance is enabled.
 * @return True if auto white balance is enabled, false otherwise.
 */
bool HikCamera::getBalanceRatioAuto() const {
    std::shared_lock lock(mutex_);
    return auto_white_balance_;
}

/**
 * @brief Enables or disables auto white balance.
 * @param balance_auto Whether to enable auto white balance.
 * @return True if the auto white balance setting was applied successfully,
 * false otherwise.
 */
bool HikCamera::setBalanceRatioAuto(bool balance_auto) {
    std::unique_lock lock(mutex_);
    auto_white_balance_ = balance_auto;
    setBalanceRatioInner();
    return true;
}

/**
 * @brief Gets the camera's serial number.
 * @return The camera's serial number as a string.
 */
std::string HikCamera::getCameraSn() const {
    std::shared_lock lock(mutex_);
    return camera_sn_;
}

/**
 * @brief Sets the resolution of the camera internally.
 * @return True if the resolution was set successfully, false otherwise.
 */
bool HikCamera::setResolutionInner() {
    CALL_AND_CHECK(MV_CC_SetWidth, "set width", handle_, width_);
    CALL_AND_CHECK(MV_CC_SetHeight, "set height", handle_, height_);
    return true;
}

/**
 * @brief Sets the white balance ratio internally.
 * @return True if the white balance ratio was set successfully, false
 * otherwise.
 */
bool HikCamera::setBalanceRatioInner() {
    if (auto_white_balance_) {
        CALL_AND_CHECK(MV_CC_SetBalanceWhiteAuto, "set balance white auto",
                       handle_,
                       static_cast<unsigned int>(BalanceWhiteAuto::Continuous));
    } else {
        CALL_AND_CHECK(MV_CC_SetBalanceWhiteAuto, "set balance white auto",
                       handle_,
                       static_cast<unsigned int>(BalanceWhiteAuto::Off));
        CALL_AND_CHECK(MV_CC_SetBalanceRatioRed, "set balance ratio red",
                       handle_, balance_ratio_[0]);
        CALL_AND_CHECK(MV_CC_SetBalanceRatioGreen, "set balance ratio green",
                       handle_, balance_ratio_[1]);
        CALL_AND_CHECK(MV_CC_SetBalanceRatioBlue, "set balance ratio blue",
                       handle_, balance_ratio_[2]);
    }
    return true;
}

/**
 * @brief Sets the exposure value internally.
 * @return True if the exposure value was set successfully, false otherwise.
 */
bool HikCamera::setExposureInner() {
    if (exposure_ > 0) {
        CALL_AND_CHECK(MV_CC_SetExposureAutoMode, "set exposure auto", handle_,
                       static_cast<unsigned int>(ExposureAuto::Off));
        CALL_AND_CHECK(MV_CC_SetExposureTime, "set exposure time", handle_,
                       exposure_);
    } else {
        CALL_AND_CHECK(MV_CC_SetExposureAutoMode, "set exposure auto", handle_,
                       static_cast<unsigned int>(ExposureAuto::Continuous));
    }
    return true;
}

/**
 * @brief Sets the gamma correction value internally.
 * @return True if the gamma value was set successfully, false otherwise.
 */
bool HikCamera::setGammaInner() {
    if (gamma_ > 0.0f) {
        CALL_AND_CHECK(MV_CC_SetBoolValue, "set gamma enable", handle_,
                       "GammaEnable", true);
        CALL_AND_CHECK(MV_CC_SetGammaSelector, "set gamma selector", handle_,
                       static_cast<unsigned int>(GammaSelector::User));
        CALL_AND_CHECK(MV_CC_SetGamma, "set gamma", handle_, gamma_);
    }
    return true;
}

/**
 * @brief Sets the gain value internally.
 * @return True if the gain value was set successfully, false otherwise.
 */
bool HikCamera::setGainInner() {
    if (gain_ > 0) {
        CALL_AND_CHECK(MV_CC_SetGainMode, "set gain auto", handle_,
                       static_cast<unsigned int>(GainAuto::Off));
        CALL_AND_CHECK(MV_CC_SetGain, "set gain", handle_, gain_);
    } else {
        CALL_AND_CHECK(MV_CC_SetGainMode, "set gain auto", handle_,
                       static_cast<unsigned int>(GainAuto::Continuous));
    }
    return true;
}

/**
 * @brief Checks if the camera is currently open.
 * @return True if the camera is open, false otherwise.
 */
bool HikCamera::isOpen() const { return is_open_; }

/**
 * @brief Checks if the camera is currently capturing images.
 * @return True if the camera is capturing, false otherwise.
 */
bool HikCamera::isCapturing() const { return is_capturing_; }

/**
 * @brief Sets the exception flag for the camera.
 * @param flag New value for the exception flag.
 */
void HikCamera::setExceptionFlag(bool flag) { exception_flag_ = flag; }

/**
 * @brief Gets the current value of the exception flag.
 * @return True if an exception has occurred, false otherwise.
 */
bool HikCamera::getExceptionFlag() const { return exception_flag_; }

/**
 * @brief Gets camera information from a device info structure.
 * @param device_info Pointer to the device info structure.
 * @return A formatted string containing the camera information.
 */
std::string HikCamera::getCameraInfo(MV_CC_DEVICE_INFO* device_info) {
    assert(device_info->nTLayerType == MV_USB_DEVICE && "Wrong device type");
    auto info = &device_info->SpecialInfo.stUsb3VInfo;
    return fmt::format(
        "device guid: {}, device version: {}, family name: {}, manufacturer "
        "name: {}, model name: {}, serial number: {}, user defined name: {}, "
        "vendor name: {}",
        reinterpret_cast<const char*>(info->chDeviceGUID),
        reinterpret_cast<const char*>(info->chDeviceVersion),
        reinterpret_cast<const char*>(info->chFamilyName),
        reinterpret_cast<const char*>(info->chManufacturerName),
        reinterpret_cast<const char*>(info->chModelName),
        reinterpret_cast<const char*>(info->chSerialNumber),
        reinterpret_cast<const char*>(info->chUserDefinedName),
        reinterpret_cast<const char*>(info->chVendorName));
}

/**
 * @brief Gets camera information from the current device.
 * @return A formatted string containing the camera information.
 */
std::string HikCamera::getCameraInfo() const {
    std::shared_lock lock(mutex_);
    return HikCamera::getCameraInfo(device_info_.get());
}

/**
 * @brief Exception handler callback function.
 * @param code Exception code.
 * @param user Pointer to the user-defined data (in this case, the camera).
 */
void HikCamera::exceptionHandler(unsigned int code, void* user) {
    auto camera = static_cast<HikCamera*>(user);
    spdlog::error("Exception occurred in HikCamera {}: code {}",
                  camera->getCameraSn(), code);
    camera->setExceptionFlag(true);
}

/**
 * @brief Starts a daemon thread to monitor camera exceptions.
 */
void HikCamera::startDaemonThread() {
    daemon_thread_ = std::jthread([this](std::stop_token token) {
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (getExceptionFlag() == true) {
                spdlog::info(
                    "Exception detected in HikCamera {} daemon thread. Try "
                    "reconnecting...",
                    camera_sn_);
                if (reconnect()) {
                    spdlog::info("Successfully reconnected HikCamera {}.",
                                 camera_sn_);
                    setExceptionFlag(false);
                } else {
                    spdlog::error(
                        "Failed to reconnect HikCamera {}, will wait and "
                        "reconnect again.",
                        camera_sn_);
                }
            }
        }
        spdlog::info("Daemon thread stopping...");
    });
    spdlog::info("HikCamera {} daemon thread started, id: {:#x}", camera_sn_,
                 std::hash<std::thread::id>{}(daemon_thread_.get_id()));
}

/**
 * @brief Retrieves the list of available device information.
 * @return A span of pointers to the device information structures.
 */
std::span<MV_CC_DEVICE_INFO*> HikCamera::getDeviceInfoList() {
    std::call_once(device_info_list_init_flag_, [] {
        device_info_list_ = std::make_shared<MV_CC_DEVICE_INFO_LIST>();
        std::memset(device_info_list_.get(), 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        spdlog::trace("Initialized MV_CC_DEVICE_INFO_LIST structure");

        int ret = MV_CC_EnumDevices(MV_USB_DEVICE, device_info_list_.get());
        if (ret != MV_OK) {
            spdlog::critical("Failed to enum devices, error code: {}", ret);
        }

        spdlog::info("Enumerated devices, found {} devices.",
                     device_info_list_->nDeviceNum);
        for (unsigned int i = 0; i < device_info_list_->nDeviceNum; ++i) {
            auto device_info = device_info_list_->pDeviceInfo[i];
            spdlog::info("Camera {}: {}", i,
                         HikCamera::getCameraInfo(device_info));
        }
    });
    return std::span<MV_CC_DEVICE_INFO*>(device_info_list_->pDeviceInfo,
                                         device_info_list_->nDeviceNum);
}

/**
 * @brief Converts the pixel format of an image.
 * @param image The image to convert.
 * @param format The desired pixel format.
 */
void HikCamera::convertPixelFormat(cv::Mat& image, PixelFormat format) {
    switch (pixel_type_) {
        case PixelType::RGB8Packed:
            switch (format) {
                case PixelFormat::GRAY:
                    cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
                    break;
                case PixelFormat::RGB:
                    break;
                case PixelFormat::BGR:
                    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
                    break;
                case PixelFormat::RGBA:
                    cv::cvtColor(image, image, cv::COLOR_RGB2RGBA);
                    break;
                case PixelFormat::BGRA:
                    cv::cvtColor(image, image, cv::COLOR_RGB2BGRA);
                    break;
                case PixelFormat::HSV:
                    cv::cvtColor(image, image, cv::COLOR_RGB2HSV);
                    break;
                case PixelFormat::YUV:
                    cv::cvtColor(image, image, cv::COLOR_RGB2YUV);
                    break;
                default:
                    assert(0 && "unreachable code");
            }
            break;

        case PixelType::BayerBG8:
            switch (format) {
                case PixelFormat::GRAY:
                    cv::cvtColor(image, image, cv::COLOR_BayerBG2GRAY);
                    break;
                case PixelFormat::BGR:
                    cv::cvtColor(image, image, cv::COLOR_BayerBG2BGR);
                    break;
                case PixelFormat::RGB:
                    cv::cvtColor(image, image, cv::COLOR_BayerBG2RGB);
                    break;
                case PixelFormat::BGRA:
                    cv::cvtColor(image, image, cv::COLOR_BayerBG2BGRA);
                    break;
                case PixelFormat::RGBA:
                    cv::cvtColor(image, image, cv::COLOR_BayerBG2RGBA);
                    break;
                case PixelFormat::HSV: {
                    cv::cvtColor(image, image, cv::COLOR_BayerBG2BGR);
                    cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
                    break;
                }
                case PixelFormat::YUV: {
                    cv::cvtColor(image, image, cv::COLOR_BayerBG2BGR);
                    cv::cvtColor(image, image, cv::COLOR_BGR2YUV);
                    break;
                }
                default:
                    assert(0 && "unreachable code");
            }
            break;

        case PixelType::BayerGB8:
            switch (format) {
                case PixelFormat::GRAY:
                    cv::cvtColor(image, image, cv::COLOR_BayerGB2GRAY);
                    break;
                case PixelFormat::BGR:
                    cv::cvtColor(image, image, cv::COLOR_BayerGB2BGR);
                    break;
                case PixelFormat::RGB:
                    cv::cvtColor(image, image, cv::COLOR_BayerGB2RGB);
                    break;
                case PixelFormat::BGRA:
                    cv::cvtColor(image, image, cv::COLOR_BayerGB2BGRA);
                    break;
                case PixelFormat::RGBA:
                    cv::cvtColor(image, image, cv::COLOR_BayerGB2RGBA);
                    break;
                case PixelFormat::HSV: {
                    cv::cvtColor(image, image, cv::COLOR_BayerGB2BGR);
                    cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
                    break;
                }
                case PixelFormat::YUV: {
                    cv::cvtColor(image, image, cv::COLOR_BayerGB2BGR);
                    cv::cvtColor(image, image, cv::COLOR_BGR2YUV);
                    break;
                }
                default:
                    assert(0 && "unreachable code");
            }
            break;

        case PixelType::BayerGR8:
            switch (format) {
                case PixelFormat::GRAY:
                    cv::cvtColor(image, image, cv::COLOR_BayerGR2GRAY);
                    break;
                case PixelFormat::BGR:
                    cv::cvtColor(image, image, cv::COLOR_BayerGR2BGR);
                    break;
                case PixelFormat::RGB:
                    cv::cvtColor(image, image, cv::COLOR_BayerGR2RGB);
                    break;
                case PixelFormat::BGRA:
                    cv::cvtColor(image, image, cv::COLOR_BayerGR2BGRA);
                    break;
                case PixelFormat::RGBA:
                    cv::cvtColor(image, image, cv::COLOR_BayerGR2RGBA);
                    break;
                case PixelFormat::HSV: {
                    cv::cvtColor(image, image, cv::COLOR_BayerGR2BGR);
                    cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
                    break;
                }
                case PixelFormat::YUV: {
                    cv::cvtColor(image, image, cv::COLOR_BayerGR2BGR);
                    cv::cvtColor(image, image, cv::COLOR_BGR2YUV);
                    break;
                }
                default:
                    assert(0 && "unreachable code");
            }
            break;

        case PixelType::BayerRG8:
            switch (format) {
                case PixelFormat::GRAY:
                    cv::cvtColor(image, image, cv::COLOR_BayerRG2GRAY);
                    break;
                case PixelFormat::BGR:
                    cv::cvtColor(image, image, cv::COLOR_BayerRG2BGR);
                    break;
                case PixelFormat::RGB:
                    cv::cvtColor(image, image, cv::COLOR_BayerRG2RGB);
                    break;
                case PixelFormat::BGRA:
                    cv::cvtColor(image, image, cv::COLOR_BayerRG2BGRA);
                    break;
                case PixelFormat::RGBA:
                    cv::cvtColor(image, image, cv::COLOR_BayerRG2RGBA);
                    break;
                case PixelFormat::HSV: {
                    cv::cvtColor(image, image, cv::COLOR_BayerRG2BGR);
                    cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
                    break;
                }
                case PixelFormat::YUV: {
                    cv::cvtColor(image, image, cv::COLOR_BayerRG2BGR);
                    cv::cvtColor(image, image, cv::COLOR_BGR2YUV);
                    break;
                }
                default:
                    assert(0 && "unreachable code");
            }
            break;

        case PixelType::YUV422_8:
            switch (format) {
                case PixelFormat::GRAY:
                    cv::cvtColor(image, image, cv::COLOR_YUV2GRAY_YUYV);
                    break;
                case PixelFormat::BGR:
                    cv::cvtColor(image, image, cv::COLOR_YUV2BGR_YUYV);
                    break;
                case PixelFormat::RGB:
                    cv::cvtColor(image, image, cv::COLOR_YUV2RGB_YUYV);
                    break;
                case PixelFormat::BGRA:
                    cv::cvtColor(image, image, cv::COLOR_YUV2BGRA_YUYV);
                    break;
                case PixelFormat::RGBA:
                    cv::cvtColor(image, image, cv::COLOR_YUV2RGBA_YUYV);
                    break;
                case PixelFormat::HSV: {
                    cv::cvtColor(image, image, cv::COLOR_YUV2BGR_YUYV);
                    cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
                    break;
                }
                case PixelFormat::YUV: {
                    cv::cvtColor(image, image, cv::COLOR_YUV2BGR_YUYV);
                    cv::cvtColor(image, image, cv::COLOR_BGR2YUV);
                    break;
                }
                default:
                    assert(0 && "unreachable code");
            }
            break;

        case PixelType::YUV422_8_UYVY:
            switch (format) {
                case PixelFormat::GRAY:
                    cv::cvtColor(image, image, cv::COLOR_YUV2GRAY_UYVY);
                    break;
                case PixelFormat::BGR:
                    cv::cvtColor(image, image, cv::COLOR_YUV2BGR_UYVY);
                    break;
                case PixelFormat::RGB:
                    cv::cvtColor(image, image, cv::COLOR_YUV2RGB_UYVY);
                    break;
                case PixelFormat::BGRA:
                    cv::cvtColor(image, image, cv::COLOR_YUV2BGRA_UYVY);
                    break;
                case PixelFormat::RGBA:
                    cv::cvtColor(image, image, cv::COLOR_YUV2RGBA_UYVY);
                    break;
                case PixelFormat::HSV: {
                    cv::cvtColor(image, image, cv::COLOR_YUV2BGR_UYVY);
                    cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
                    break;
                }
                case PixelFormat::YUV: {
                    cv::cvtColor(image, image, cv::COLOR_YUV2BGR_UYVY);
                    cv::cvtColor(image, image, cv::COLOR_BGR2YUV);
                    break;
                }
                default:
                    assert(0 && "unreachable code");
            }
            break;

        default:
            assert(0 && "unreachable code");
    }
}

}  // namespace radar::camera