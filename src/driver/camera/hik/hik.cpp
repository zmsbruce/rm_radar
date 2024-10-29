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

#include <magic_enum/magic_enum.hpp>
#include <stdexcept>
#include <vector>

#define HIK_CHECK_RETURN_BOOL(func, msg, ...)                     \
    do {                                                          \
        int ret = func(__VA_ARGS__);                              \
        if (MV_OK != ret) {                                       \
            spdlog::error("Failed to {}, error code: {:#x}", msg, \
                          static_cast<unsigned int>(ret));        \
            return false;                                         \
        }                                                         \
    } while (0)

#define HIK_CHECK_NORETURN(func, msg, ...)                        \
    do {                                                          \
        int ret = func(__VA_ARGS__);                              \
        if (MV_OK != ret) {                                       \
            spdlog::error("Failed to {}, error code: {:#x}", msg, \
                          static_cast<unsigned int>(ret));        \
        }                                                         \
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
 * @param pixel_format Camera pixel format.
 * @param grab_timeout Timeout for grabbing an image.
 * @param auto_white_balance Whether auto white balance is enabled.
 * @param balance_ratio White balance ratio for RGB channels.
 */
HikCamera::HikCamera(std::string_view camera_sn, unsigned int width,
                     unsigned int height, float exposure, float gamma,
                     float gain, std::string_view pixel_format,
                     unsigned int grab_timeout, bool auto_white_balance,
                     std::array<unsigned int, 3>&& balance_ratio)
    : camera_sn_{camera_sn},
      width_{width},
      height_{height},
      exposure_{exposure},
      gamma_{gamma},
      gain_{gain},
      pixel_format_{magic_enum::enum_cast<HikPixelFormat>(pixel_format)
                        .value_or(HikPixelFormat::Unknown)},
      grab_timeout_{grab_timeout},
      auto_white_balance_{auto_white_balance},
      balance_ratio_{balance_ratio},
      frame_out_{std::make_unique<MV_FRAME_OUT>()},
      device_info_{std::make_unique<MV_CC_DEVICE_INFO>()},
      supported_pixel_formats_{std::make_unique<MVCC_ENUMVALUE>()} {
    spdlog::trace("Initializing HikCamera with SN: {}", camera_sn);
    spdlog::debug(
        "Camera parameters: width={}, height={}, exposure={}, gamma={}, "
        "gain={}, pixel_format: {} grab timeout: {}, auto white balance: {}, "
        "balance ratio: "
        "[{}, {}, {}]",
        width, height, exposure, gamma, gain, pixel_format, grab_timeout,
        auto_white_balance, balance_ratio[0], balance_ratio[1],
        balance_ratio[2]);

    // Initializing device info
    std::memset(device_info_.get(), 0, sizeof(MV_CC_DEVICE_INFO));
    spdlog::trace("Device info structure initialized.");

    // Initializing supported pixel formats
    std::memset(supported_pixel_formats_.get(), 0, sizeof(MVCC_ENUMVALUE));
    spdlog::trace("Supported pixel formats structure initialized.");

    // Starting daemon thread
    spdlog::trace("Starting daemon thread.");
    startDaemonThread();
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
        close();
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
    if (isOpen()) {
        spdlog::warn(
            "Camera with serial number: {} is not closed. Attempting to close "
            "it before open.",
            camera_sn_);
        close();
    }
    spdlog::info("Opening camera with serial number: {}", camera_sn_);

    // Retrieve available device information
    spdlog::trace("Retrieving device information list.");
    auto device_info_list = HikCamera::getDeviceInfoList();
    spdlog::debug("Found {} devices.", device_info_list.size());

    // Acquire lock for thread-safety when opening the camera
    std::unique_lock lock(mutex_);
    spdlog::debug("Acquired lock for opening camera {}", camera_sn_);

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
    HIK_CHECK_RETURN_BOOL(MV_CC_CreateHandle, "create handle", &handle_,
                          *device_info_iter);
    spdlog::debug("Handle created successfully for camera {}.", camera_sn_);

    // Open the camera device
    spdlog::trace("Opening device for camera {}.", camera_sn_);
    HIK_CHECK_RETURN_BOOL(MV_CC_OpenDevice, "open device", handle_);
    spdlog::info("Device for camera {} opened successfully.", camera_sn_);

    // Set the pixel type based on supported formats
    spdlog::trace("Setting pixel type for camera {}.", camera_sn_);
    if (!setPixelFormat(getSupportedPixelFormats())) {
        spdlog::critical("Failed to set pixel format for camera {}.",
                         camera_sn_);
        return false;
    }
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
    HIK_CHECK_RETURN_BOOL(MV_CC_RegisterExceptionCallBack,
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
void HikCamera::close() {
    spdlog::info("Closing camera with serial number: {}", camera_sn_);

    // Check if the camera is currently capturing images
    spdlog::trace("Checking if camera {} is capturing.", camera_sn_);
    if (isCapturing()) {
        spdlog::debug("Camera {} is capturing, attempting to stop capture.",
                      camera_sn_);
        if (stopCapture()) {
            spdlog::info("Capture stopped successfully for camera {}.",
                         camera_sn_);
        } else {
            spdlog::error("Failed to stop capture for camera {}.", camera_sn_);
        }
    }

    // Acquire lock before modifying camera state
    spdlog::trace("Acquiring lock to close camera {}.", camera_sn_);
    {
        std::unique_lock lock(mutex_);
        spdlog::debug("Lock acquired for closing camera {}.", camera_sn_);

        // Close the camera device
        spdlog::trace("Closing device for camera {}.", camera_sn_);
        HIK_CHECK_NORETURN(MV_CC_CloseDevice, "close device", handle_);

        // Destroy the camera handle
        spdlog::trace("Destroying handle for camera {}.", camera_sn_);
        HIK_CHECK_NORETURN(MV_CC_DestroyHandle, "destroy handle", handle_);

        // Reset handle and mark as closed
        handle_ = nullptr;
        is_open_ = false;
        spdlog::debug("Camera {} handle reset and marked as closed.",
                      camera_sn_);
    }

    spdlog::info("Camera {} closed.", camera_sn_);
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
    close();

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
        HIK_CHECK_RETURN_BOOL(MV_CC_StartGrabbing, "start grabbing", handle_);
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
        HIK_CHECK_RETURN_BOOL(MV_CC_StopGrabbing, "stop grabbing", handle_);
        spdlog::debug("MV_CC_StopGrabbing called successfully for camera {}.",
                      camera_sn_);

        // Update the camera's capturing status
        is_capturing_ = false;
        spdlog::debug("Camera {} marked as not capturing.", camera_sn_);
    }

    spdlog::debug("Capture stopped successfully on camera {}.", camera_sn_);
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

    std::unique_lock lock(mutex_);
    spdlog::debug("Lock acquired for grabbing image from camera {}.",
                  camera_sn_);

    // Clear the frame output buffer before grabbing a new image
    spdlog::trace("Clearing frame output buffer for camera {}.", camera_sn_);
    std::memset(frame_out_.get(), 0, sizeof(MV_FRAME_OUT));

    // Attempt to get the image buffer from the camera
    spdlog::trace("Calling MV_CC_GetImageBuffer for camera {}.", camera_sn_);
    HIK_CHECK_RETURN_BOOL(MV_CC_GetImageBuffer, "get image buffer", handle_,
                          frame_out_.get(), grab_timeout_);
    spdlog::debug("Image buffer retrieved successfully for camera {}.",
                  camera_sn_);

    // Convert the buffer to a cv::Mat object
    spdlog::trace("Converting frame buffer to cv::Mat for camera {}.",
                  camera_sn_);
    cv::Mat temp_image =
        cv::Mat(frame_out_->stFrameInfo.nHeight, frame_out_->stFrameInfo.nWidth,
                CV_8UC3, frame_out_->pBufAddr);

    // Check if the image is valid (not empty)
    if (temp_image.empty()) {
        spdlog::error("Failed to grab image from camera {}: image is empty.",
                      camera_sn_);
        return false;
    }
    spdlog::debug("Image successfully converted to cv::Mat for camera {}.",
                  camera_sn_);

    // Convert the pixel format as needed
    spdlog::trace("Converting pixel format for camera {}.", camera_sn_);
    image = convertPixelFormat(temp_image, pixel_format);
    if (image.empty()) {
        spdlog::error("Failed to convert image from camera {}: image is empty.",
                      camera_sn_);
        return false;
    }
    spdlog::debug("Pixel format conversion completed for camera {}.",
                  camera_sn_);

    // Free the frame output buffer after image convertion completed
    spdlog::trace("Calling MV_CC_FreeImageBuffer for camera {}.", camera_sn_);
    HIK_CHECK_RETURN_BOOL(MV_CC_FreeImageBuffer, "free image buffer", handle_,
                          frame_out_.get());
    spdlog::debug("Image buffer released successfully for camera {}.",
                  camera_sn_);

    spdlog::trace("Image grabbed successfully from camera {}.", camera_sn_);
    return true;
}

/**
 * @brief Gets all pixel formats supported for the camera.
 * @return Supported pixel formats.
 */
std::span<unsigned int> HikCamera::getSupportedPixelFormats() {
    spdlog::trace("Getting supported pixel formats for camera {}.", camera_sn_);
    int ret = MV_CC_GetPixelFormat(handle_, supported_pixel_formats_.get());
    if (ret != MV_OK) {
        spdlog::error("Failed to get pixel format, error code: {}");
        return {};
    }
    auto supported_pixel_format =
        std::span(supported_pixel_formats_->nSupportValue,
                  supported_pixel_formats_->nSupportedNum);
    spdlog::debug("Supported pixel format for camera {}: [{:#x}]", camera_sn_,
                  fmt::join(supported_pixel_format, ", "));
    return supported_pixel_format;
}

/**
 * @brief Sets the pixel format for the camera.
 * @param supported_formats Span of supported pixel formats.
 * @return True if the pixel format was set successfully, false otherwise.
 */
bool HikCamera::setPixelFormat(std::span<unsigned int> supported_formats) {
    spdlog::debug("Attempting to set pixel format for camera {}.", camera_sn_);

    static const std::vector<HikPixelFormat> candidate_formats{
        HikPixelFormat::RGB8Packed,    HikPixelFormat::BayerBG8,
        HikPixelFormat::BayerGB8,      HikPixelFormat::BayerGR8,
        HikPixelFormat::BayerRG8,      HikPixelFormat::YUV422_8,
        HikPixelFormat::YUV422_8_UYVY,
    };

    spdlog::trace("Supported pixel formats for camera {}: [{:#x}]", camera_sn_,
                  fmt::join(supported_formats, ", "));

    if (std::ranges::any_of(
            supported_formats, [this](unsigned int supported_type) {
                return getPixelFormatValue(pixel_format_) == supported_type;
            })) {
        return true;
    } else if (pixel_format_ != HikPixelFormat::Unknown) {
        spdlog::warn(
            "Current pixel format is {} which is not supported, will choose "
            "another.",
            magic_enum::enum_name(pixel_format_));
    }

    // Iterate over candidate formats and check if any are supported
    for (auto type : candidate_formats) {
        spdlog::trace("Checking if pixel format {}: {:#x} is supported.",
                      magic_enum::enum_name(type), getPixelFormatValue(type));

        if (std::ranges::any_of(
                supported_formats, [type](unsigned int supported_type) {
                    return getPixelFormatValue(type) == supported_type;
                })) {
            pixel_format_ = type;
            spdlog::debug("Pixel format {} selected for camera {}.",
                          magic_enum::enum_name(pixel_format_), camera_sn_);
            break;
        }
    }

    // If no supported pixel type was found, log a critical error
    if (pixel_format_ == HikPixelFormat::Unknown) {
        spdlog::error(
            "Failed to set pixel format for camera {}: no supported format "
            "found.",
            camera_sn_);
        return false;
    }

    // Set the pixel format using the selected pixel type
    spdlog::trace("Setting pixel format to {} for camera {}.",
                  magic_enum::enum_name(pixel_format_), camera_sn_);
    HIK_CHECK_RETURN_BOOL(MV_CC_SetPixelFormat, "set pixel format", handle_,
                          getPixelFormatValue(pixel_format_));

    spdlog::info("Pixel format set successfully to {} for camera {}.",
                 magic_enum::enum_name(pixel_format_), camera_sn_);
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
    if (isCapturing()) {
        spdlog::error("Setting resolution is not supported when capturing.");
        return false;
    }
    std::unique_lock lock(mutex_);
    width_ = width;
    height_ = height;
    return setResolutionInner();
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
    return setGainInner();
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
    return setExposureInner();
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
    return setBalanceRatioInner();
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
    return setBalanceRatioInner();
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
    MVCC_INTVALUE width;
    MVCC_INTVALUE height;
    std::memset(&width, 0, sizeof(MVCC_INTVALUE));
    std::memset(&height, 0, sizeof(MVCC_INTVALUE));

    HIK_CHECK_NORETURN(MV_CC_GetWidth, "get width", handle_, &width);
    HIK_CHECK_NORETURN(MV_CC_GetHeight, "get width", handle_, &height);
    if (width.nMax != 0 || height.nMax != 0) {
        if (width_ < width.nMin || width_ > width.nMax ||
            height_ < height.nMin || height_ > height.nMax) {
            spdlog::error(
                "Invalid resolution {}x{}. Required width: {}~{}, height: "
                "{}~{}.",
                width_, height_, width.nMin, width.nMax, height.nMin,
                height.nMax);
            return false;
        }
        if (width_ % width.nInc != 0) {
            auto width_old = width_;
            width_ = (width_ / width.nInc + 1) * width.nInc;
            if (width_ > width.nMax) {
                width_ -= width.nInc;
            }
            spdlog::warn("Width should be times of {}, fixed from {} to {}.",
                         width.nInc, width_old, width_);
        }
        if (height_ % height.nInc != 0) {
            auto height_old = height_;
            height_ = (height_ / height.nInc + 1) * height.nInc;
            if (height_ > height.nMax) {
                height_ -= height.nInc;
            }
            spdlog::warn("Height should be times of {}, fixed from {} to {}.",
                         height.nInc, height_old, height_);
        }
    }

    // TODO: Check whether width changes when height changes (and vice versa)
    HIK_CHECK_RETURN_BOOL(MV_CC_SetWidth, "set width", handle_, width_);
    HIK_CHECK_RETURN_BOOL(MV_CC_SetHeight, "set height", handle_, height_);
    return true;
}

/**
 * @brief Sets the white balance ratio internally.
 * @return True if the white balance ratio was set successfully, false
 * otherwise.
 */
bool HikCamera::setBalanceRatioInner() {
    if (auto_white_balance_) {
        HIK_CHECK_RETURN_BOOL(
            MV_CC_SetBalanceWhiteAuto, "set balance white auto", handle_,
            static_cast<unsigned int>(HikBalanceWhiteAuto::Continuous));
    } else {
        HIK_CHECK_RETURN_BOOL(
            MV_CC_SetBalanceWhiteAuto, "set balance white auto", handle_,
            static_cast<unsigned int>(HikBalanceWhiteAuto::Off));
        HIK_CHECK_RETURN_BOOL(MV_CC_SetBalanceRatioRed, "set balance ratio red",
                              handle_, balance_ratio_[0]);
        HIK_CHECK_RETURN_BOOL(MV_CC_SetBalanceRatioGreen,
                              "set balance ratio green", handle_,
                              balance_ratio_[1]);
        HIK_CHECK_RETURN_BOOL(MV_CC_SetBalanceRatioBlue,
                              "set balance ratio blue", handle_,
                              balance_ratio_[2]);
    }
    return true;
}

/**
 * @brief Sets the exposure value internally.
 * @return True if the exposure value was set successfully, false otherwise.
 */
bool HikCamera::setExposureInner() {
    if (exposure_ > 0) {
        HIK_CHECK_RETURN_BOOL(MV_CC_SetExposureAutoMode, "set exposure auto",
                              handle_,
                              static_cast<unsigned int>(HikExposureAuto::Off));
        HIK_CHECK_RETURN_BOOL(MV_CC_SetExposureTime, "set exposure time",
                              handle_, exposure_);
    } else {
        HIK_CHECK_RETURN_BOOL(
            MV_CC_SetExposureAutoMode, "set exposure auto", handle_,
            static_cast<unsigned int>(HikExposureAuto::Continuous));
    }
    return true;
}

/**
 * @brief Sets the gamma correction value internally.
 * @return True if the gamma value was set successfully, false otherwise.
 */
bool HikCamera::setGammaInner() {
    if (gamma_ > 0.0f) {
        HIK_CHECK_RETURN_BOOL(MV_CC_SetBoolValue, "set gamma enable", handle_,
                              "GammaEnable", true);
        HIK_CHECK_RETURN_BOOL(
            MV_CC_SetGammaSelector, "set gamma selector", handle_,
            static_cast<unsigned int>(HikGammaSelector::User));
        HIK_CHECK_RETURN_BOOL(MV_CC_SetGamma, "set gamma", handle_, gamma_);
    }
    return true;
}

/**
 * @brief Sets the gain value internally.
 * @return True if the gain value was set successfully, false otherwise.
 */
bool HikCamera::setGainInner() {
    if (gain_ > 0) {
        HIK_CHECK_RETURN_BOOL(MV_CC_SetGainMode, "set gain auto", handle_,
                              static_cast<unsigned int>(HikGainAuto::Off));
        HIK_CHECK_RETURN_BOOL(MV_CC_SetGain, "set gain", handle_, gain_);
    } else {
        HIK_CHECK_RETURN_BOOL(
            MV_CC_SetGainMode, "set gain auto", handle_,
            static_cast<unsigned int>(HikGainAuto::Continuous));
    }
    return true;
}

/**
 * @brief Sets the exception flag for the camera.
 * @param occurred New value for the exception flag.
 */
void HikCamera::setExceptionOccurred(bool occurred) {
    exception_occurred_ = occurred;
}

/**
 * @brief Gets the current value of the exception flag.
 * @return True if an exception has occurred, false otherwise.
 */
bool HikCamera::isExceptionOccurred() const { return exception_occurred_; }

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
    camera->setExceptionOccurred(true);
}

/**
 * @brief Starts a daemon thread to monitor camera exceptions.
 */
void HikCamera::startDaemonThread() {
    daemon_thread_ = std::jthread([this](std::stop_token token) {
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (isExceptionOccurred()) {
                spdlog::info(
                    "Exception detected in HikCamera {} daemon thread. Try "
                    "reconnecting...",
                    camera_sn_);
                if (reconnect()) {
                    spdlog::info("Successfully reconnected HikCamera {}.",
                                 camera_sn_);
                    setExceptionOccurred(false);
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
    std::call_once(is_device_info_list_init_, [] {
        device_info_list_ = std::make_shared<MV_CC_DEVICE_INFO_LIST>();
        std::memset(device_info_list_.get(), 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        spdlog::trace("Initialized MV_CC_DEVICE_INFO_LIST structure");

        HIK_CHECK_NORETURN(MV_CC_EnumDevices, "enum devices", MV_USB_DEVICE,
                           device_info_list_.get());
        spdlog::info("Enumerated devices, found {} devices.",
                     device_info_list_->nDeviceNum);
        for (unsigned int i = 0; i < device_info_list_->nDeviceNum; ++i) {
            auto device_info = device_info_list_->pDeviceInfo[i];
            spdlog::info("Device {}: {}", i,
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
 * @return The image converted.
 */
cv::Mat HikCamera::convertPixelFormat(const cv::Mat& input_image,
                                      PixelFormat format) {
    spdlog::debug("Starting pixel format conversion from {} to {}",
                  magic_enum::enum_name(pixel_format_),
                  magic_enum::enum_name(format));
    cv::Mat output_image;
    switch (pixel_format_) {
        case HikPixelFormat::RGB8Packed:
            spdlog::trace("Handling HikPixelFormat::RGB8Packed");
            switch (format) {
                case PixelFormat::GRAY:
                    spdlog::debug("Converting from RGB to GRAY");
                    cv::cvtColor(input_image, output_image, cv::COLOR_RGB2GRAY);
                    break;
                case PixelFormat::RGB:
                    spdlog::debug(
                        "No conversion needed for RGB, will clone image");
                    output_image = input_image.clone();
                    break;
                case PixelFormat::BGR:
                    spdlog::debug("Converting from RGB to BGR");
                    cv::cvtColor(input_image, output_image, cv::COLOR_RGB2BGR);
                    break;
                case PixelFormat::RGBA:
                    spdlog::debug("Converting from RGB to RGBA");
                    cv::cvtColor(input_image, output_image, cv::COLOR_RGB2RGBA);
                    break;
                case PixelFormat::BGRA:
                    spdlog::debug("Converting from RGB to BGRA");
                    cv::cvtColor(input_image, output_image, cv::COLOR_RGB2BGRA);
                    break;
                case PixelFormat::HSV:
                    spdlog::debug("Converting from RGB to HSV");
                    cv::cvtColor(input_image, output_image, cv::COLOR_RGB2HSV);
                    break;
                case PixelFormat::YUV:
                    spdlog::debug("Converting from RGB to YUV");
                    cv::cvtColor(input_image, output_image, cv::COLOR_RGB2YUV);
                    break;
                default:
                    spdlog::error("Invalid target format for RGB8Packed");
                    assert(0 && "unreachable code");
            }
            break;

        case HikPixelFormat::BayerBG8:
            spdlog::trace("Handling HikPixelFormat::BayerBG8");
            switch (format) {
                case PixelFormat::GRAY:
                    spdlog::debug("Converting from BayerBG to GRAY");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerBG2GRAY);
                    break;
                case PixelFormat::BGR:
                    spdlog::debug("Converting from BayerBG to BGR");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerBG2BGR);
                    break;
                case PixelFormat::RGB:
                    spdlog::debug("Converting from BayerBG to RGB");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerBG2RGB);
                    break;
                case PixelFormat::BGRA:
                    spdlog::debug("Converting from BayerBG to BGRA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerBG2BGRA);
                    break;
                case PixelFormat::RGBA:
                    spdlog::debug("Converting from BayerBG to RGBA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerBG2RGBA);
                    break;
                case PixelFormat::HSV:
                    spdlog::debug("Converting from BayerBG to HSV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerBG2BGR);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2HSV);
                    break;
                case PixelFormat::YUV:
                    spdlog::debug("Converting from BayerBG to YUV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerBG2BGR);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2YUV);
                    break;
                default:
                    spdlog::error("Invalid target format for BayerBG8");
                    assert(0 && "unreachable code");
            }
            break;

        case HikPixelFormat::BayerGB8:
            spdlog::trace("Handling HikPixelFormat::BayerGB8");
            switch (format) {
                case PixelFormat::GRAY:
                    spdlog::debug("Converting from BayerGB to GRAY");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGB2GRAY);
                    break;
                case PixelFormat::BGR:
                    spdlog::debug("Converting from BayerGB to BGR");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGB2BGR);
                    break;
                case PixelFormat::RGB:
                    spdlog::debug("Converting from BayerGB to RGB");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGB2RGB);
                    break;
                case PixelFormat::BGRA:
                    spdlog::debug("Converting from BayerGB to BGRA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGB2BGRA);
                    break;
                case PixelFormat::RGBA:
                    spdlog::debug("Converting from BayerGB to RGBA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGB2RGBA);
                    break;
                case PixelFormat::HSV:
                    spdlog::debug("Converting from BayerGB to HSV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGB2BGR);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2HSV);
                    break;
                case PixelFormat::YUV:
                    spdlog::debug("Converting from BayerGB to YUV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGB2BGR);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2YUV);
                    break;
                default:
                    spdlog::error("Invalid target format for BayerGB8");
                    assert(0 && "unreachable code");
            }
            break;

        case HikPixelFormat::BayerGR8:
            spdlog::trace("Handling HikPixelFormat::BayerGR8");
            switch (format) {
                case PixelFormat::GRAY:
                    spdlog::debug("Converting from BayerGR to GRAY");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGR2GRAY);
                    break;
                case PixelFormat::BGR:
                    spdlog::debug("Converting from BayerGR to BGR");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGR2BGR);
                    break;
                case PixelFormat::RGB:
                    spdlog::debug("Converting from BayerGR to RGB");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGR2RGB);
                    break;
                case PixelFormat::BGRA:
                    spdlog::debug("Converting from BayerGR to BGRA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGR2BGRA);
                    break;
                case PixelFormat::RGBA:
                    spdlog::debug("Converting from BayerGR to RGBA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGR2RGBA);
                    break;
                case PixelFormat::HSV:
                    spdlog::debug("Converting from BayerGR to HSV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGR2BGR);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2HSV);
                    break;
                case PixelFormat::YUV:
                    spdlog::debug("Converting from BayerGR to YUV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerGR2BGR);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2YUV);
                    break;
                default:
                    spdlog::error("Invalid target format for BayerGR8");
                    assert(0 && "unreachable code");
            }
            break;

        case HikPixelFormat::BayerRG8:
            spdlog::trace("Handling HikPixelFormat::BayerRG8");
            switch (format) {
                case PixelFormat::GRAY:
                    spdlog::debug("Converting from BayerRG to GRAY");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerRG2GRAY);
                    break;
                case PixelFormat::BGR:
                    spdlog::debug("Converting from BayerRG to BGR");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerRG2BGR);
                    break;
                case PixelFormat::RGB:
                    spdlog::debug("Converting from BayerRG to RGB");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerRG2RGB);
                    break;
                case PixelFormat::BGRA:
                    spdlog::debug("Converting from BayerRG to BGRA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerRG2BGRA);
                    break;
                case PixelFormat::RGBA:
                    spdlog::debug("Converting from BayerRG to RGBA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerRG2RGBA);
                    break;
                case PixelFormat::HSV:
                    spdlog::debug("Converting from BayerRG to HSV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerRG2BGR);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2HSV);
                    break;
                case PixelFormat::YUV:
                    spdlog::debug("Converting from BayerRG to YUV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_BayerRG2BGR);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2YUV);
                    break;
                default:
                    spdlog::error("Invalid target format for BayerRG8");
                    assert(0 && "unreachable code");
            }
            break;

        case HikPixelFormat::YUV422_8:
            spdlog::trace("Handling HikPixelFormat::YUV422_8");
            switch (format) {
                case PixelFormat::GRAY:
                    spdlog::debug("Converting from YUV422 to GRAY");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2GRAY_YUYV);
                    break;
                case PixelFormat::BGR:
                    spdlog::debug("Converting from YUV422 to BGR");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2BGR_YUYV);
                    break;
                case PixelFormat::RGB:
                    spdlog::debug("Converting from YUV422 to RGB");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2RGB_YUYV);
                    break;
                case PixelFormat::BGRA:
                    spdlog::debug("Converting from YUV422 to BGRA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2BGRA_YUYV);
                    break;
                case PixelFormat::RGBA:
                    spdlog::debug("Converting from YUV422 to RGBA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2RGBA_YUYV);
                    break;
                case PixelFormat::HSV:
                    spdlog::debug("Converting from YUV422 to HSV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2BGR_YUYV);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2HSV);
                    break;
                case PixelFormat::YUV:
                    spdlog::debug("Converting from YUV422 to YUV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2BGR_YUYV);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2YUV);
                    break;
                default:
                    spdlog::error("Invalid target format for YUV422_8");
                    assert(0 && "unreachable code");
            }
            break;

        case HikPixelFormat::YUV422_8_UYVY:
            spdlog::trace("Handling HikPixelFormat::YUV422_8_UYVY");
            switch (format) {
                case PixelFormat::GRAY:
                    spdlog::debug("Converting from YUV422_UYVY to GRAY");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2GRAY_UYVY);
                    break;
                case PixelFormat::BGR:
                    spdlog::debug("Converting from YUV422_UYVY to BGR");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2BGR_UYVY);
                    break;
                case PixelFormat::RGB:
                    spdlog::debug("Converting from YUV422_UYVY to RGB");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2RGB_UYVY);
                    break;
                case PixelFormat::BGRA:
                    spdlog::debug("Converting from YUV422_UYVY to BGRA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2BGRA_UYVY);
                    break;
                case PixelFormat::RGBA:
                    spdlog::debug("Converting from YUV422_UYVY to RGBA");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2RGBA_UYVY);
                    break;
                case PixelFormat::HSV:
                    spdlog::debug("Converting from YUV422_UYVY to HSV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2BGR_UYVY);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2HSV);
                    break;
                case PixelFormat::YUV:
                    spdlog::debug("Converting from YUV422_UYVY to YUV");
                    cv::cvtColor(input_image, output_image,
                                 cv::COLOR_YUV2BGR_UYVY);
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2YUV);
                    break;
                default:
                    spdlog::error("Invalid target format for YUV422_8_UYVY");
                    assert(0 && "unreachable code");
            }
            break;

        default:
            spdlog::error(
                "Unknown pixel format encountered in convertPixelFormat");
    }

    spdlog::debug("Pixel format conversion completed");
    return output_image;
}

unsigned int HikCamera::getPixelFormatValue(HikPixelFormat format) {
    switch (format) {
        case HikPixelFormat::RGB8Packed:
            return 0x02180014;
        case HikPixelFormat::YUV422_8:
            return 0x02100032;
        case HikPixelFormat::YUV422_8_UYVY:
            return 0x0210001F;
        case HikPixelFormat::BayerGR8:
            return 0x01080008;
        case HikPixelFormat::BayerRG8:
            return 0x01080009;
        case HikPixelFormat::BayerGB8:
            return 0x0108000A;
        case HikPixelFormat::BayerBG8:
            return 0x0108000B;
        default:
            return 0x0;
    }
}

}  // namespace radar::camera