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

namespace radar::camera {

HikCamera::HikCamera(
    std::string_view camera_sn,
    std::optional<std::pair<unsigned int, unsigned int>> resolution,
    std::optional<float> exposure, std::optional<float> gamma,
    std::optional<float> gain,
    std::optional<std::array<unsigned int, 3>> balance_ratio,
    std::optional<std::string_view> pixel_format, unsigned int grab_timeout)
    : camera_sn_(camera_sn),
      resolution_(resolution),
      exposure_(exposure),
      gamma_(gamma),
      gain_(gain),
      pixel_format_(
          magic_enum::enum_cast<HikPixelFormat>(pixel_format.value_or(""))
              .value_or(HikPixelFormat::Unknown)),
      grab_timeout_(grab_timeout),
      balance_ratio_(balance_ratio),
      frame_out_{std::make_unique<MV_FRAME_OUT>()},
      device_info_{std::make_unique<MV_CC_DEVICE_INFO>()},
      supported_pixel_formats_{std::make_unique<MVCC_ENUMVALUE>()} {
    spdlog::trace("Initializing HikCamera with SN: {}", camera_sn);

    if (spdlog::get_level() <= spdlog::level::debug) {
        std::string resolution_str =
            resolution
                ? fmt::format("{}x{}", resolution->first, resolution->second)
                : "null";

        std::string exposure_str =
            exposure ? fmt::format("{}", *exposure) : "null";
        std::string gamma_str = gamma ? fmt::format("{}", *gamma) : "null";
        std::string gain_str = gain ? fmt::format("{}", *gain) : "null";

        std::string balance_ratio_str =
            balance_ratio
                ? fmt::format("[{}, {}, {}]", (*balance_ratio)[0],
                              (*balance_ratio)[1], (*balance_ratio)[2])
                : "null";

        spdlog::debug(
            "Camera parameters: resolution: {}, exposure: {}, gamma: {}, gain: "
            "{}, "
            "pixel_format: {}, grab timeout: {}, balance ratio: {}",
            resolution_str, exposure_str, gamma_str, gain_str,
            magic_enum::enum_name(pixel_format_), grab_timeout,
            balance_ratio_str);
    }

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

HikCamera::~HikCamera() {
    spdlog::trace("Destroying camera with serial number: {}", camera_sn_);

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

bool HikCamera::open() {
    if (isOpen()) {
        spdlog::warn(
            "Camera with serial number: {} is not closed. Attempting to close "
            "it before open.",
            camera_sn_);
        close();
    }
    spdlog::trace("Opening camera with serial number: {}", camera_sn_);

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
    spdlog::trace("Device with serial number {} found.", camera_sn_);

    // Create a handle for the camera device
    spdlog::trace("Creating handle for the camera.");
    HIK_CHECK_RETURN_BOOL(MV_CC_CreateHandle, "create handle", &handle_,
                          *device_info_iter);
    spdlog::debug("Handle created successfully for camera {}.", camera_sn_);

    // Open the camera device
    spdlog::trace("Opening device for camera {}.", camera_sn_);
    HIK_CHECK_RETURN_BOOL(MV_CC_OpenDevice, "open device", handle_);
    spdlog::debug("Device for camera {} opened successfully.", camera_sn_);

    // Set the pixel type based on supported formats
    spdlog::trace("Setting pixel type for camera {}.", camera_sn_);
    if (!setPixelFormatInner()) {
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
    spdlog::trace("Exception callback registered successfully for camera {}.",
                  camera_sn_);

    // Mark the camera as open
    is_open_ = true;
    spdlog::info("Camera {} opened.", camera_sn_);
    return true;
}

void HikCamera::close() {
    spdlog::trace("Closing camera with serial number: {}", camera_sn_);

    // Check if the camera is currently capturing images
    spdlog::trace("Checking if camera {} is capturing.", camera_sn_);
    if (isCapturing()) {
        spdlog::debug("Camera {} is capturing, attempting to stop capture.",
                      camera_sn_);
        if (!stopCapture()) {
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
        spdlog::info("Camera {} reconnected.", camera_sn_);
        return true;
    }
}

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

    spdlog::info("Camera {} started capturing.", camera_sn_);
    return true;
}

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

    spdlog::info("Camera {} stopped capturing.", camera_sn_);
    return true;
}

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

    int type;
    switch (pixel_format_) {
        case HikPixelFormat::BayerBG8:
        case HikPixelFormat::BayerGB8:
        case HikPixelFormat::BayerGR8:
        case HikPixelFormat::BayerRG8:
            type = CV_8UC1;
            break;
        default:
            type = CV_8UC3;
            break;
    }
    cv::Mat temp_image =
        cv::Mat(frame_out_->stFrameInfo.nHeight, frame_out_->stFrameInfo.nWidth,
                type, frame_out_->pBufAddr);

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
    // spdlog::debug("Supported pixel format for camera {}: [{:#x}]",
    // camera_sn_,
    //               fmt::join(supported_pixel_format, ", "));
    return supported_pixel_format;
}

bool HikCamera::setPixelFormat(HikPixelFormat pixel_format) {
    spdlog::trace("Entering setPixelFormat() for camera {}", camera_sn_);
    if (isOpen()) {
        spdlog::error(
            "Setting pixel format is not supported when camera is not open.");
        return false;
    }
    if (isCapturing()) {
        spdlog::error("Setting pixel format is not supported when capturing.");
        return false;
    }

    std::unique_lock lock(mutex_);
    pixel_format_ = pixel_format;
    return setPixelFormatInner();
}

bool HikCamera::setPixelFormatInner() {
    spdlog::debug("Attempting to set pixel format for camera {}.", camera_sn_);

    static const std::vector<HikPixelFormat> candidate_formats{
        HikPixelFormat::BayerBG8, HikPixelFormat::BayerGB8,
        HikPixelFormat::BayerGR8, HikPixelFormat::BayerRG8,
        HikPixelFormat::BGR8,     HikPixelFormat::RGB8,
        HikPixelFormat::YUV422_8, HikPixelFormat::YUV422_8_UYVY,
    };

    const auto supported_formats = getSupportedPixelFormats();

    // spdlog::trace("Supported pixel formats for camera {}: [{:#x}]",
    // camera_sn_,
    //               fmt::join(supported_formats, ", "));

    if (!std::ranges::any_of(supported_formats, [this](unsigned int format) {
            return getPixelFormatValue(pixel_format_) == format;
        })) {
        if (pixel_format_ != HikPixelFormat::Unknown) {
            spdlog::warn(
                "Current pixel format is {} which is not supported, will "
                "choose another.",
                magic_enum::enum_name(pixel_format_));
        }

        // Iterate over candidate formats and check if any are supported
        for (auto type : candidate_formats) {
            spdlog::trace(
                "Checking if pixel format {}: {:#x} is supported for camera "
                "{}.",
                magic_enum::enum_name(type), getPixelFormatValue(type),
                camera_sn_);

            if (std::ranges::any_of(
                    supported_formats, [type](unsigned int supported_type) {
                        return getPixelFormatValue(type) == supported_type;
                    })) {
                pixel_format_ = type;
                spdlog::debug("Pixel format {} selected for camera {}.",
                              magic_enum::enum_name(pixel_format_), camera_sn_);
                break;
            } else {
                spdlog::trace("{} is not supported for camera {}",
                              magic_enum::enum_name(type), camera_sn_);
            }
        }
    } else {
        spdlog::trace("Pixel format {} is supported for camera {}",
                      magic_enum::enum_name(pixel_format_), camera_sn_);
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
    spdlog::trace("Setting pixel format to {}: {:#x} for camera {}.",
                  magic_enum::enum_name(pixel_format_),
                  getPixelFormatValue(pixel_format_), camera_sn_);
    HIK_CHECK_RETURN_BOOL(MV_CC_SetPixelFormat, "set pixel format", handle_,
                          getPixelFormatValue(pixel_format_));

    spdlog::trace("Pixel format set successfully to {} for camera {}.",
                  magic_enum::enum_name(pixel_format_), camera_sn_);
    return true;
}

std::pair<int, int> HikCamera::getResolution() const {
    std::shared_lock lock(mutex_);

    if (!resolution_.has_value()) {
        spdlog::trace("Resolution value is not set, will read from sdk.");
        if (!isOpen()) {
            spdlog::error("Camera is not open, cannot get resolution.");
            return UNKNOWN_RESOLUTION;
        }
        MVCC_INTVALUE width, height;
        if (MV_CC_GetWidth(handle_, &width) != MV_OK ||
            MV_CC_GetHeight(handle_, &height) != MV_OK) {
            spdlog::error("Failed to get resolution from sdk.");
            return UNKNOWN_RESOLUTION;
        }
        spdlog::debug("Getting resolution value from sdk: {}x{}.",
                      width.nCurValue, height.nCurValue);
        return {width.nCurValue, height.nCurValue};
    } else {
        spdlog::debug("Getting resolution value directly: {}x{}.",
                      resolution_.value().first, resolution_.value().second);
        return {resolution_.value().first, resolution_.value().second};
    }
}

bool HikCamera::setResolution(int width, int height) {
    spdlog::trace("Entering setResolution() for camera {}", camera_sn_);
    if (isCapturing()) {
        spdlog::error("Setting resolution is not supported when capturing.");
        return false;
    }
    std::unique_lock lock(mutex_);
    resolution_ = std::make_pair(width, height);
    return setResolutionInner();
}

float HikCamera::getGain() const {
    std::shared_lock lock(mutex_);
    if (!gain_.has_value()) {
        spdlog::trace("Gain value is not set, will read from sdk.");
        if (!isOpen()) {
            spdlog::error("Camera is not open, cannot get gain.");
            return UNKNOWN_GAIN;
        }
        MVCC_FLOATVALUE gain;
        if (MV_CC_GetGain(handle_, &gain) != MV_OK) {
            spdlog::error("Failed to get gain from sdk.");
            return UNKNOWN_GAIN;
        }
        spdlog::debug("Getting gain value from sdk: {}.", gain.fCurValue);
        return gain.fCurValue;
    } else {
        spdlog::debug("Getting gain value directly: {}.", gain_.value());
        return gain_.value();
    }
}

bool HikCamera::setGain(float gain) {
    spdlog::trace("Entering setGain() for camera {}", camera_sn_);
    std::unique_lock lock(mutex_);
    gain_ = gain;
    return setGainInner();
}

int HikCamera::getExposureTime() const {
    std::shared_lock lock(mutex_);
    if (!exposure_.has_value()) {
        spdlog::trace("Exposure value is not set, will read from sdk.");
        if (!isOpen()) {
            spdlog::error("Camera is not open, cannot get gain.");
            return UNKNOWN_EXPOSURE;
        }

        MVCC_FLOATVALUE exposure;
        if (MV_CC_GetExposureTime(handle_, &exposure) != MV_OK) {
            spdlog::error("Failed to get exposure from sdk.");
            return UNKNOWN_EXPOSURE;
        }
        spdlog::debug("Getting exposure time from sdk: {} μs.",
                      exposure.fCurValue);
        return exposure.fCurValue;
    } else {
        spdlog::debug("Getting exposure time directly: {} μs.",
                      exposure_.value());
        return static_cast<int>(exposure_.value());
    }
}

bool HikCamera::setExposureTime(int exposure) {
    spdlog::trace("Entering setExposureTime() for camera {}", camera_sn_);
    std::unique_lock lock(mutex_);
    exposure_ = exposure;
    return setExposureInner();
}

std::array<unsigned int, 3> HikCamera::getBalanceRatio() const {
    constexpr std::array<unsigned int, 3> UNKNOWN_BALANCE_RATIO{0, 0, 0};

    std::shared_lock lock(mutex_);
    if (!balance_ratio_.has_value()) {
        spdlog::trace("Balance ratio is not set, will read from sdk.");
        if (!isOpen()) {
            spdlog::error("Camera is not open, cannot get gain.");
            return UNKNOWN_BALANCE_RATIO;
        }

        MVCC_INTVALUE red, green, blue;
        if (MV_CC_GetBalanceRatioRed(handle_, &red) != MV_OK ||
            MV_CC_GetBalanceRatioGreen(handle_, &green) != MV_OK ||
            MV_CC_GetBalanceRatioBlue(handle_, &blue) != MV_OK) {
            spdlog::error("Failed to get balance ratio from sdk.");
            return UNKNOWN_BALANCE_RATIO;
        }
        spdlog::debug("Getting balance ratio from sdk: [{}, {}, {}].",
                      red.nCurValue, green.nCurValue, blue.nCurValue);
        return {red.nCurValue, green.nCurValue, blue.nCurValue};
    } else {
        spdlog::debug("Getting balance ratio directly: [{}, {}, {}].",
                      balance_ratio_.value()[0], balance_ratio_.value()[1],
                      balance_ratio_.value()[2]);
        return balance_ratio_.value();
    }
}

bool HikCamera::setBalanceRatio(unsigned int red, unsigned int green,
                                unsigned int blue) {
    std::unique_lock lock(mutex_);

    balance_ratio_ = {red, green, blue};

    return setBalanceRatioInner();
}

bool HikCamera::getBalanceRatioAuto() const {
    std::shared_lock lock(mutex_);
    return !balance_ratio_.has_value();
}

/**
 * @brief Enables auto white balance.
 * @param balance_auto Whether to enable auto white balance.
 * @return True if the auto white balance setting was applied successfully,
 * false otherwise.
 */
bool HikCamera::setBalanceRatioAuto() {
    spdlog::trace("Entering setBalanceRatioAuto() for camera {}", camera_sn_);
    std::unique_lock lock(mutex_);

    balance_ratio_ = std::nullopt;

    return setBalanceRatioInner();
}

std::string HikCamera::getCameraSn() const {
    std::shared_lock lock(mutex_);
    spdlog::debug("Getting camera serial number: {}.", camera_sn_);
    return camera_sn_;
}

bool HikCamera::setResolutionInner() {
    spdlog::trace("Entering setResolutionInner()");

    MVCC_INTVALUE width;
    MVCC_INTVALUE height;
    HIK_CHECK_RETURN_BOOL(MV_CC_GetWidth, "get width", handle_, &width);
    HIK_CHECK_RETURN_BOOL(MV_CC_GetHeight, "get width", handle_, &height);

    spdlog::debug("Current width limits: nMin = {}, nMax = {}, nInc = {}",
                  width.nMin, width.nMax, width.nInc);
    spdlog::debug("Current height limits: nMin = {}, nMax = {}, nInc = {}",
                  height.nMin, height.nMax, height.nInc);

    if (!resolution_.has_value()) {
        spdlog::trace(
            "Resolution configuration is not set, will set to max resolution "
            "{}x{}.",
            width.nMax, height.nMax);
        HIK_CHECK_RETURN_BOOL(MV_CC_SetWidth, "set width", handle_, width.nMax);
        HIK_CHECK_RETURN_BOOL(MV_CC_SetHeight, "set height", handle_,
                              height.nMax);
    } else if (resolution_.value().first < width.nMin ||
               resolution_.value().first > width.nMax ||
               resolution_.value().second < height.nMin ||
               resolution_.value().second > height.nMax) {
        spdlog::warn(
            "Resolution {}x{} out of range (width: {}~{}, height: {}~{}), will "
            "set to max resolution {}x{}.",
            resolution_.value().first, resolution_.value().second, width.nMin,
            width.nMax, height.nMin, height.nMax, width.nMin, width.nMax);
        HIK_CHECK_RETURN_BOOL(MV_CC_SetWidth, "set width", handle_, width.nMax);
        HIK_CHECK_RETURN_BOOL(MV_CC_SetHeight, "set height", handle_,
                              height.nMax);
    } else {
        auto [curr_width, curr_height] = resolution_.value();

        if (curr_width % width.nInc != 0) {
            auto width_old = curr_width;
            curr_width = (curr_width / width.nInc + 1) * width.nInc;
            if (curr_width > width.nMax) {
                curr_width -= width.nInc;
            }
            spdlog::warn("Width should be times of {}, fixed from {} to {}.",
                         width.nInc, width_old, curr_width);
        }
        if (curr_height % width.nInc != 0) {
            auto height_old = curr_height;
            curr_height = (curr_height / width.nInc + 1) * width.nInc;
            if (curr_height > width.nMax) {
                curr_height -= width.nInc;
            }
            spdlog::warn("Width should be times of {}, fixed from {} to {}.",
                         width.nInc, height_old, curr_height);
        }
        spdlog::trace("Setting resolution to {}x{}.", curr_width, curr_height);
        HIK_CHECK_RETURN_BOOL(MV_CC_SetWidth, "set width", handle_, curr_width);
        HIK_CHECK_RETURN_BOOL(MV_CC_SetHeight, "set height", handle_,
                              curr_height);
    }

    spdlog::trace("Resolution set successfully.");
    return true;
}

bool HikCamera::setExposureInner() {
    spdlog::trace("Entering setExposureInner()");

    if (!exposure_.has_value()) {
        spdlog::trace(
            "Exposure value is null, will set to auto exposure mode.");
        HIK_CHECK_RETURN_BOOL(
            MV_CC_SetExposureAutoMode, "set exposure auto", handle_,
            static_cast<unsigned int>(HikExposureAuto::Continuous));
    } else {
        MVCC_FLOATVALUE exposure;
        HIK_CHECK_RETURN_BOOL(MV_CC_GetExposureTime, "get exposure time",
                              handle_, &exposure);
        if (exposure_ > exposure.fMax || exposure_ < exposure.fMin) {
            spdlog::warn(
                "Exposure value {} out of range ({}~{}), exposure mode will be "
                "set to auto.",
                exposure_.value(), exposure.fMin, exposure.fMax);
            HIK_CHECK_RETURN_BOOL(
                MV_CC_SetExposureAutoMode, "set exposure auto", handle_,
                static_cast<unsigned int>(HikExposureAuto::Continuous));
        } else {
            spdlog::trace("Setting manual exposure with exposure time {} μs.",
                          exposure_.value());
            HIK_CHECK_RETURN_BOOL(
                MV_CC_SetExposureAutoMode, "set exposure auto", handle_,
                static_cast<unsigned int>(HikExposureAuto::Off));
            HIK_CHECK_RETURN_BOOL(MV_CC_SetExposureTime, "set exposure time",
                                  handle_, exposure_.value());
        }
    }

    spdlog::trace("Exposure setting applied successfully.");
    return true;
}

bool HikCamera::setGammaInner() {
    spdlog::trace("Entering setGammaInner()");

    if (!gamma_.has_value()) {
        spdlog::trace(
            "Gamma value is null, gamma correction will not be enabled.");
        HIK_CHECK_RETURN_BOOL(MV_CC_SetBoolValue, "set gamma enable", handle_,
                              "GammaEnable", false);
    } else {
        MVCC_FLOATVALUE gamma;
        HIK_CHECK_RETURN_BOOL(MV_CC_GetGamma, "get gamma", handle_, &gamma);

        if (gamma_ > gamma.fMax || gamma_ < gamma.fMin) {
            spdlog::error(
                "Gamma value {} out of range ({}~{}), gamma correction will be "
                "disabled.",
                gamma_.value(), gamma.fMin, gamma.fMax);
            HIK_CHECK_RETURN_BOOL(MV_CC_SetBoolValue, "set gamma enable",
                                  handle_, "GammaEnable", false);
        } else {
            spdlog::trace("Enabling gamma correction with gamma value: {}.",
                          gamma_.value());
            HIK_CHECK_RETURN_BOOL(MV_CC_SetBoolValue, "set gamma enable",
                                  handle_, "GammaEnable", true);
            HIK_CHECK_RETURN_BOOL(
                MV_CC_SetGammaSelector, "set gamma selector", handle_,
                static_cast<unsigned int>(HikGammaSelector::User));
            HIK_CHECK_RETURN_BOOL(MV_CC_SetGamma, "set gamma", handle_,
                                  gamma_.value());
        }
    }

    spdlog::trace("Gamma setting applied successfully.");
    return true;
}

bool HikCamera::setGainInner() {
    spdlog::trace("Entering setGainInner()");

    if (!gain_.has_value()) {
        spdlog::trace("Gain value is null, gain mode will be set to auto.");
        HIK_CHECK_RETURN_BOOL(
            MV_CC_SetGainMode, "set gain auto", handle_,
            static_cast<unsigned int>(HikGainAuto::Continuous));
    } else {
        MVCC_FLOATVALUE gain;
        HIK_CHECK_RETURN_BOOL(MV_CC_GetGain, "get gain", handle_, &gain);

        if (gain_ > gain.fMax || gain_ < gain.fMin) {
            spdlog::error(
                "Gain value {} out of range ({}~{}), gain mode will be set to "
                "auto.",
                gain_.value(), gain.fMin, gain.fMax);
            HIK_CHECK_RETURN_BOOL(
                MV_CC_SetGainMode, "set gain auto", handle_,
                static_cast<unsigned int>(HikGainAuto::Continuous));
        } else {
            spdlog::trace("Setting manual gain with gain value: {}.",
                          gain_.value());

            HIK_CHECK_RETURN_BOOL(MV_CC_SetGainMode, "set gain auto", handle_,
                                  static_cast<unsigned int>(HikGainAuto::Off));
            HIK_CHECK_RETURN_BOOL(MV_CC_SetGain, "set gain", handle_,
                                  gain_.value());
        }
    }

    spdlog::trace("Gain setting applied successfully.");
    return true;
}

bool HikCamera::setBalanceRatioInner() {
    spdlog::trace("Entering setBalanceRatioInner()");

    if (!balance_ratio_.has_value()) {
        spdlog::trace("Setting auto white balance to Continuous mode.");
        HIK_CHECK_RETURN_BOOL(
            MV_CC_SetBalanceWhiteAuto, "set balance white auto", handle_,
            static_cast<unsigned int>(HikBalanceWhiteAuto::Continuous));

        spdlog::trace("Balance ratio auto set successfully.");
        return true;
    } else {
        MVCC_INTVALUE red_info, green_info, blue_info;
        HIK_CHECK_RETURN_BOOL(MV_CC_GetBalanceRatioRed, "get balance ratio red",
                              handle_, &red_info);
        HIK_CHECK_RETURN_BOOL(MV_CC_GetBalanceRatioGreen,
                              "get balance ratio green", handle_, &green_info);
        HIK_CHECK_RETURN_BOOL(MV_CC_GetBalanceRatioBlue,
                              "get balance ratio blue", handle_, &blue_info);

        auto& [red, green, blue] = balance_ratio_.value();

        if (red > red_info.nMax || red < red_info.nMin ||
            green > green_info.nMax || green < green_info.nMin ||
            blue > blue_info.nMax || blue < blue_info.nMin) {
            spdlog::error(
                "Invalid balance ratio: red={} ({}~{}), green={} ({}~{}), "
                "blue={} "
                "({}~{})",
                red, red_info.nMin, red_info.nMax, green, green_info.nMin,
                green_info.nMax, blue, blue_info.nMin, blue_info.nMax);

            return false;
        } else {
            spdlog::trace(
                "Setting manual balance ratios: Red = {}, Green = {}, Blue = "
                "{}.",
                red, green, blue);
            HIK_CHECK_RETURN_BOOL(MV_CC_SetBalanceRatioRed,
                                  "set balance ratio red", handle_, red);
            HIK_CHECK_RETURN_BOOL(MV_CC_SetBalanceRatioGreen,
                                  "set balance ratio green", handle_, green);
            HIK_CHECK_RETURN_BOOL(MV_CC_SetBalanceRatioBlue,
                                  "set balance ratio blue", handle_, blue);
            spdlog::trace("Balance ratio set successfully.");

            return true;
        }
    }
}

void HikCamera::setExceptionOccurred(bool occurred) {
    spdlog::trace("Setting exception occurred state to: {}.", occurred);
    exception_occurred_ = occurred;
}

bool HikCamera::isExceptionOccurred() const {
    spdlog::debug("Checking if exception occurred: {}.",
                  exception_occurred_.load());
    return exception_occurred_;
}

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

std::string HikCamera::getCameraInfo() const {
    std::shared_lock lock(mutex_);
    spdlog::debug("Acquired lock for opening camera {}", camera_sn_);
    return HikCamera::getCameraInfo(device_info_.get());
}

void HikCamera::exceptionHandler(unsigned int code, void* user) {
    spdlog::trace("Reaching exception handler.");
    auto camera = static_cast<HikCamera*>(user);
    spdlog::error("Exception occurred in HikCamera {}: code {}",
                  camera->getCameraSn(), code);
    camera->setExceptionOccurred(true);
}

void HikCamera::startDaemonThread() {
    daemon_thread_ = std::jthread([this](std::stop_token token) {
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (isExceptionOccurred()) {
                spdlog::warn(
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
        spdlog::info("Daemon thread for camera {} stopped.", camera_sn_);
    });
    spdlog::info("Camera {} daemon thread started.", camera_sn_);
}

std::span<MV_CC_DEVICE_INFO*> HikCamera::getDeviceInfoList() {
    std::call_once(is_device_info_list_init_, [] {
        device_info_list_ = std::make_shared<MV_CC_DEVICE_INFO_LIST>();
        std::memset(device_info_list_.get(), 0, sizeof(MV_CC_DEVICE_INFO_LIST));
        spdlog::trace("Initialized MV_CC_DEVICE_INFO_LIST structure");

        HIK_CHECK_NORETURN(MV_CC_EnumDevices, "enum devices", MV_USB_DEVICE,
                           device_info_list_.get());
        spdlog::debug("Enumerated devices, found {} devices.",
                      device_info_list_->nDeviceNum);
        for (unsigned int i = 0; i < device_info_list_->nDeviceNum; ++i) {
            auto device_info = device_info_list_->pDeviceInfo[i];
            spdlog::debug("Device {}: {}", i,
                          HikCamera::getCameraInfo(device_info));
        }
    });
    return std::span<MV_CC_DEVICE_INFO*>(device_info_list_->pDeviceInfo,
                                         device_info_list_->nDeviceNum);
}

cv::Mat HikCamera::convertPixelFormat(const cv::Mat& input_image,
                                      PixelFormat format) {
    spdlog::debug("Starting pixel format conversion from {} to {}",
                  magic_enum::enum_name(pixel_format_),
                  magic_enum::enum_name(format));
    spdlog::debug("Input image size: {}x{}x{}, type: {}",
                  input_image.size().width, input_image.size().height,
                  input_image.channels(), input_image.type());
    cv::Mat output_image;
    switch (pixel_format_) {
        case HikPixelFormat::BGR8:
            spdlog::trace("Handling HikPixelFormat::RGB8");
            switch (format) {
                case PixelFormat::GRAY:
                    spdlog::debug("Converting from BGR to GRAY");
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2GRAY);
                    break;
                case PixelFormat::RGB:
                    spdlog::debug("Converting from BGR to RGB");
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2RGB);
                    break;
                case PixelFormat::BGR:
                    spdlog::debug(
                        "No conversion needed for BGR, will clone image");
                    output_image = input_image.clone();
                    break;
                case PixelFormat::RGBA:
                    spdlog::debug("Converting from BGR to RGBA");
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2RGBA);
                    break;
                case PixelFormat::BGRA:
                    spdlog::debug("Converting from BGR to BGRA");
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2BGRA);
                    break;
                case PixelFormat::HSV:
                    spdlog::debug("Converting from BGR to HSV");
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2HSV);
                    break;
                case PixelFormat::YUV:
                    spdlog::debug("Converting from BGR to YUV");
                    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2YUV);
                    break;
                default:
                    spdlog::error("Invalid target format for BGR8");
                    assert(0 && "unreachable code");
            }
            break;
        case HikPixelFormat::RGB8:
            spdlog::trace("Handling HikPixelFormat::RGB8");
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
                    spdlog::error("Invalid target format for RGB8");
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

    spdlog::trace("Pixel format conversion completed");
    spdlog::debug("Output image size: {}x{}x{}, type: {}",
                  output_image.size().width, output_image.size().height,
                  output_image.channels(), output_image.type());
    return output_image;
}

unsigned int HikCamera::getPixelFormatValue(HikPixelFormat format) {
    spdlog::debug("Pixel format in getPixelFormatValue: {}",
                  magic_enum::enum_name(format));
    switch (format) {
        case HikPixelFormat::RGB8:
            return PixelType_Gvsp_RGB8_Packed;
        case HikPixelFormat::BGR8:
            return PixelType_Gvsp_BGR8_Packed;
        case HikPixelFormat::YUV422_8:
            return PixelType_Gvsp_YUV422_Packed;
        case HikPixelFormat::YUV422_8_UYVY:
            return PixelType_Gvsp_YUV422_YUYV_Packed;
        case HikPixelFormat::BayerGR8:
            return PixelType_Gvsp_BayerGR8;
        case HikPixelFormat::BayerRG8:
            return PixelType_Gvsp_BayerRG8;
        case HikPixelFormat::BayerGB8:
            return PixelType_Gvsp_BayerGB8;
        case HikPixelFormat::BayerBG8:
            return PixelType_Gvsp_BayerBG8;
        default:
            return 0x0;
    }
}

}  // namespace radar::camera