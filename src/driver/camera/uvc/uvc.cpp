#include "uvc.h"

#include <spdlog/spdlog.h>

#include <magic_enum/magic_enum_all.hpp>
#include <stdexcept>
#include <string>

#define UVC_CHECK_RETURN_BOOL(func, msg, ...)                                 \
    do {                                                                      \
        uvc_error_t err = func(__VA_ARGS__);                                  \
        if (UVC_SUCCESS != err) {                                             \
            spdlog::error("Failed to {}, error: {}", msg, uvc_strerror(err)); \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define UVC_CHECK_NORETURN(func, msg, ...)                                    \
    do {                                                                      \
        uvc_error_t err = func(__VA_ARGS__);                                  \
        if (UVC_SUCCESS != err) {                                             \
            spdlog::error("Failed to {}, error: {}", msg, uvc_strerror(err)); \
        }                                                                     \
    } while (0)

namespace radar::camera {

UvcCamera::UvcCamera(int vendor_id, int product_id,
                     std::string_view serial_number,
                     std::string_view pixel_format, int width, int height,
                     int fps, int exposure, int gamma, int gain,
                     bool auto_white_balance,
                     std::array<unsigned int, 3>&& balance_ratio)
    : vendor_id_{vendor_id},
      product_id_{product_id},
      serial_number_{serial_number},
      pixel_format_{magic_enum::enum_cast<UvcPixelFormat>(pixel_format)
                        .value_or(UvcPixelFormat::Unknown)},
      width_{width},
      height_{height},
      fps_{fps},
      exposure_{exposure},
      gamma_{gamma},
      gain_{gain},
      auto_white_balance_{auto_white_balance},
      balance_ratio_{balance_ratio} {
    spdlog::trace("Initializing uvc camera with vid: {}, pid: {}, sn: {}",
                  vendor_id, product_id, serial_number);
    spdlog::debug(
        "Camera parameters: width={}, height={}, fps={} exposure={}, gamma={}, "
        "gain={}, auto white balance: {}, balance ratio: [{}, {}, {}]",
        width, height, fps, exposure, gamma, gain, auto_white_balance,
        balance_ratio[0], balance_ratio[1], balance_ratio[2]);
}

UvcCamera::~UvcCamera() {
    spdlog::info("Destroying uvc camera with vid: {}, pid: {}, sn: {}",
                 vendor_id_, product_id_, serial_number_);

    // Check if the camera is open before attempting to close it
    spdlog::trace("Checking if camera with vid: {}, pid: {}, sn: {} is open.",
                  vendor_id_, product_id_, serial_number_);
    if (isOpen()) {
        spdlog::debug(
            "Camera with vid: {}, pid: {}, sn: {} is open. Attempting to close "
            "it.",
            vendor_id_, product_id_, serial_number_);
        close();
    }
}

bool UvcCamera::open() {
    spdlog::info("Opening camera with vid: {}, pid: {}, sn: {}", vendor_id_,
                 product_id_, serial_number_);

    // Initialize UVC context
    spdlog::trace("Initializing UVC camera");
    UVC_CHECK_RETURN_BOOL(uvc_init, "init", &context_, nullptr);

    // Find the device
    spdlog::trace(
        "Searching for UVC device with vendor_id: {}, product_id: {}, serial: "
        "{}",
        vendor_id_, product_id_, serial_number_);
    UVC_CHECK_RETURN_BOOL(
        uvc_find_device, "find device", context_, &device_, vendor_id_,
        product_id_, serial_number_.empty() ? nullptr : serial_number_.c_str());

    // Open the device
    spdlog::trace("Opening UVC device");
    UVC_CHECK_RETURN_BOOL(uvc_open, "open", device_, &device_handle_);

    // Set format and log the supported formats
    spdlog::trace("Setting UVC camera format");
    auto supported_formats = getSupportedFormats();
    for (const auto& format : supported_formats) {
        spdlog::debug(
            "Supported format: pixel={}, width={}, height={}, fps={}",
            magic_enum::enum_name(getPixelFormatFromGuid(format.guid)),
            format.width, format.height, format.fps);
    }
    setFormat(supported_formats);

    // Get stream control format size
    spdlog::trace(
        "Getting stream control format size for pixel format: {}, width: {}, "
        "height: {}, fps: {}",
        magic_enum::enum_name(pixel_format_), width_, height_, fps_);
    UVC_CHECK_RETURN_BOOL(
        uvc_get_stream_ctrl_format_size, "get stream control format size",
        device_handle_, stream_ctrl_,
        static_cast<uvc_frame_format>(pixel_format_), width_, height_, fps_);

    spdlog::trace(
        "Setting camera configuration (balance ratio, exposure, "
        "gamma, gain) for camera with vid: {}, pid: {}, sn: {}",
        vendor_id_, product_id_, serial_number_);
    if (!setBalanceRatioInner()) {
        spdlog::critical(
            "Failed to set balance ratio for camera with vid: {}, pid: {}, sn: "
            "{}",
            vendor_id_, product_id_, serial_number_);
        return false;
    }
    spdlog::debug(
        "Balance ratio set successfully for camera with vid: {}, pid: {}, sn: "
        "{}",
        vendor_id_, product_id_, serial_number_);

    if (!setExposureInner()) {
        spdlog::critical(
            "Failed to set exposure for camera with vid: {}, pid: {}, sn: {}",
            vendor_id_, product_id_, serial_number_);
        return false;
    }
    spdlog::debug(
        "Exposure set successfully for camera with vid: {}, pid: {}, sn: {}",
        vendor_id_, product_id_, serial_number_);

    if (!setGammaInner()) {
        spdlog::critical(
            "Failed to set gamma for camera with vid: {}, pid: {}, sn: {}",
            vendor_id_, product_id_, serial_number_);
        return false;
    }
    spdlog::debug(
        "Gamma set successfully for camera with vid: {}, pid: {}, sn: {}",
        vendor_id_, product_id_, serial_number_);

    if (!setGainInner()) {
        spdlog::critical(
            "Failed to set gain for camera with vid: {}, pid: {}, sn: {}",
            vendor_id_, product_id_, serial_number_);
        return false;
    }
    spdlog::debug(
        "Gain set successfully for camera with vid: {}, pid: {}, sn: {}",
        vendor_id_, product_id_, serial_number_);

    is_open_ = true;
    spdlog::info("Camera with vid: {}, pid: {}, sn: {} opened successfully.",
                 vendor_id_, product_id_, serial_number_);
    return true;
}

void UvcCamera::close() {
    spdlog::info("Closing camera with vid: {}, pid: {}, sn: {}", vendor_id_,
                 product_id_, serial_number_);

    if (isCapturing()) {
        spdlog::debug(
            "Camera with vid: {}, pid: {}, sn: {} is capturing, attempting to "
            "stop capture.",
            vendor_id_, product_id_, serial_number_);
        if (stopCapture()) {
            spdlog::info(
                "Capture stopped successfully for camera with vid: {}, pid: "
                "{}, sn: {}",
                vendor_id_, product_id_, serial_number_);
        } else {
            spdlog::error(
                "Failed to stop capture for camera with vid: {}, pid: {}, sn: "
                "{}",
                vendor_id_, product_id_, serial_number_);
        }
    }

    if (device_handle_ != nullptr) {
        spdlog::trace("Closing device for camera with vid: {}, pid: {}, sn: {}",
                      vendor_id_, product_id_, serial_number_);
        uvc_close(device_handle_);
        device_handle_ = nullptr;
    }

    if (device_ != nullptr) {
        spdlog::trace(
            "Unreferencing device for camera with vid: {}, pid: {}, sn: {}",
            vendor_id_, product_id_, serial_number_);
        uvc_unref_device(device_);
        device_ = nullptr;
    }

    if (context_ != nullptr) {
        spdlog::trace(
            "Exiting context for camera with vid: {}, pid: {}, sn: {}",
            vendor_id_, product_id_, serial_number_);
        uvc_exit(context_);
        context_ = nullptr;
    }

    is_open_ = false;
    spdlog::info("Camera with vid: {}, pid: {}, sn: {} closed.", vendor_id_,
                 product_id_, serial_number_);
}

bool UvcCamera::reconnect() {
    spdlog::warn("Reconnecting camera with vid: {}, pid: {}, sn: {}.",
                 vendor_id_, product_id_, serial_number_);

    spdlog::trace(
        "Closing camera with vid: {}, pid: {}, sn: {} before attempting "
        "reconnection.",
        vendor_id_, product_id_, serial_number_);
    close();

    spdlog::trace(
        "Reopening camera with vid: {}, pid: {}, sn: {} during reconnection.",
        vendor_id_, product_id_, serial_number_);
    if (!open()) {
        spdlog::error(
            "Failed to open camera with vid: {}, pid: {}, sn: {} during "
            "reconnection.",
            vendor_id_, product_id_, serial_number_);
        return false;
    } else {
        spdlog::info(
            "Camera with vid: {}, pid: {}, sn: {} reconnected successfully.",
            vendor_id_, product_id_, serial_number_);
        return true;
    }
}

bool UvcCamera::startCapture() {
    spdlog::debug(
        "Attempting to start capture on camera with vid: {}, pid: {}, sn: {}",
        vendor_id_, product_id_, serial_number_);

    spdlog::trace(
        "Calling uvc_start_streaming for camera with vid: {}, pid: {}, sn: {}.",
        vendor_id_, product_id_, serial_number_);
    UVC_CHECK_RETURN_BOOL(uvc_start_streaming, "start streaming",
                          device_handle_, stream_ctrl_,
                          &UvcCamera::frameCallback, this, 0);
    spdlog::debug(
        "uvc_start_streaming called successfully for camera with vid: {}, pid: "
        "{}, sn: {}.",
        vendor_id_, product_id_, serial_number_);

    is_capturing_ = true;
    spdlog::debug("Camera with vid: {}, pid: {}, sn: {} marked as capturing.",
                  vendor_id_, product_id_, serial_number_);

    spdlog::info(
        "Capture started successfully on camera with vid: {}, pid: {}, sn: {}",
        vendor_id_, product_id_, serial_number_);
    return true;
}

bool UvcCamera::stopCapture() {
    spdlog::trace(
        "Attempting to stop capture on camera with vid: {}, pid: {}, sn: {}",
        vendor_id_, product_id_, serial_number_);

    spdlog::trace(
        "Calling uvc_stop_streaming for camera with vid: {}, pid: {}, sn: {}.",
        vendor_id_, product_id_, serial_number_);
    uvc_stop_streaming(device_handle_);
    spdlog::debug(
        "uvc_stop_streaming called successfully for camera with vid: {}, pid: "
        "{}, sn: {}.",
        vendor_id_, product_id_, serial_number_);

    is_capturing_ = false;
    spdlog::debug(
        "Camera with vid: {}, pid: {}, sn: {} marked as not capturing.",
        vendor_id_, product_id_, serial_number_);

    return true;
}

std::span<UvcFormat> UvcCamera::getSupportedFormats() {
    const uvc_format_desc_t* format_desc;
    const uvc_frame_desc_t* frame_desc;
    format_desc = uvc_get_format_descs(device_handle_);

    while (format_desc != nullptr) {
        // Convert guidFormat to string once per format_desc
        auto guid = format_desc->guidFormat;

        // Process each frame_desc for this format_desc
        frame_desc = format_desc->frame_descs;
        while (frame_desc != nullptr) {
            uint32_t* interval_ptr = frame_desc->intervals;
            if (interval_ptr) {
                while (*interval_ptr != 0) {
                    UvcFormat format;
                    std::copy(std::begin(format_desc->guidFormat),
                              std::end(format_desc->guidFormat),
                              format.guid.begin());
                    format.width = frame_desc->wWidth;
                    format.height = frame_desc->wHeight;
                    format.fps = static_cast<int>(10000000.0 / *interval_ptr);

                    supported_formats_.emplace_back(std::move(format));
                    interval_ptr++;
                }
            }

            // Move to the next frame descriptor
            frame_desc = frame_desc->next;
        }

        // Move to the next format descriptor
        format_desc = format_desc->next;
    }

    return supported_formats_;
}

bool UvcCamera::setFormat(std::span<UvcFormat> supported_formats) {
    // Define the priority order of candidate pixel formats
    static const std::vector<UvcPixelFormat> candidate_pixel_formats = {
        UvcPixelFormat::MJPEG, UvcPixelFormat::RGB24, UvcPixelFormat::YUYV,
        UvcPixelFormat::UYVY};

    // Iterate through all supported formats and attempt to find the best match
    for (const auto& uvc_format : supported_formats) {
        // Get the pixel format from the GUID
        UvcPixelFormat format = getPixelFormatFromGuid(uvc_format.guid);

        // Case 1: Both pixel format and resolution are known
        if (pixel_format_ != UvcPixelFormat::Unknown && width_ > 0 &&
            height_ > 0) {
            if (format == pixel_format_ && uvc_format.width == width_ &&
                uvc_format.height == height_) {
                fps_ = uvc_format.fps;
                return true;
            }
        }

        // Case 2: Pixel format is unknown, but resolution is known
        if (pixel_format_ == UvcPixelFormat::Unknown && width_ > 0 &&
            height_ > 0) {
            if (uvc_format.width == width_ && uvc_format.height == height_) {
                fps_ = uvc_format.fps;
                pixel_format_ = format;
                return true;
            }
        }

        // Case 3: Pixel format is known, but resolution is unknown
        if (pixel_format_ != UvcPixelFormat::Unknown &&
            (width_ <= 0 || height_ <= 0)) {
            if (format == pixel_format_) {
                width_ = uvc_format.width;
                height_ = uvc_format.height;
                fps_ = uvc_format.fps;
                return true;
            }
        }

        // Case 4: Both pixel format and resolution are unknown
        if (pixel_format_ == UvcPixelFormat::Unknown && width_ <= 0 &&
            height_ <= 0) {
            for (auto candidate_format : candidate_pixel_formats) {
                if (format == candidate_format) {
                    width_ = uvc_format.width;
                    height_ = uvc_format.height;
                    fps_ = uvc_format.fps;
                    pixel_format_ = format;
                    return true;
                }
            }
        }
    }

    // If no suitable format is found, return false
    return false;
}

std::string UvcCamera::getCameraInfo() const {
    uvc_device_descriptor_t* desc;
    uvc_get_device_descriptor(device_, &desc);
    auto descriptor_info = fmt::format(
        "Camera vid: {}, pid: {}, sn: {}, manufacturer: {}, product: {}",
        vendor_id_, product_id_, serial_number_, desc->manufacturer,
        desc->product);
    std::string format_info = "supported formats: [";
    for (size_t i = 0; i < supported_formats_.size(); ++i) {
        auto format = supported_formats_[i];
        auto& guid = format.guid;
        format_info += fmt::format(
            "format GUID: "
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:"
            "02x}{:02x}{:02x}{:02x}{:02x}{:02x}, resolution: {}x{}, fps: {}",
            guid[0], guid[1], guid[2], guid[3], guid[4], guid[5], guid[6],
            guid[7], guid[8], guid[9], guid[10], guid[11], guid[12], guid[13],
            guid[14], guid[15], format.width, format.height, format.fps);
        if (i != supported_formats_.size() - 1) {
            format_info += "; ";
        }
    }
    format_info += "]";
    return descriptor_info + ", " + format_info + ".";
}

void UvcCamera::frameCallback(uvc_frame_t* frame, void* userdata) {
    auto camera = static_cast<UvcCamera*>(userdata);
    if (frame->data == nullptr || frame->data_bytes == 0) {
        spdlog::warn(
            "Frame is empty or null for camera with vid: {}, pid: {}, sn: {}.",
            camera->vendor_id_, camera->product_id_, camera->serial_number_);
        return;
    }

    if (camera->pixel_format_ == UvcPixelFormat::MJPEG) {
        spdlog::trace(
            "Pixel format is MJPEG for camera with vid: {}, pid: "
            "{}, sn: {}.",
            camera->vendor_id_, camera->product_id_, camera->serial_number_);
        auto begin = static_cast<unsigned char*>(frame->data);
        std::vector<unsigned char> buffer(begin, begin + frame->data_bytes);
        std::unique_lock lock(camera->image_mutex_);
        spdlog::debug(
            "Lock acquired for grabbing image for camera with vid: {}, pid: "
            "{}, sn: {}.",
            camera->vendor_id_, camera->product_id_, camera->serial_number_);
        camera->image_ = cv::imdecode(buffer, cv::IMREAD_COLOR);
        if (camera->image_.empty()) {
            spdlog::error(
                "Failed to decode MJPEG image for camera with vid: {}, pid: "
                "{}, sn: {}.",
                camera->vendor_id_, camera->product_id_,
                camera->serial_number_);
        }
    } else if (camera->pixel_format_ == UvcPixelFormat::UYVY ||
               camera->pixel_format_ == UvcPixelFormat::YUYV) {
        spdlog::trace(
            "Pixel format is UYVY or YUYV for camera with vid: {}, pid: "
            "{}, sn: {}.",
            camera->vendor_id_, camera->product_id_, camera->serial_number_);
        std::unique_lock lock(camera->image_mutex_);
        spdlog::debug(
            "Lock acquired for grabbing image for camera with vid: {}, pid: "
            "{}, sn: {}.",
            camera->vendor_id_, camera->product_id_, camera->serial_number_);
        camera->image_ =
            cv::Mat(camera->height_, camera->width_, CV_8UC2, frame->data);
    } else if (camera->pixel_format_ == UvcPixelFormat::RGB24) {
        spdlog::trace(
            "Pixel format is RGB24 for camera with vid: {}, pid: "
            "{}, sn: {}.",
            camera->vendor_id_, camera->product_id_, camera->serial_number_);
        std::unique_lock lock(camera->image_mutex_);
        spdlog::debug(
            "Lock acquired for grabbing image for camera with vid: {}, pid: "
            "{}, sn: {}.",
            camera->vendor_id_, camera->product_id_, camera->serial_number_);
        camera->image_ =
            cv::Mat(camera->height_, camera->width_, CV_8UC3, frame->data);
    } else {
        spdlog::error(
            "Unknown pixel format for camera with vid: {}, pid: {}, sn: {}.",
            camera->vendor_id_, camera->product_id_, camera->serial_number_);
    }
}

bool UvcCamera::grabImage(cv::Mat& image,
                          camera::PixelFormat pixel_format) noexcept {
    spdlog::debug(
        "Attempting to grab image from camera with vid: {}, pid: {}, sn: {}",
        vendor_id_, product_id_, serial_number_);

    // Check if the camera is open and capturing
    if (!isOpen() || !isCapturing()) {
        spdlog::error(
            "Camera with vid: {}, pid: {}, sn: {} is not open or capturing.",
            vendor_id_, product_id_, serial_number_);
        return false;
    }

    spdlog::trace(
        "Acquiring lock to grab image for camera with vid: {}, pid: {}, sn: "
        "{}.",
        vendor_id_, product_id_, serial_number_);
    {
        std::shared_lock lock(image_mutex_);
        spdlog::debug(
            "Lock acquired for grabbing image for camera with vid: {}, pid: "
            "{}, sn: {}.",
            vendor_id_, product_id_, serial_number_);

        image = image_.clone();
        spdlog::trace("Image cloned for camera with vid: {}, pid: {}, sn: {}.",
                      vendor_id_, product_id_, serial_number_);
    }

    spdlog::trace(
        "Converting pixel format for camera with vid: {}, pid: {}, sn: {}.",
        vendor_id_, product_id_, serial_number_);
    convertPixelFormat(image, pixel_format);
    spdlog::debug(
        "Pixel format conversion completed for camera with vid: {}, pid: {}, "
        "sn: {}.",
        vendor_id_, product_id_, serial_number_);

    spdlog::trace(
        "Image grabbed successfully from camera with vid: {}, pid: {}, sn: {}.",
        vendor_id_, product_id_, serial_number_);
    return true;
}

bool UvcCamera::getBalanceRatioAuto() const { return auto_white_balance_; }

bool UvcCamera::setBalanceRatioAuto(bool balance_auto) {
    auto_white_balance_ = balance_auto;
    return setBalanceRatioInner();
}

std::array<unsigned int, 3> UvcCamera::getBalanceRatio() const {
    return balance_ratio_;
}

bool UvcCamera::setBalanceRatio(std::array<unsigned int, 3>&& balance) {
    balance_ratio_ = balance;
    return setBalanceRatioInner();
}

std::pair<int, int> UvcCamera::getResolution() const {
    return std::make_pair(width_, height_);
}

bool UvcCamera::setResolution(int width, int height) {
    spdlog::error("Setting resolution is not supported for uvc camera.");
    return false;
}

float UvcCamera::getGain() const { return static_cast<float>(gain_); }

bool UvcCamera::setGain(float gain) {
    gain_ = static_cast<int>(gain);
    return setGainInner();
}

int UvcCamera::getExposureTime() const { return exposure_; }

bool UvcCamera::setExposureTime(int exposure) {
    exposure_ = exposure;
    return setExposureInner();
}

bool UvcCamera::setBalanceRatioInner() {
    if (auto_white_balance_) {
        UVC_CHECK_RETURN_BOOL(uvc_set_white_balance_component_auto,
                              "set white balance component auto",
                              device_handle_, 1);
    } else {
        UVC_CHECK_NORETURN(uvc_set_white_balance_component_auto,
                           "set white balance component auto", device_handle_,
                           0);
        UVC_CHECK_RETURN_BOOL(
            uvc_set_white_balance_component, "set white balance component",
            device_handle_,
            static_cast<uint16_t>(balance_ratio_[2] / balance_ratio_[1]),
            static_cast<uint16_t>(balance_ratio_[0] / balance_ratio_[1]));
    }
    return true;
}

bool UvcCamera::setExposureInner() {
    if (exposure_ < 0) {
        UVC_CHECK_RETURN_BOOL(uvc_set_ae_mode, "set auto exposure mode",
                              device_handle_, 2);
    } else {
        UVC_CHECK_RETURN_BOOL(uvc_set_ae_mode, "set auto exposure mode",
                              device_handle_, 1);
        UVC_CHECK_RETURN_BOOL(uvc_set_exposure_abs, "set exposure abs",
                              device_handle_, static_cast<uint32_t>(exposure_));
    }
    return true;
}

bool UvcCamera::setGammaInner() {
    if (gamma_ > 0) {
        UVC_CHECK_RETURN_BOOL(uvc_set_gamma, "set gamma", device_handle_,
                              static_cast<uint16_t>(gamma_));
    }
    return true;
}

bool UvcCamera::setGainInner() {
    if (gain_ > 0) {
        UVC_CHECK_RETURN_BOOL(uvc_set_gain, "set gain", device_handle_,
                              static_cast<uint16_t>(gain_));
    }
    return true;
}

UvcCamera::UvcPixelFormat UvcCamera::getPixelFormatFromGuid(
    const std::array<uint8_t, 16>& guid) {
    // todo
    return UvcCamera::UvcPixelFormat::YUYV;
}

void UvcCamera::convertPixelFormat(cv::Mat& image, PixelFormat format) {
    switch (pixel_format_) {
        case UvcPixelFormat::RGB24:
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

        case UvcPixelFormat::MJPEG:
            // Image has been decoded to BGR format
            switch (format) {
                case PixelFormat::GRAY:
                    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
                    break;
                case PixelFormat::RGB:
                    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                    break;
                case PixelFormat::BGR:
                    break;
                case PixelFormat::RGBA:
                    cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
                    break;
                case PixelFormat::BGRA:
                    cv::cvtColor(image, image, cv::COLOR_BGR2BGRA);
                    break;
                case PixelFormat::HSV:
                    cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
                    break;
                case PixelFormat::YUV:
                    cv::cvtColor(image, image, cv::COLOR_BGR2YUV);
                    break;
                default:
                    assert(0 && "unreachable code");
            }
            break;

        case UvcPixelFormat::UYVY:
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

        case UvcPixelFormat::YUYV:
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

        default:
            spdlog::error("Pixel format is unknown in format converting");
    }
}

}  // namespace radar::camera