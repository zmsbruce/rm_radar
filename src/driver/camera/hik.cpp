#include "hik.h"

#include <spdlog/spdlog.h>

#include <span>
#include <thread>

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
      frame_out_{new MV_FRAME_OUT},
      device_info_{new MV_CC_DEVICE_INFO} {
    std::memset(frame_out_, 0, sizeof(MV_FRAME_OUT));
    std::memset(device_info_, 0, sizeof(MV_CC_DEVICE_INFO));
    startDaemonThread();
}

HikCamera::~HikCamera() {
    delete frame_out_;
    delete device_info_;
}

bool HikCamera::open() {
    std::unique_lock lock(mutex_);

    MV_CC_DEVICE_INFO_LIST device_info_list;
    std::memset(&device_info_list, 0, sizeof(MV_CC_DEVICE_INFO_LIST));

    CALL_AND_CHECK(MV_CC_EnumDevices, "enum devices", MV_USB_DEVICE,
                   &device_info_list);

    auto device_info_ptr = std::find_if(
        device_info_list.pDeviceInfo,
        device_info_list.pDeviceInfo + device_info_list.nDeviceNum,
        [this](MV_CC_DEVICE_INFO* device_info) {
            return reinterpret_cast<const char*>(
                       device_info->SpecialInfo.stUsb3VInfo.chSerialNumber) ==
                   camera_sn_;
        });
    if (device_info_ptr ==
        device_info_list.pDeviceInfo + device_info_list.nDeviceNum) {
        spdlog::critical("No devices found for {}", camera_sn_);
        return false;
    }

    CALL_AND_CHECK(MV_CC_CreateHandle, "create handle", &handle_,
                   *device_info_ptr);

    CALL_AND_CHECK(MV_CC_OpenDevice, "open device", handle_);

    MVCC_ENUMVALUE val;
    CALL_AND_CHECK(MV_CC_GetPixelFormat, "get pixel format", handle_, &val);
    auto supported_pixel_format =
        std::span(val.nSupportValue, val.nSupportedNum);
    spdlog::info("Supported pixel format: [{:#x}]",
                 fmt::join(supported_pixel_format, ", "));

    // todo: add condition that RGB8Packed is not supported
    CALL_AND_CHECK(MV_CC_SetPixelFormat, "set pixel format", handle_,
                   static_cast<unsigned int>(PixelFormat::RGB8Packed));

    if (!setResolutionInner() || !setBalanceRatioInner() ||
        !setExposureInner() || !setGammaInner() || !setGainInner()) {
        return false;
    }

    CALL_AND_CHECK(MV_CC_RegisterExceptionCallBack,
                   "register exception callback", handle_,
                   HikCamera::exceptionHandler, this);

    is_open_ = true;
    return true;
}

void HikCamera::close() {
    stopCapture();

    std::unique_lock lock(mutex_);

    int ret;
    ret = MV_CC_CloseDevice(handle_);
    if (ret != MV_OK) {
        spdlog::error("Failed to close device, error code: {:#x}", ret);
    }
    ret = MV_CC_DestroyHandle(handle_);
    if (ret != MV_OK) {
        spdlog::error("Failed to destroy handle, error code: {:#x}", ret);
    }

    handle_ = nullptr;
    is_open_ = false;
}

bool HikCamera::reconnect() {
    close();
    return open();
}

bool HikCamera::startCapture() {
    std::unique_lock lock(mutex_);

    CALL_AND_CHECK(MV_CC_StartGrabbing, "start grabbing", handle_);
    is_capturing_ = true;
    return true;
}

bool HikCamera::stopCapture() {
    std::unique_lock lock(mutex_);

    CALL_AND_CHECK(MV_CC_StopGrabbing, "stop grabbing", handle_);
    is_capturing_ = false;
    return true;
}

bool HikCamera::grabImage(cv::Mat& image,
                          camera::PixelFormat pixel_format) noexcept {
    if (!isOpen() || !isCapturing()) {
        return false;
    }

    std::unique_lock lock(mutex_);
    std::memset(frame_out_, 0, sizeof(MV_FRAME_OUT));
    CALL_AND_CHECK(MV_CC_GetImageBuffer, "get image buffer", handle_,
                   frame_out_, grab_timeout_);
    image =
        cv::Mat(frame_out_->stFrameInfo.nHeight, frame_out_->stFrameInfo.nWidth,
                CV_8UC3, frame_out_->pBufAddr);

    switch (pixel_format) {
        case camera::PixelFormat::GRAY:
            cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
            break;
        case camera::PixelFormat::BGR:
            cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
            break;
        case camera::PixelFormat::BGRA:
            cv::cvtColor(image, image, cv::COLOR_RGB2BGRA);
            break;
        case camera::PixelFormat::RGB:
            break;
        case camera::PixelFormat::RGBA:
            cv::cvtColor(image, image, cv::COLOR_RGB2RGBA);
            break;
        case camera::PixelFormat::HSV:
            cv::cvtColor(image, image, cv::COLOR_RGB2HSV);
            break;
        case camera::PixelFormat::YUV:
            cv::cvtColor(image, image, cv::COLOR_RGB2YUV);
            break;
        default:
            assert(false && "Unreachable code");
    }
    return true;
}

std::pair<int, int> HikCamera::getResolution() const {
    std::shared_lock lock(mutex_);
    return std::make_pair(width_, height_);
}

bool HikCamera::setResolution(int width, int height) {
    std::unique_lock lock(mutex_);
    width_ = width;
    height_ = height;
    setBalanceRatioInner();
    return true;
}

float HikCamera::getGain() const {
    std::shared_lock lock(mutex_);
    return gain_;
}

bool HikCamera::setGain(float gain) {
    std::unique_lock lock(mutex_);
    gain_ = gain;
    setGainInner();
    return true;
}

int HikCamera::getExposureTime() const {
    std::shared_lock lock(mutex_);
    return static_cast<int>(exposure_);
}

bool HikCamera::setExposureTime(int exposure) {
    std::unique_lock lock(mutex_);
    exposure_ = exposure;
    setExposureInner();
    return true;
}

std::array<unsigned int, 3> HikCamera::getBalanceRatio() const {
    std::shared_lock lock(mutex_);
    return balance_ratio_;
}

bool HikCamera::setBalanceRatio(std::array<unsigned int, 3>&& balance) {
    std::unique_lock lock(mutex_);
    balance_ratio_ = balance;
    setBalanceRatioInner();
    return true;
}

bool HikCamera::getBalanceRatioAuto() const {
    std::shared_lock lock(mutex_);
    return auto_white_balance_;
}

bool HikCamera::setBalanceRatioAuto(bool balance_auto) {
    std::unique_lock lock(mutex_);
    auto_white_balance_ = balance_auto;
    setBalanceRatioInner();
    return true;
}

std::string HikCamera::getCameraSn() const {
    std::shared_lock lock(mutex_);
    return camera_sn_;
}

bool HikCamera::setResolutionInner() {
    CALL_AND_CHECK(MV_CC_SetWidth, "set width", handle_, width_);
    CALL_AND_CHECK(MV_CC_SetHeight, "set height", handle_, height_);
    return true;
}

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

bool HikCamera::isOpen() const {
    std::shared_lock lock(mutex_);
    return is_open_;
}

bool HikCamera::isCapturing() const {
    std::shared_lock lock(mutex_);
    return is_capturing_;
}

void HikCamera::setExceptionFlag(bool flag) { exception_flag_ = flag; }

bool HikCamera::getExceptionFlag() const { return exception_flag_.load(); }

std::string HikCamera::getCameraInfo() const {
    std::shared_lock lock(mutex_);
    assert(device_info_->nTLayerType == MV_USB_DEVICE && "Wrong device type");
    auto info = &device_info_->SpecialInfo.stUsb3VInfo;
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

void HikCamera::exceptionHandler(unsigned int code, void* user) {
    auto camera = static_cast<HikCamera*>(user);
    spdlog::error("Exception occurred in HikCamera {}: code {}",
                  camera->getCameraSn(), code);
    camera->setExceptionFlag(true);
}

void HikCamera::startDaemonThread() {
    auto daemon_thread = std::thread([this] {
        while (true) {
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
    });
    spdlog::info("HikCamera {} daemon thread starts, id: {:#x}", camera_sn_,
                 std::hash<std::thread::id>{}(daemon_thread.get_id()));
    daemon_thread.detach();
}

}  // namespace radar::camera