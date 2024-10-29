/**
 * @file hik.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief Defines the HikCamera class for interfacing with Hikvision cameras.
 *
 * This file contains the declaration of the `HikCamera` class, a specialized
 * implementation of the `ColorCamera` interface. The class supports a wide
 * variety of camera control functionalities, including resolution, exposure,
 * gain, white balance, and pixel format conversion, and integrates with
 * Hikvision's SDK through `MvCameraControl.h`.
 * @date 2024-10-27
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <MvCameraControl.h>
#include <spdlog/spdlog.h>

#include <array>
#include <atomic>
#include <memory>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <thread>

#include "driver/camera/base.h"

namespace radar::camera {

/**
 * @brief A class representing a Hikvision camera interface.
 *
 * This class extends `ColorCamera` and provides functionality for controlling
 * and capturing images from Hikvision cameras. It supports features such as
 * exposure control, white balance, gain adjustment, and pixel format
 * management. It also handles device connection and disconnection, and includes
 * methods for querying camera information.
 */
class HikCamera : public ColorCamera {
   public:
    /**
     * @brief Deleted default constructor.
     */
    HikCamera() = delete;

    HikCamera(std::string_view camera_sn, unsigned int width,
              unsigned int height, float exposure, float gamma, float gain,
              std::string_view pixel_format = "default",
              unsigned int grab_timeout = 2000, bool auto_white_balance = true,
              std::array<unsigned int, 3>&& balance_ratio = {0, 0, 0});
    ~HikCamera() override;
    bool open() override;
    void close() override;
    bool reconnect() override;
    bool startCapture() override;
    bool stopCapture() override;
    bool grabImage(cv::Mat& image,
                   camera::PixelFormat pixel_format) noexcept override;
    std::pair<int, int> getResolution() const override;
    bool setResolution(int width, int height) override;
    float getGain() const override;
    bool setGain(float gain) override;
    int getExposureTime() const override;
    bool setExposureTime(int exposure) override;
    std::array<unsigned int, 3> getBalanceRatio() const override;
    bool setBalanceRatio(std::array<unsigned int, 3>&& balance) override;
    bool setBalanceRatioAuto(bool balance_auto) override;
    bool getBalanceRatioAuto() const override;
    std::string getCameraInfo() const override;
    std::string getCameraSn() const;
    void setExceptionOccurred(bool occurred);
    bool isExceptionOccurred() const;

   private:
    /**
     * @brief Enum class representing the white balance auto mode.
     */
    enum class HikBalanceWhiteAuto { Off = 0, Continuous = 1, Once = 2 };

    /**
     * @brief Enum class representing the exposure auto mode.
     */
    enum class HikExposureAuto { Off = 0, Once = 1, Continuous = 2 };

    /**
     * @brief Enum class representing the gamma selection mode.
     */
    enum class HikGammaSelector { User = 1, sRGB = 2 };

    /**
     * @brief Enum class representing the gain auto mode.
     */
    enum class HikGainAuto { Off = 0, Once = 1, Continuous = 2 };

    /**
     * @brief Enum class representing the white balance ratio selector.
     */
    enum class HikBalanceRatioSelector { Red = 0, Green = 1, Blue = 2 };

    /**
     * @brief Enum class representing the trigger mode.
     */
    enum class HikTriggerMode { Off = 0, On = 1 };

    /**
     * @brief Enum class representing different pixel types supported by the
     * camera.
     */
    enum class HikPixelFormat {
        Unknown,        // 0x0
        RGB8Packed,     // 0x02180014
        YUV422_8,       // 0x02100032
        YUV422_8_UYVY,  // 0x0210001F
        BayerGR8,       // 0x01080008
        BayerRG8,       // 0x01080009
        BayerGB8,       // 0x0108000A
        BayerBG8        // 0x0108000B
    };

    static std::span<MV_CC_DEVICE_INFO*> getDeviceInfoList();
    static std::string getCameraInfo(MV_CC_DEVICE_INFO* device_info);
    static void exceptionHandler(unsigned int msg_type, void* user);
    static unsigned int getPixelFormatValue(HikPixelFormat format);
    bool setPixelFormat(std::span<unsigned int> supported_types);
    std::span<unsigned int> getSupportedPixelFormats();
    cv::Mat convertPixelFormat(const cv::Mat& image, PixelFormat format);
    void startDaemonThread();
    bool setResolutionInner();
    bool setBalanceRatioInner();
    bool setExposureInner();
    bool setGammaInner();
    bool setGainInner();

    std::string camera_sn_;
    unsigned int width_;
    unsigned int height_;
    float exposure_;
    float gamma_;
    float gain_;
    HikPixelFormat pixel_format_ = HikPixelFormat::Unknown;
    unsigned int grab_timeout_;
    bool auto_white_balance_;
    std::array<unsigned int, 3> balance_ratio_;
    void* handle_ = nullptr;
    std::atomic_bool exception_occurred_ = false;
    std::unique_ptr<MV_FRAME_OUT> frame_out_ = nullptr;
    std::unique_ptr<MV_CC_DEVICE_INFO> device_info_ = nullptr;
    mutable std::shared_mutex mutex_;
    std::unique_ptr<MVCC_ENUMVALUE> supported_pixel_formats_;
    inline static std::shared_ptr<MV_CC_DEVICE_INFO_LIST> device_info_list_;
    inline static std::once_flag is_device_info_list_init_;
    std::jthread daemon_thread_;
};

}  // namespace radar::camera