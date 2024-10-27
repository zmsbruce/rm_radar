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
              unsigned int grab_timeout = 1000, bool auto_white_balance = true,
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
    bool isOpen() const override;
    bool isCapturing() const override;
    std::string getCameraSn() const;
    void setExceptionFlag(bool flag);
    bool getExceptionFlag() const;

   private:
    /**
     * @brief Enum class representing the white balance auto mode.
     */
    enum class BalanceWhiteAuto { Off = 0, Continuous = 1, Once = 2 };

    /**
     * @brief Enum class representing the exposure auto mode.
     */
    enum class ExposureAuto { Off = 0, Once = 1, Continuous = 2 };

    /**
     * @brief Enum class representing the gamma selection mode.
     */
    enum class GammaSelector { User = 1, sRGB = 2 };

    /**
     * @brief Enum class representing the gain auto mode.
     */
    enum class GainAuto { Off = 0, Once = 1, Continuous = 2 };

    /**
     * @brief Enum class representing the white balance ratio selector.
     */
    enum class BalanceRatioSelector { Red = 0, Green = 1, Blue = 2 };

    /**
     * @brief Enum class representing the trigger mode.
     */
    enum class TriggerMode { Off = 0, On = 1 };

    /**
     * @brief Enum class representing different pixel types supported by the
     * camera.
     */
    enum class PixelType {
        Unknown = 0x0,               ///< Unknown pixel type.
        RGB8Packed = 0x02180014,     ///< RGB 8-bit packed pixel format.
        YUV422_8 = 0x02100032,       ///< YUV 4:2:2 8-bit format.
        YUV422_8_UYVY = 0x0210001F,  ///< YUV 4:2:2 8-bit UYVY format.
        BayerGR8 = 0x01080008,       ///< Bayer GR 8-bit format.
        BayerRG8 = 0x01080009,       ///< Bayer RG 8-bit format.
        BayerGB8 = 0x0108000A,       ///< Bayer GB 8-bit format.
        BayerBG8 = 0x0108000B        ///< Bayer BG 8-bit format.
    };

    static std::span<MV_CC_DEVICE_INFO*> getDeviceInfoList();
    static std::string getCameraInfo(MV_CC_DEVICE_INFO* device_info);
    static void exceptionHandler(unsigned int msg_type, void* user);
    bool setPixelType(std::span<unsigned int> supported_types);
    void convertPixelFormat(cv::Mat& image, PixelFormat format);
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
    unsigned int grab_timeout_;
    bool auto_white_balance_;
    std::array<unsigned int, 3> balance_ratio_;
    void* handle_ = nullptr;
    std::atomic_bool exception_flag_ = false;
    std::unique_ptr<MV_FRAME_OUT> frame_out_ = nullptr;
    std::unique_ptr<MV_CC_DEVICE_INFO> device_info_ = nullptr;
    PixelType pixel_type_ = PixelType::Unknown;
    mutable std::shared_mutex mutex_;
    static std::shared_ptr<MV_CC_DEVICE_INFO_LIST> device_info_list_;
    static std::once_flag device_info_list_init_flag_;
    std::jthread daemon_thread_;
};

}  // namespace radar::camera