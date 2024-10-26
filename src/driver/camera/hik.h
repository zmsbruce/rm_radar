#pragma once

#include <MvCameraControl.h>
#include <spdlog/spdlog.h>

#include <array>
#include <atomic>
#include <shared_mutex>
#include <stdexcept>

#include "driver/camera/base.h"

namespace radar::camera {

class HikCamera : public ColorCamera {
   public:
    HikCamera() = delete;
    HikCamera(const std::string& camera_sn, unsigned int width,
              unsigned int height, float exposure, float gamma, float gain,
              int grab_timeout = 1000, bool auto_white_balance = true,
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
    void setExceptionFlag(bool flag);
    bool getExceptionFlag() const;
    bool isOpen() const override;
    bool isCapturing() const override;

   private:
    enum class BalanceWhiteAuto { Off = 0, Continuous = 1, Once = 2 };
    enum class ExposureAuto { Off = 0, Once = 1, Continuous = 2 };
    enum class GammaSelector { User = 1, sRGB = 2 };
    enum class GainAuto { Off = 0, Once = 1, Continuous = 2 };
    enum class BalanceRatioSelector { Red = 0, Green = 1, Blue = 2 };
    enum class TriggerMode { Off = 0, On = 1 };
    enum class PixelFormat : int {
        Mono10 = 0x01100003,
        Mono10Packed = 0x010C0004,
        Mono12 = 0x01100005,
        Mono12Packed = 0x010C0006,
        Mono16 = 0x01100007,
        RGB8Packed = 0x02180014,
        YUV422_8 = 0x02100032,
        YUV422_8_UYVY = 0x0210001F,
        BayerGR8 = 0x01080008,
        BayerRG8 = 0x01080009,
        BayerGB8 = 0x0108000A,
        BayerBG8 = 0x0108000B,
        BayerGB10 = 0x0110000e,
        BayerGB12 = 0x01100012,
        BayerGB12Packed = 0x010C002C,
    };

    static void exceptionHandler(unsigned int msg_type, void* user);
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
    int grab_timeout_;
    bool auto_white_balance_;
    std::array<unsigned int, 3> balance_ratio_;
    void* handle_ = nullptr;
    std::atomic_bool exception_flag_ = false;
    MV_FRAME_OUT* frame_out_ = nullptr;
    MV_CC_DEVICE_INFO* device_info_ = nullptr;
    mutable std::shared_mutex mutex_;
};

}  // namespace radar::camera