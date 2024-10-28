#pragma once

#include <libuvc/libuvc.h>

#include <array>
#include <opencv2/opencv.hpp>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <vector>

#include "driver/camera/base.h"

namespace radar::camera {

struct UvcFormat {
    std::array<uint8_t, 16> guid;
    int width;
    int height;
    int fps;
};

class UvcCamera : public ColorCamera {
   public:
    /**
     * @brief Deleted default constructor.
     */
    UvcCamera() = delete;
    UvcCamera(int vendor_id, int product_id, std::string_view serial_number,
              std::string_view pixel_format = "default", int width = -1,
              int height = -1, int fps = -1, int exposure = -1, int gamma = -1,
              int gain = -1, bool auto_white_balance = true,
              std::array<unsigned int, 3>&& balance_ratio = {0, 0, 0});
    ~UvcCamera() override;
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

   private:
    enum class UvcPixelFormat {
        YUYV = UVC_FRAME_FORMAT_YUYV,
        MJPEG = UVC_FRAME_FORMAT_MJPEG,
        UYVY = UVC_FRAME_FORMAT_UYVY,
        RGB24 = UVC_FRAME_FORMAT_RGB,
        Unknown = UVC_FRAME_FORMAT_UNKNOWN,
    };

    static void frameCallback(uvc_frame_t* frame, void* userdata);
    std::span<UvcFormat> getSupportedFormats();
    bool setFormat(std::span<UvcFormat> supported_formats);
    bool setBalanceRatioInner();
    bool setExposureInner();
    bool setGammaInner();
    bool setGainInner();
    static UvcPixelFormat getPixelFormatFromGuid(
        const std::array<uint8_t, 16>& guid);
    void convertPixelFormat(cv::Mat& image, PixelFormat format);

    int vendor_id_;
    int product_id_;
    std::string serial_number_;
    UvcPixelFormat pixel_format_ = UvcPixelFormat::Unknown;
    int width_;
    int height_;
    int fps_;
    bool auto_white_balance_;
    std::array<unsigned int, 3> balance_ratio_;
    int exposure_;
    int gamma_;
    int gain_;
    uvc_context_t* context_ = nullptr;
    uvc_device_t* device_ = nullptr;
    uvc_device_handle_t* device_handle_ = nullptr;
    uvc_stream_ctrl_t* stream_ctrl_ = nullptr;
    std::vector<UvcFormat> supported_formats_;
    cv::Mat image_;
    std::shared_mutex image_mutex_;
};

}  // namespace radar::camera