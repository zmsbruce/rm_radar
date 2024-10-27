#pragma once

#include <array>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <string>

namespace radar::camera {

enum class PixelFormat {
    GRAY,
    RGB,
    BGR,
    RGBA,
    BGRA,
    HSV,
    YUV,
};

class Camera {
   public:
    virtual ~Camera() = default;
    virtual bool open() = 0;
    virtual bool close() = 0;
    virtual bool reconnect() = 0;
    virtual bool startCapture() = 0;
    virtual bool stopCapture() = 0;
    virtual bool grabImage(cv::Mat& image,
                           PixelFormat pixel_format) noexcept = 0;
    virtual std::pair<int, int> getResolution() const = 0;
    virtual bool setResolution(int width, int height) = 0;
    virtual int getExposureTime() const = 0;
    virtual bool setExposureTime(int exposure_time) = 0;
    virtual float getGain() const = 0;
    virtual bool setGain(float gain) = 0;
    virtual std::string getCameraInfo() const = 0;
    virtual bool isOpen() const = 0;
    virtual bool isCapturing() const = 0;

   protected:
    std::atomic_bool is_open_ = false;
    std::atomic_bool is_capturing_ = false;
};

class ColorCamera : public Camera {
   public:
    virtual bool setBalanceRatioAuto(bool balance_auto) = 0;
    virtual bool getBalanceRatioAuto() const = 0;
    virtual std::array<unsigned int, 3> getBalanceRatio() const = 0;
    virtual bool setBalanceRatio(std::array<unsigned int, 3>&& balance) = 0;
};

}  // namespace radar::camera