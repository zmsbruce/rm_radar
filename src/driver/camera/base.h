/**
 * @file base.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief Provides an abstract interface for cameras, including support for
 * color cameras and pixel format management.
 *
 * This file defines the `Camera` class, which represents a general-purpose
 * camera interface, and the `ColorCamera` class, which extends the `Camera`
 * interface to support color-specific functionalities such as white balance
 * control. The `PixelFormat` enum defines several common pixel formats
 * supported by the camera.
 * @date 2024-10-27
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <array>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <string>

namespace radar::camera {

/**
 * @brief Enum class representing different pixel formats supported by the
 * camera.
 */
enum class PixelFormat {
    GRAY,  ///< Grayscale format
    RGB,   ///< RGB format
    BGR,   ///< BGR format
    RGBA,  ///< RGBA format
    BGRA,  ///< BGRA format
    HSV,   ///< HSV format
    YUV,   ///< YUV format
};

/**
 * @brief Abstract base class representing a generic camera interface.
 */
class Camera {
   public:
    /**
     * @brief Virtual destructor for the Camera interface.
     */
    virtual ~Camera() = default;

    /**
     * @brief Opens the camera for use.
     * @return True if the camera was successfully opened, false otherwise.
     */
    virtual bool open() = 0;

    /**
     * @brief Closes the camera.
     * @return True if the camera was successfully closed, false otherwise.
     */
    virtual bool close() = 0;

    /**
     * @brief Reconnects the camera, typically used when the connection is lost.
     * @return True if the camera was successfully reconnected, false otherwise.
     */
    virtual bool reconnect() = 0;

    /**
     * @brief Starts capturing images from the camera.
     * @return True if capturing started successfully, false otherwise.
     */
    virtual bool startCapture() = 0;

    /**
     * @brief Stops capturing images from the camera.
     * @return True if capturing stopped successfully, false otherwise.
     */
    virtual bool stopCapture() = 0;

    /**
     * @brief Grabs a single image from the camera.
     * @param[out] image The OpenCV matrix (cv::Mat) where the captured image
     * will be stored.
     * @param[in] pixel_format The desired pixel format of the captured image.
     * @return True if the image was successfully grabbed, false otherwise.
     */
    virtual bool grabImage(cv::Mat& image,
                           PixelFormat pixel_format) noexcept = 0;

    /**
     * @brief Gets the camera's resolution (width and height).
     * @return A pair containing the width and height of the camera resolution.
     */
    virtual std::pair<int, int> getResolution() const = 0;

    /**
     * @brief Sets the camera's resolution.
     * @param[in] width The desired width of the resolution.
     * @param[in] height The desired height of the resolution.
     * @return True if the resolution was successfully set, false otherwise.
     */
    virtual bool setResolution(int width, int height) = 0;

    /**
     * @brief Gets the current exposure time of the camera.
     * @return The exposure time in microseconds.
     */
    virtual int getExposureTime() const = 0;

    /**
     * @brief Sets the camera's exposure time.
     * @param[in] exposure_time The desired exposure time in microseconds.
     * @return True if the exposure time was successfully set, false otherwise.
     */
    virtual bool setExposureTime(int exposure_time) = 0;

    /**
     * @brief Gets the current gain value of the camera.
     * @return The gain value.
     */
    virtual float getGain() const = 0;

    /**
     * @brief Sets the camera's gain value.
     * @param[in] gain The desired gain value.
     * @return True if the gain was successfully set, false otherwise.
     */
    virtual bool setGain(float gain) = 0;

    /**
     * @brief Gets information about the camera (e.g., model, serial number).
     * @return A string containing the camera information.
     */
    virtual std::string getCameraInfo() const = 0;

    /**
     * @brief Checks if the camera is currently open.
     * @return True if the camera is open, false otherwise.
     */
    virtual bool isOpen() const = 0;

    /**
     * @brief Checks if the camera is currently capturing.
     * @return True if the camera is capturing, false otherwise.
     */
    virtual bool isCapturing() const = 0;

   protected:
    std::atomic_bool is_open_ =
        false;  ///< Atomic flag indicating if the camera is open.
    std::atomic_bool is_capturing_ =
        false;  ///< Atomic flag indicating if the camera is capturing.
};

/**
 * @brief Abstract class representing a color camera interface.
 */
class ColorCamera : public Camera {
   public:
    /**
     * @brief Sets whether the automatic white balance (balance ratio) is
     * enabled.
     * @param[in] balance_auto Set to true to enable automatic balance ratio,
     * false to disable.
     * @return True if successfully set, false otherwise.
     */
    virtual bool setBalanceRatioAuto(bool balance_auto) = 0;

    /**
     * @brief Checks whether the automatic white balance (balance ratio) is
     * enabled.
     * @return True if automatic balance ratio is enabled, false otherwise.
     */
    virtual bool getBalanceRatioAuto() const = 0;

    /**
     * @brief Gets the current white balance ratio for the red, green, and blue
     * channels.
     * @return An array of three unsigned integers representing the balance
     * ratio for the red, green, and blue channels.
     */
    virtual std::array<unsigned int, 3> getBalanceRatio() const = 0;

    /**
     * @brief Sets the white balance ratio for the red, green, and blue
     * channels.
     * @param[in] balance An array containing the balance ratio for the red,
     * green, and blue channels.
     * @return True if the balance ratios were successfully set, false
     * otherwise.
     */
    virtual bool setBalanceRatio(std::array<unsigned int, 3>&& balance) = 0;
};

}  // namespace radar::camera