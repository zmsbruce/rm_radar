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
#include <optional>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <thread>

#include "driver/camera/camera.h"

/**
 * @brief Macro to execute a function and check its return value, returning
 * `false` on failure.
 *
 * This macro executes the function `func` with the provided arguments
 * (`__VA_ARGS__`), checks if the return value is not equal to `MV_OK`, and logs
 * an error message if the function fails. If the function fails, it returns
 * `false` from the calling function.
 *
 * @param func The function to be executed.
 * @param msg The message to be included in the error log.
 * @param ... The arguments to be passed to the function `func`.
 *
 * @note The error message is logged using `spdlog::error`, and the error code
 * is formatted as a hexadecimal number.
 *
 * @warning If `MV_OK != ret`, this macro will return `false` immediately from
 * the current function.
 */
#define HIK_CHECK_RETURN_BOOL(func, msg, ...)                     \
    do {                                                          \
        int ret = func(__VA_ARGS__);                              \
        if (MV_OK != ret) {                                       \
            spdlog::error("Failed to {}, error code: {:#x}", msg, \
                          static_cast<unsigned int>(ret));        \
            return false;                                         \
        }                                                         \
    } while (0)

/**
 * @brief Macro to execute a function and check its return value, logging an
 * error on failure.
 *
 * This macro executes the function `func` with the provided arguments
 * (`__VA_ARGS__`), checks if the return value is not equal to `MV_OK`, and logs
 * an error message if the function fails. Unlike `HIK_CHECK_RETURN_BOOL`, this
 * macro does not return a value and simply continues execution after logging
 * the error.
 *
 * @param func The function to be executed.
 * @param msg The message to be included in the error log.
 * @param ... The arguments to be passed to the function `func`.
 *
 * @note The error message is logged using `spdlog::error`, and the error code
 * is formatted as a hexadecimal number.
 */
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
 * @brief Enum class representing different pixel types supported by the
 * camera.
 */
enum class HikPixelFormat {
    Unknown,        // 0x0
    RGB8,           // 0x02180014
    BGR8,           // 0x02180014
    YUV422_8,       // 0x02100032
    YUV422_8_UYVY,  // 0x0210001F
    BayerGR8,       // 0x01080008
    BayerRG8,       // 0x01080009
    BayerGB8,       // 0x0108000A
    BayerBG8        // 0x0108000B
};

/**
 * @brief Constant representing an unknown resolution in function
 * `HikCamera::getResolution()`.
 *
 * This pair holds two integers where both values are set to 0, indicating that
 * the resolution is unknown or has not been specified.
 *
 * @note The first element of the pair represents the width, and the second
 * element represents the height.
 */
constexpr std::pair<int, int> UNKNOWN_RESOLUTION{0, 0};

/**
 * @brief Constant representing an unknown gain value in function
 * `HikCamera::getGain()`.
 *
 * This constant holds a floating-point value of `-1.0f`, which indicates that
 * the gain is unknown or has not been specified.
 */
constexpr float UNKNOWN_GAIN{-1.0f};

/**
 * @brief Constant representing an unknown exposure value in function
 * `HikCamera::getExposure()`.
 *
 * This constant holds an integer value of `-1`, which indicates that the
 * exposure duration is unknown or has not been specified.
 */
constexpr int UNKNOWN_EXPOSURE{-1};

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

    /**
     * @brief Constructor for the HikCamera class.
     *
     * This constructor initializes the camera with various optional parameters
     * such as resolution, exposure, gamma, gain, pixel format, balance ratio,
     * and a grab timeout. It also initializes device info and supported pixel
     * formats, and starts a background daemon thread for handling camera
     * operations.
     *
     * @param camera_sn The serial number of the camera.
     * @param resolution Optional resolution in the format of a pair (width,
     * height). If not provided, the default resolution is used.
     * @param exposure Optional exposure value. If not provided, the default
     * exposure is used.
     * @param gamma Optional gamma value. If not provided, the default gamma is
     * used.
     * @param gain Optional gain value. If not provided, the default gain is
     * used.
     * @param balance_ratio Optional white balance ratio as an array of three
     * unsigned integers. If not provided, the default balance ratio is used.
     * @param pixel_format Optional pixel format as a string view. If not
     * provided or invalid, the default or unknown pixel format is used.
     * @param grab_timeout Timeout for grabbing frames from the camera, in
     * milliseconds. Defaults to 1000ms.
     */
    HikCamera(
        std::string_view camera_sn,
        std::optional<std::pair<unsigned int, unsigned int>> resolution =
            std::nullopt,
        std::optional<float> exposure = std::nullopt,
        std::optional<float> gamma = std::nullopt,
        std::optional<float> gain = std::nullopt,
        std::optional<std::array<unsigned int, 3>> balance_ratio = std::nullopt,
        std::optional<std::string_view> pixel_format = std::nullopt,
        unsigned int grab_timeout = 1000);

    /**
     * @brief Destructor for the HikCamera class.
     *
     * This destructor handles the proper shutdown of the camera. It ensures
     * that if the camera is open, it is closed before the object is destroyed.
     * Additionally, it requests the stop of the daemon thread that manages the
     * camera.
     */
    ~HikCamera() override;

    /**
     * @brief pens the camera device, prepares it for capturing images, and
     * applies necessary configurations.
     *
     * @return `true` if the camera is opened successfully; `false` otherwise.
     *
     * @note This method is thread-safe and uses a mutex to ensure that only one
     * thread can open the camera at a time.
     *
     * @warning If the camera cannot be opened or configured properly, the
     * method returns `false` and logs critical errors.
     */
    bool open() override;

    /**
     * @brief Closes the camera device and releases any associated resources.
     *
     * This method performs the necessary steps to safely close the camera and
     * release the resources associated with it. If the camera is currently
     * capturing images, it stops the capture process before closing the device.
     *
     * @note This method is thread-safe and uses a mutex to prevent concurrent
     * operations on the camera while it is being closed.
     *
     * @warning If the camera is capturing, the method attempts to stop the
     * capture before proceeding with the closure. If stopping the capture
     * fails, an error is logged, but the method continues to close the camera.
     */
    void close() override;

    /**
     * @brief Attempts to reconnect the camera by closing and reopening it.
     *
     * This method closes the current connection to the camera, if one exists,
     * and then tries to reopen the camera. If the reconnection is successful,
     * the camera will be ready for use again.
     *
     * @return `true` if the camera is successfully reconnected; `false`
     * otherwise.
     *
     * @note The method logs a warning when attempting to reconnect and provides
     * detailed logging for the closing and reopening steps. If the reconnection
     * fails, an error message is logged.
     *
     * @warning If the camera cannot be reopened, the method returns `false` and
     * the camera remains disconnected.
     */
    bool reconnect() override;

    /**
     * @brief Starts capturing images from the camera.
     *
     * This method initiates the image capture process by calling the SDK
     * function to start grabbing frames. It ensures thread safety by acquiring
     * a lock before interacting with the camera's state. Once the capture
     * process has started successfully, the camera is marked as capturing.
     *
     * @return `true` if the capture is started successfully; `false` otherwise.
     *
     * @note The method uses a mutex to synchronize access to the camera,
     * ensuring that only one thread can start the capture process at a time.
     * Detailed logging is provided at various levels (`trace`, `debug`, `info`)
     * to trace the capture process.
     *
     * @warning If the capture process fails to start, an error message is
     * logged, and the method returns `false`.
     */
    bool startCapture() override;

    /**
     * @brief Stops capturing images from the camera.
     *
     * This method stops the image capture process by calling the SDK function
     * to stop grabbing frames. It ensures thread safety by acquiring a lock
     * before interacting with the camera's state. Once the capture process is
     * successfully stopped, the camera is marked as not capturing.
     *
     * @return `true` if the capture is stopped successfully; `false` otherwise.
     *
     * @note The method uses a mutex to synchronize access to the camera,
     * ensuring that only one thread can stop the capture process at a time.
     * Detailed logging is provided at various levels (`trace`, `debug`) to
     * trace the stopping process.
     *
     * @warning If the capture process fails to stop, an error message is
     * logged, and the method returns `false`.
     */
    bool stopCapture() override;

    /**
     * @brief Grabs an image from the camera and converts it to a specified
     * pixel format.
     *
     * This method captures a frame from the camera, converts it to a `cv::Mat`
     * object, and applies the specified pixel format. It ensures thread-safe
     * access to the camera by using a mutex and performs several checks to
     * ensure the camera is open and capturing before attempting to grab the
     * image.
     *
     * @param[out] image A reference to a `cv::Mat` object where the captured
     * image will be stored.
     * @param[in] pixel_format The desired pixel format for the output image.
     *
     * @return `true` if the image is successfully grabbed and converted;
     * `false` otherwise.
     *
     * @warning If any of the steps fail (e.g., the camera is not open, the
     * image buffer is invalid, or the conversion fails), the method logs an
     * error and returns `false`.
     *
     * @note The method ensures that the image buffer is freed after the
     * operation to prevent memory leaks.
     */
    bool grabImage(cv::Mat& image, camera::PixelFormat pixel_format =
                                       PixelFormat::BGR) noexcept override;

    /**
     * @brief Sets the pixel format for the camera.
     *
     * This method sets the camera's pixel format if the camera is open and not
     * currently capturing. It locks the camera's internal mutex to ensure
     * thread-safe access. If the pixel format is valid and supported, it is
     * applied via the `setPixelFormatInner` function.
     *
     * @param[in] pixel_format The desired pixel format to set for the camera.
     * @return `true` if the pixel format was successfully set; `false`
     * otherwise.
     *
     * @note The camera must be open and not capturing in order to set the pixel
     * format. If the camera is capturing or closed, the function returns
     * `false` and logs an error.
     */
    bool setPixelFormat(HikPixelFormat pixel_format);

    /**
     * @brief Retrieves the current resolution of the camera.
     *
     * This method returns the camera's resolution as a pair of integers
     * representing the width and height. If the resolution has already been
     * cached in the `resolution_` member variable, it is returned directly.
     * Otherwise, the method queries the resolution from the camera SDK.
     *
     * @return A `std::pair<int, int>` representing the width and height of the
     * camera resolution. If the resolution cannot be obtained, it returns the
     * `UNKNOWN_RESOLUTION` constant.
     *
     * @note The method uses a shared lock to ensure that the resolution can be
     * read safely in a multi-threaded environment.
     *
     * @warning If the camera is not open or if the SDK fails to retrieve the
     * resolution, the method returns `UNKNOWN_RESOLUTION` and logs an error
     * message.
     */
    std::pair<int, int> getResolution() const override;

    /**
     * @brief Sets the resolution of the camera.
     *
     * This method sets the camera's resolution to the specified width and
     * height. The new resolution is applied only if the camera is not currently
     * capturing. The resolution is updated both internally and through the
     * camera's SDK.
     *
     * @param[in] width The desired width of the camera resolution.
     * @param[in] height The desired height of the camera resolution.
     *
     * @return `true` if the resolution is successfully set; `false` otherwise.
     *
     * @note The method acquires a lock to ensure thread-safe access to the
     * camera's resolution state.
     *
     * @warning Resolution changes are not allowed while the camera is
     * capturing. If the camera is capturing, the method logs an error and
     * returns `false`.
     */
    bool setResolution(int width, int height) override;

    /**
     * @brief Retrieves the current gain of the camera.
     *
     * This method returns the camera's gain value. If the gain has already been
     * cached in the `gain_` member variable, it is returned directly.
     * Otherwise, the method queries the gain value from the camera SDK.
     *
     * @return The gain value as a `float`. If the gain cannot be obtained, the
     * method returns `UNKNOWN_GAIN`.
     *
     * @note The method ensures thread safety by acquiring a shared lock before
     * accessing the gain value.
     *
     * @warning If the camera is not open or if there is an error retrieving the
     * gain from the SDK, the method returns `UNKNOWN_GAIN` and logs an error
     * message.
     */
    float getGain() const override;

    /**
     * @brief Sets the gain of the camera.
     *
     * This method sets the camera's gain to the specified value. The gain value
     * is updated both internally and through the camera's SDK.
     *
     * @param[in] gain The desired gain value.
     *
     * @return `true` if the gain is successfully set; `false` otherwise.
     *
     * @note The method acquires a lock to ensure thread-safe access to the
     * camera's gain state.
     */
    bool setGain(float gain) override;

    /**
     * @brief Retrieves the current exposure time of the camera.
     *
     * This method returns the camera's exposure time in microseconds. If the
     * exposure time has already been cached in the `exposure_` member variable,
     * it is returned directly. Otherwise, the method queries the exposure time
     * from the camera SDK.
     *
     * @return The exposure time as an `int` in microseconds. If the exposure
     * time cannot be obtained, the method returns `UNKNOWN_EXPOSURE`.
     *
     * @note The method ensures thread safety by acquiring a shared lock before
     * accessing the exposure time.
     *
     * @warning If the camera is not open or if there is an error retrieving the
     * exposure time from the SDK, the method returns `UNKNOWN_EXPOSURE` and
     * logs an error message.
     */
    int getExposureTime() const override;

    /**
     * @brief Sets the exposure time of the camera.
     *
     * This method sets the camera's exposure time to the specified value in
     * microseconds. The exposure time is updated both internally and through
     * the camera's SDK.
     *
     * @param[in] exposure The desired exposure time in microseconds.
     *
     * @return `true` if the exposure time is successfully set; `false`
     * otherwise.
     *
     * @note The method acquires a lock to ensure thread-safe access to the
     * camera's exposure state.
     */
    bool setExposureTime(int exposure) override;

    /**
     * @brief Retrieves the current white balance ratio (red, green, blue) of
     * the camera.
     *
     * This method returns the camera's white balance ratio as an array of three
     * unsigned integers representing the red, green, and blue channel values.
     * If the balance ratio has already been cached in the `balance_ratio_`
     * member variable, it is returned directly. Otherwise, the method queries
     * the balance ratio values from the camera SDK.
     *
     * @return A `std::array<unsigned int, 3>` representing the red, green, and
     * blue balance ratios. If the balance ratio cannot be obtained, the method
     * returns `UNKNOWN_BALANCE_RATIO`.
     *
     * @note The method ensures thread safety by acquiring a shared lock before
     * accessing the balance ratio.
     *
     * @warning If the camera is not open or if there is an error retrieving the
     * balance ratio from the SDK, the method returns `UNKNOWN_BALANCE_RATIO`
     * and logs an error message.
     */
    std::array<unsigned int, 3> getBalanceRatio() const override;

    /**
     * @brief Sets the white balance ratio (red, green, blue) of the camera.
     *
     * This method sets the camera's white balance ratio for the red, green, and
     * blue channels to the specified values. The balance ratio is updated both
     * internally and through the camera's SDK.
     *
     * @param[in] red The desired red channel balance ratio.
     * @param[in] green The desired green channel balance ratio.
     * @param[in] blue The desired blue channel balance ratio.
     *
     * @return `true` if the balance ratio is successfully set; `false`
     * otherwise.
     *
     * @note The method acquires a lock to ensure thread-safe access to the
     * camera's balance ratio state.
     */
    bool setBalanceRatio(unsigned int red, unsigned int green,
                         unsigned int blue) override;

    /**
     * @brief Sets the white balance ratio (red, green, blue) of the camera.
     *
     * This method sets the camera's white balance ratio for the red, green, and
     * blue channels to the specified values. The balance ratio is updated both
     * internally and through the camera's SDK.
     *
     * @param[in] red The desired red channel balance ratio.
     * @param[in] green The desired green channel balance ratio.
     * @param[in] blue The desired blue channel balance ratio.
     *
     * @return `true` if the balance ratio is successfully set; `false`
     * otherwise.
     *
     * @note The method acquires a lock to ensure thread-safe access to the
     * camera's balance ratio state.
     */
    bool setBalanceRatioAuto() override;

    /**
     * @brief Checks if the camera's white balance ratio is set to automatic
     * mode.
     *
     * This method checks whether the white balance ratio is in automatic mode
     * by determining if the balance ratio has not been manually set (i.e., if
     * `balance_ratio_` has no value).
     *
     * @return `true` if the balance ratio is set to automatic mode; `false` if
     * it has been manually set.
     *
     * @note The method acquires a shared lock to ensure thread-safe access to
     * the camera's balance ratio state.
     */
    bool getBalanceRatioAuto() const override;

    /**
     * @brief Retrieves the camera's information as a string.
     *
     * This method returns the camera's information by querying the stored
     * `device_info_` and formatting it into a readable string. It ensures
     * thread-safe access by acquiring a shared lock before accessing the
     * camera's information.
     *
     * @return A `std::string` containing the camera's information.
     *
     * @note The method acquires a shared lock to ensure thread-safe access to
     * the camera's `device_info_`.
     */
    std::string getCameraInfo() const override;

    /**
     * @brief Retrieves the camera's serial number.
     *
     * This method returns the camera's serial number as a string. It ensures
     * thread-safe access by acquiring a shared lock before accessing the serial
     * number.
     *
     * @return A `std::string` containing the camera's serial number.
     *
     * @note The method acquires a shared lock to ensure thread-safe access to
     * the camera's serial number.
     */
    std::string getCameraSn() const;

    /**
     * @brief Internal function to apply the pixel format to the camera.
     *
     * This method attempts to set the camera's pixel format based on the
     * supported formats. If the current pixel format is supported, it is
     * retained. Otherwise, it selects a supported format from a list of
     * candidates.
     *
     * @return `true` if a valid pixel format was successfully applied; `false`
     * otherwise.
     *
     * @note The method checks if the current pixel format is supported by the
     * camera. If not, it iterates over a list of candidate formats and selects
     * one that is supported. If no supported format is found, the function logs
     * an error and returns `false`.
     */
    bool setPixelFormatInner();

    /**
     * @brief Sets the exception occurred state for the camera.
     *
     * This method updates the internal flag that indicates whether an exception
     * has occurred during camera operation. The state is logged and then
     * updated.
     *
     * @param[in] occurred A boolean indicating whether an exception has
     * occurred (`true`) or not (`false`).
     */
    void setExceptionOccurred(bool occurred);

    /**
     * @brief Gets the current value of the exception flag.
     * @return True if an exception has occurred, false otherwise.
     */
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
     * @brief Retrieves a list of available camera devices.
     *
     * This method returns a span of pointers to `MV_CC_DEVICE_INFO`,
     * representing the available devices enumerated by the SDK. It ensures that
     * the device info list is initialized only once using a `std::call_once`
     * mechanism.
     *
     * @return A `std::span<MV_CC_DEVICE_INFO*>` representing the list of
     * enumerated devices.
     *
     * @note The method initializes the device info list on the first call by
     * querying the camera SDK, storing the result in `device_info_list_`. It
     * logs the number of devices found and their details. The device list is
     * then cached for subsequent calls.
     */
    static std::span<MV_CC_DEVICE_INFO*> getDeviceInfoList();

    /**
     * @brief Retrieves detailed information about a specific camera device.
     *
     * This method formats and returns information about a camera device,
     * including its GUID, version, family name, manufacturer, model, serial
     * number, user-defined name, and vendor name. The information is extracted
     * from the provided `MV_CC_DEVICE_INFO` structure, which must represent a
     * USB device.
     *
     * @param[in] device_info A pointer to an `MV_CC_DEVICE_INFO` structure
     * containing the device information.
     *
     * @return A `std::string` containing the camera's detailed information.
     *
     * @warning The method asserts that the device type is `MV_USB_DEVICE`. If
     * the device type is incorrect, the program will terminate with an
     * assertion failure.
     */
    static std::string getCameraInfo(MV_CC_DEVICE_INFO* device_info);

    /**
     * @brief Handles exceptions that occur during camera operation.
     *
     * This method is invoked when an exception is detected by the camera SDK.
     * It logs the exception code and the camera's serial number, and sets the
     * internal exception state to `true`.
     *
     * @param[in] code The exception code returned by the SDK.
     * @param[in] user A pointer to the user data, which is expected to be a
     * `HikCamera` object.
     *
     * @note The method casts the `user` pointer to a `HikCamera*` and updates
     * the exception state of the corresponding camera.
     */
    static void exceptionHandler(unsigned int msg_type, void* user);

    /**
     * @brief Converts a `HikPixelFormat` enum value to its corresponding SDK
     * pixel format value.
     *
     * This method maps the `HikPixelFormat` enum to the corresponding
     * SDK-defined pixel format value.
     *
     * @param[in] format The `HikPixelFormat` enum value to be converted.
     *
     * @return The corresponding SDK pixel format value as an `unsigned int`. If
     * the format is not recognized, the method returns `0x0`.
     */
    static unsigned int getPixelFormatValue(HikPixelFormat format);

    /**
     * @brief Retrieves the list of pixel formats supported by the camera.
     *
     * This method queries the camera SDK to get the supported pixel formats and
     * returns them as a span of unsigned integers.
     *
     * @return A `std::span<unsigned int>` containing the supported pixel
     * formats. If the query fails, an empty span is returned.
     *
     * @note The method logs the supported pixel formats using the camera's
     * serial number. If the SDK function `MV_CC_GetPixelFormat` fails, an error
     * is logged, and an empty span is returned.
     */
    std::span<unsigned int> getSupportedPixelFormats();

    /**
     * @brief Converts the pixel format of an image.
     * @param image The image to convert.
     * @param format The desired pixel format.
     * @return The image converted.
     */
    cv::Mat convertPixelFormat(const cv::Mat& image, PixelFormat format);

    /**
     * @brief Starts a daemon thread to monitor the camera for exceptions and
     * attempt reconnection if needed.
     *
     * This method launches a daemon thread that periodically checks if an
     * exception has occurred in the camera. If an exception is detected, the
     * thread attempts to reconnect the camera. The check occurs every second.
     * The thread can be stopped using a cancellation token (`std::stop_token`).
     *
     * @note The daemon thread runs in a loop, sleeping for 1 second between
     * checks. If an exception is detected, it logs the event, attempts to
     * reconnect, and logs whether the reconnection was successful.
     *
     * @warning If the reconnection fails, the thread will wait and retry after
     * the next iteration. If the `stop_requested()` signal is triggered, the
     * thread will exit and log that it is stopping.
     */
    void startDaemonThread();

    /**
     * @brief Sets the resolution of the camera using the camera SDK.
     *
     * This method internally handles setting the camera's resolution based on
     * the current configuration. If the resolution is not set, or if the
     * requested resolution is out of the supported range, the camera's
     * resolution is set to the maximum allowable values. It also ensures that
     * the width and height are multiples of the camera's increment value
     * (`nInc`), adjusting them if necessary.
     *
     * @return `true` if the resolution is successfully set; `false` otherwise.
     *
     * @warning If any of the SDK calls fail, the method logs an error and
     * returns `false`.
     */
    bool setResolutionInner();

    /**
     * @brief Sets the white balance ratio of the camera using the camera SDK.
     *
     * This method configures the camera's white balance mode and ratios. If no
     * manual balance ratio is set, the method enables automatic white balance
     * in "Continuous" mode. If a manual balance ratio is configured, it checks
     * the provided red, green, and blue ratios against the camera's valid range
     * and applies them if they are valid.
     *
     * @return `true` if the balance ratio or auto white balance is successfully
     * set; `false` otherwise.
     *
     * @warning If any of the SDK calls fail, the method logs an error and
     * returns `false`.
     */
    bool setBalanceRatioInner();

    /**
     * @brief Sets the camera's exposure time using the camera SDK.
     *
     * This method configures the camera's exposure mode and time. If no manual
     * exposure time is set, the method switches to automatic exposure mode. If
     * a manual exposure time is provided, it checks whether the time falls
     * within the camera's valid range and applies it if valid. Otherwise, it
     * defaults back to automatic exposure mode.
     *
     * @return `true` if the exposure setting is successfully applied; `false`
     * otherwise.
     *
     * @warning If any of the SDK calls fail, the method logs an error and
     * returns `false`.
     */
    bool setExposureInner();

    /**
     * @brief Sets the camera's gamma correction using the camera SDK.
     *
     * This method configures the camera's gamma correction settings. If no
     * manual gamma value is set, gamma correction is disabled. If a gamma value
     * is provided, the method checks whether the value falls within the
     * camera's valid range and applies it if valid. If the gamma value is out
     * of range, gamma correction is disabled.
     *
     * @return `true` if the gamma setting is successfully applied; `false`
     * otherwise.
     *
     * @warning If any of the SDK calls fail, the method logs an error and
     * returns `false`.
     */
    bool setGammaInner();

    /**
     * @brief Sets the camera's gain using the camera SDK.
     *
     * This method configures the camera's gain settings. If no manual gain
     * value is set, the gain mode is switched to automatic. If a manual gain
     * value is provided, the method checks whether the value falls within the
     * camera's valid range and applies it if valid. If the gain value is out of
     * range, the method reverts to automatic gain mode.
     *
     * @return `true` if the gain setting is successfully applied; `false`
     * otherwise.
     *
     * @note The method performs the following steps:
     *   - If no manual gain value is set (`gain_` is not present), the gain
     * mode is set to "Continuous" auto mode.
     *   - If a manual gain value is set, the method retrieves the valid gain
     * range from the camera SDK.
     *   - If the provided gain value is outside the valid range, the method
     * logs an error and switches to automatic gain mode.
     *   - If the gain value is within the valid range, automatic gain is
     * disabled, and the manual gain value is applied using the SDK.
     *
     * @warning If any of the SDK calls fail, the method logs an error and
     * returns `false`.
     */
    bool setGainInner();

    /**
     * @brief The serial number of the camera.
     *
     * This string uniquely identifies the camera.
     */
    std::string camera_sn_;

    /**
     * @brief The camera's resolution as a pair of width and height.
     *
     * This mutable optional stores the resolution in terms of width and height.
     * If it is not set, the resolution is undefined.
     */
    mutable std::optional<std::pair<unsigned int, unsigned int>> resolution_;

    /**
     * @brief The camera's exposure time.
     *
     * This mutable optional stores the exposure time in microseconds. If not
     * set, the exposure time is undefined.
     */
    mutable std::optional<float> exposure_;

    /**
     * @brief The camera's gamma correction value.
     *
     * This optional stores the gamma value. If not set, gamma correction is
     * disabled.
     */
    std::optional<float> gamma_;

    /**
     * @brief The camera's gain value.
     *
     * This mutable optional stores the manual gain value. If not set, the gain
     * is either undefined or set to automatic mode.
     */
    mutable std::optional<float> gain_;

    /**
     * @brief The pixel format of the camera.
     *
     * This stores the current pixel format of the camera, initialized to
     * `HikPixelFormat::Unknown` by default.
     */
    HikPixelFormat pixel_format_ = HikPixelFormat::Unknown;

    /**
     * @brief The frame grab timeout value in milliseconds.
     *
     * This unsigned integer defines how long the camera waits when attempting
     * to grab a frame before timing out.
     */
    unsigned int grab_timeout_;

    /**
     * @brief The white balance ratio for the camera: Red, Green, Blue.
     *
     * This mutable optional stores the white balance ratios for the red, green,
     * and blue channels respectively. If not set, the camera uses automatic
     * white balance.
     */
    mutable std::optional<std::array<unsigned int, 3>> balance_ratio_;

    /**
     * @brief A raw pointer to the camera's SDK handle.
     *
     * This handle is used to interact with the camera through the SDK.
     */
    void* handle_ = nullptr;

    /**
     * @brief A flag indicating whether an exception has occurred in the camera.
     *
     * This atomic boolean is set to `true` when an exception occurs and `false`
     * otherwise.
     */
    std::atomic_bool exception_occurred_ = false;

    /**
     * @brief The current frame output data.
     *
     * This unique pointer holds the most recent frame output retrieved from the
     * camera.
     */
    std::unique_ptr<MV_FRAME_OUT> frame_out_ = nullptr;

    /**
     * @brief The device information for the camera.
     *
     * This unique pointer holds information about the camera's device, such as
     * model name, serial number, etc.
     */
    std::unique_ptr<MV_CC_DEVICE_INFO> device_info_ = nullptr;

    /**
     * @brief A shared mutex for ensuring thread-safe access to camera
     * properties.
     *
     * This mutex is used to synchronize access to certain mutable variables in
     * the camera class.
     */
    mutable std::shared_mutex mutex_;

    /**
     * @brief The supported pixel formats of the camera.
     *
     * This unique pointer holds the pixel formats that the camera supports,
     * retrieved from the SDK.
     */
    std::unique_ptr<MVCC_ENUMVALUE> supported_pixel_formats_;

    /**
     * @brief A static shared pointer to a list of available device information.
     *
     * This list contains the device information for all connected cameras,
     * shared across all instances of the class.
     */
    inline static std::shared_ptr<MV_CC_DEVICE_INFO_LIST> device_info_list_;

    /**
     * @brief A static flag to ensure the device info list is initialized only
     * once.
     *
     * This flag is used to control one-time initialization of the device
     * information list.
     */
    inline static std::once_flag is_device_info_list_init_;

    /**
     * @brief A thread that runs in the background to monitor exceptions and
     * handle reconnections.
     *
     * This `std::jthread` constantly checks for exceptions and attempts to
     * reconnect the camera when necessary.
     */
    std::jthread daemon_thread_;
};

}  // namespace radar::camera