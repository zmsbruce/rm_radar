/**
 * @file serial.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file declares the `Serial` class, which provides an abstraction
 * for managing serial port operations like opening, closing, reading, and
 * writing. The class ensures thread safety and allows for easy integration with
 * different baud rates.
 * @date 2024-11-02
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <libserial/SerialPort.h>

#include <atomic>
#include <mutex>
#include <span>
#include <string>
#include <string_view>

namespace radar::serial {

/**
 * @brief Serial class encapsulates the functionality for handling serial port
 *        operations such as open, close, read, and write.
 */
class Serial {
   public:
    /**
     * @brief Constructor for the Serial class.
     *
     * @param device_name The name of the serial device (e.g., "/dev/ttyUSB0").
     * @param baud_rate The baud rate for the serial communication (default is
     * 115200).
     */
    Serial(std::string_view device_name,
           LibSerial::BaudRate baud_rate = LibSerial::BaudRate::BAUD_115200);

    /**
     * @brief Destructor for the Serial class. Ensures the serial port is
     * closed.
     */
    ~Serial();

    /**
     * @brief Check if the serial port is open.
     *
     * @return true if the serial port is open, false otherwise.
     */
    inline bool isOpen() const noexcept;

    /**
     * @brief Opens the serial port with the specified device name and baud
     * rate.
     *
     * @return true if the serial port was successfully opened, false otherwise.
     */
    bool open();

    /**
     * @brief Closes the serial port if it is open.
     */
    void close();

    /**
     * @brief Reads data from the serial port into the provided buffer.
     *
     * @param buffer A span of bytes to store the read data.
     * @return true if data was successfully read, false otherwise.
     */
    bool read(std::span<std::byte> buffer);

    /**
     * @brief Writes data to the serial port.
     *
     * @param data A span of bytes containing the data to write.
     * @return true if data was successfully written, false otherwise.
     */
    bool write(const std::span<const std::byte> data);

   private:
    // Disable default constructor
    Serial() = delete;

    /// The serial port object from the LibSerial library.
    LibSerial::SerialPort serial_port_;

    /// The name of the serial device (e.g., "/dev/ttyUSB0").
    std::string device_name_;

    /// The baud rate for the serial communication.
    LibSerial::BaudRate baud_rate_;

    /// Flag indicating if the serial port is open.
    std::atomic_bool is_open_ = false;

    /// Mutex to ensure thread safety for serial port operations.
    std::mutex serial_mutex_;
};

}  // namespace radar::serial