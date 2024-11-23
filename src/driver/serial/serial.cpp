/**
 * @file serial.cpp
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the implementation of the `Serial` class, which
 * provides methods to manage serial port operations such as opening, closing,
 * reading, and writing. The class ensures thread safety via mutex locks to
 * prevent concurrent access issues.
 * @date 2024-11-02
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#include "serial.h"

#include <spdlog/spdlog.h>

namespace radar::serial {

Serial::Serial(std::string_view device_name, LibSerial::BaudRate baud_rate)
    : device_name_(device_name), baud_rate_(baud_rate) {
    spdlog::debug("Serial parameters: device name: {}, baud rate: {}",
                  device_name_,
                  (baud_rate_ == LibSerial::BaudRate::BAUD_115200) ? "115200"
                                                                   : "unknown");
    spdlog::trace("Serial {} initialized.", device_name_);
}

Serial::~Serial() {
    spdlog::trace("Destroying serial {}", device_name_);

    // Lock to ensure thread safety during destruction.
    std::lock_guard<std::mutex> lock(serial_mutex_);
    spdlog::debug("Serial {} lock acquired.", device_name_);

    // Check if the serial port is still open and attempt to close it.
    if (isOpen()) {
        spdlog::debug("Serial {} is still open, attempting to close it.",
                      device_name_);
        try {
            close();  // Close the serial port if still open.
        } catch (...) {
            spdlog::error(
                "Unknown exception caught while closing serial {} during "
                "destruction.",
                device_name_);
        }
    }
    spdlog::trace("Serial {} deconstructed.", device_name_);
}

inline bool Serial::isOpen() const noexcept { return is_open_; }

bool Serial::open() {
    std::lock_guard<std::mutex> lock(serial_mutex_);
    spdlog::debug("Serial {} lock acquired.", device_name_);

    // Check if the serial port is already open.
    if (isOpen()) {
        spdlog::error("Serial {} is already open.", device_name_);
        return false;
    }

    try {
        // Open the serial port and set the baud rate.
        serial_port_.Open(device_name_);
        // 波特率115200, 8 位数据位，1 位停止位，无硬件流控，无校验位。
        serial_port_.SetBaudRate(baud_rate_);
        serial_port_.SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8);
        serial_port_.SetFlowControl(
            LibSerial::FlowControl::FLOW_CONTROL_DEFAULT);
        serial_port_.SetParity(LibSerial::Parity::PARITY_DEFAULT);
        serial_port_.SetStopBits(LibSerial::StopBits::STOP_BITS_DEFAULT);

        is_open_ = true;  // Mark the serial port as open.
        spdlog::info("Serial {} opened with baud rate: {}.", device_name_,
                     (baud_rate_ == LibSerial::BaudRate::BAUD_115200)
                         ? "115200"
                         : "unknown");
        return true;
    } catch (const LibSerial::OpenFailed& err) {
        spdlog::error("Failed to open serial {}: {}", device_name_, err.what());
        return false;
    }
}

void Serial::close() {
    std::lock_guard<std::mutex> lock(serial_mutex_);
    spdlog::debug("Serial {} lock acquired.", device_name_);

    // Check if the serial port is already closed.
    if (!isOpen()) {
        spdlog::error("Serial {} is not open.", device_name_);
        return;
    }

    try {
        // Close the serial port and mark as not open.
        serial_port_.Close();
        is_open_ = false;

        spdlog::info("Serial {} closed.", device_name_);
    } catch (const std::exception& err) {
        spdlog::error("Failed to close serial {}: {}", device_name_,
                      err.what());
    }
}

bool Serial::read(std::vector<std::byte>& buffer) {
    if (!isOpen()) {
        spdlog::error("Attempted to read from closed serial {}", device_name_);
        return false;
    }

    try {
        spdlog::trace("Serial {} will read with buffer size: {}", device_name_,
                      buffer.size());

        // Prepare the buffer for reading data from the serial port.
        LibSerial::DataBuffer data_buffer(
            reinterpret_cast<uint8_t*>(buffer.data()),
            reinterpret_cast<uint8_t*>(buffer.data()) + buffer.size());

        serial_port_.Read(data_buffer, buffer.size());  // Read into the buffer.

        buffer.assign(reinterpret_cast<std::byte*>(data_buffer.data()),
                      reinterpret_cast<std::byte*>(data_buffer.data()) +
                          data_buffer.size());

        spdlog::debug("Buffer read successfully from serial {}", device_name_);

        return true;
    } catch (const std::exception& err) {
        spdlog::error("Failed to read from serial {}: {}", device_name_,
                      err.what());
        return false;
    }
}

bool Serial::write(const std::span<const std::byte> data) {
    if (!isOpen()) {
        spdlog::error("Attempted to write to closed serial {}", device_name_);
        return false;
    }

    try {
        spdlog::trace("Serial {} will write with data size: {}", device_name_,
                      data.size());

        // Prepare the data buffer for writing to the serial port.
        auto data_ptr = reinterpret_cast<const uint8_t*>(data.data());
        LibSerial::DataBuffer data_buffer(data_ptr, data_ptr + data.size());

        serial_port_.Write(data_buffer);  // Write the data to the serial port.
        spdlog::debug("Data wrote successfully to serial {}", device_name_);
        return true;
    } catch (const std::exception& err) {
        spdlog::error("Failed to write to serial {}: {}", device_name_,
                      err.what());
        return false;
    }
}

}  // namespace radar::serial