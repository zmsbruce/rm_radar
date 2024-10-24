/**
 * @file common.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This files implements several common-used macros and functions used in
 * detecting.
 * @date 2024-04-11
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <stdexcept>

/**
 * @brief Ensures that a CUDA call returns success.
 *
 * This macro wraps a CUDA API function call and checks its return value. If the
 * return value indicates that an error has occurred, it throws a
 * `std::runtime_error` with a message that includes the CUDA error string.
 *
 * @param call The CUDA API function call to check.
 * @throws std::runtime_error If the CUDA API call does not return cudaSuccess.
 * @note This macro is intended for use in functions that allow exceptions.
 */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        const cudaError_t error_code = call;                        \
        if (error_code != cudaSuccess) {                            \
            throw std::runtime_error(fmt::format(                   \
                "CUDA Error: {}", cudaGetErrorString(error_code))); \
        }                                                           \
    } while (0)

/**
 * @brief Ensures that a CUDA call returns success without throwing exceptions.
 *
 * Similar to `CUDA_CHECK`, this macro wraps a CUDA API function call and checks
 * its return value. However, if the return value indicates that an error has
 * occurred, it writes an error message to `std::cerr` and then calls
 * `std::abort` to terminate the program.
 *
 * @param call The CUDA API function call to check.
 *
 * @note This macro is intended for use in functions that do not allow
 * exceptions (e.g., noexcept).
 */
#define CUDA_CHECK_NOEXCEPT(call)                             \
    do {                                                      \
        const cudaError_t error_code = call;                  \
        if (error_code != cudaSuccess) {                      \
            spdlog::critical("CUDA Error: {}",                \
                             cudaGetErrorString(error_code)); \
            std::abort();                                     \
        }                                                     \
    } while (0)

namespace radar::detect {

/**
 * @brief Returns the size in bytes of the given data type.
 *
 * @param dataType The nvinfer1::DataType object.
 * @return The size in bytes of the data type.
 */
constexpr inline int sizeOfDataType(
    const nvinfer1::DataType& dataType) noexcept {
    switch (dataType) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kBOOL:
            return 1;
        default:
            return 0;
    }
}

}  // namespace radar::detect
