/**
 * @file tensor.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file implements the Tensor class including its name, size and
 * allocation of device memory.
 * @date 2024-04-11
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>

#include "common.h"

namespace radar::detect {

/**
 * @brief A class representing a tensor in CUDA memory.
 *
 */
class Tensor {
   public:
    Tensor() : device_ptr_{nullptr} {}
    Tensor(Tensor&& rhs) : name_{rhs.name_}, device_ptr_{rhs.device_ptr_} {
        // Prevent repeated release of device resources
        rhs.device_ptr_ = nullptr;
    }
    Tensor& operator=(Tensor&& rhs) {
        if (this != &rhs) {
            dims_ = rhs.dims_;
            name_ = rhs.name_;
            device_ptr_ = rhs.device_ptr_;
            // Prevent repeated release of device resources
            rhs.device_ptr_ = nullptr;
        }
        return *this;
    }
    ~Tensor() {
        if (device_ptr_) {
            try {
                CUDA_CHECK(cudaFree(device_ptr_));
            } catch (std::runtime_error& err) {
                std::cerr << err.what() << std::endl;
            }
        }
    }

    /**
     * @brief Constructs a Tensor object with the given dimensions, data type,
     * name, and maximum batch size.
     *
     * @param dims The dimensions of the tensor.
     * @param dtype The data type of the tensor.
     * @param name The name of the tensor.
     * @param max_batch_size The maximum batch size for the tensor.
     */
    explicit Tensor(const nvinfer1::Dims& dims, nvinfer1::DataType dtype,
                    const char* name, int max_batch_size)
        : name_{name}, dims_{dims} {
        if (dims.d[0] != -1 && dims.d[0] != max_batch_size) {
            throw std::logic_error("invalid dims");
        }
        // Start with dims.d[1] because dims.d[0] is -1 in dynamic network.
        auto dim_size{std::accumulate(dims.d + 1, dims.d + dims.nbDims, 1,
                                      std::multiplies<int32_t>())};
        CUDA_CHECK(cudaMalloc(
            &device_ptr_, dim_size * sizeOfDataType(dtype) * max_batch_size));
    }

    /**
     * @brief Get the name of the tensor.
     *
     * @return The name of the tensor.
     */
    inline const char* name() const noexcept { return name_.data(); }

    /**
     * @brief Get the device pointer of the tensor.
     *
     * @return The device pointer of the tensor.
     */
    inline void* data() const noexcept { return device_ptr_; }

    /**
     * @brief Get the dimensions of the tensor.
     *
     * @return The dimensions of the tensor.
     */
    inline nvinfer1::Dims dims() const noexcept { return dims_; }

   private:
    /**
     * @brief The Tensor class does not allow copying and copy assignment,
     * because the existence of two same device pointers at the same time will
     * cause two cudaFrees in one specific address during destruction.
     *
     */
    Tensor(const Tensor& rhs) = delete;

    /**
     * @brief The Tensor class does not allow copying and copy assignment,
     * because the existence of two same device pointers at the same time will
     * cause two cudaFrees in one specific address during destruction.
     *
     */
    Tensor& operator=(const Tensor& rhs) = delete;
    std::string_view name_;
    nvinfer1::Dims dims_;
    void* device_ptr_;
};

}  // namespace radar::detect