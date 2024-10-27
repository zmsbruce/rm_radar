/**
 * @file preparam.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file defines the preprocess parameters which are generated in
 * preprocessing to store image transformation and used in postprocessing to
 * restore the image.
 * @date 2024-04-11
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <cmath>
#include <opencv2/opencv.hpp>

namespace radar::detect {

/**
 * @brief Parameters obtained by preprocessing, will be used in postprocessing
 *
 */
struct PreParam {
    PreParam() = default;

    /**
     * @brief Constructs the class with parameters directly.
     *
     * @param width The width of input image.
     * @param height The height of input image.
     * @param ratio The scale factor from input to output.
     * @param dw The padding width of one side after scaling.
     * @param dh The padding height of one side after scaling.
     */
    PreParam(float width, float height, float ratio, float dw, float dh)
        : width{width}, height{height}, ratio{ratio}, dw{dw}, dh{dh} {}

    /**
     * @brief Constructs the class using the size of input and output image.
     *
     * @param input The size of input image.
     * @param output The size of output image.
     */
    PreParam(cv::Size input, cv::Size output) {
        height = static_cast<float>(input.height);
        width = static_cast<float>(input.width);
        ratio = 1 / (std::min(output.height / height, output.width / width));
        dw = (output.width - std::round(width / ratio)) * 0.5f;
        dh = (output.height - std::round(height / ratio)) * 0.5f;
    }

    float width;
    float height;
    float ratio;
    float dw;
    float dh;
};

}  // namespace radar::detect