/**
 * @file detection.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file contains the declaration of detection class and its
 * functions.
 * @date 2024-03-22
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <ostream>
#include <type_traits>

namespace radar {

/**
 * @brief Detection object with the x, y-coordinate, width, height, label and
 * confidence score.
 *
 */
struct Detection {
    Detection() {
        // Detection must be a standard layout type to support low-level memory
        // operations, e.g. Parsing results from cuda functions
        static_assert(std::is_standard_layout_v<Detection>,
                      "Detection must be a standard layout type");
    }
    Detection(float x, float y, float width, float height, float label,
              float confidence)
        : x{x},
          y{y},
          width{width},
          height{height},
          label{label},
          confidence{confidence} {}

    float x;
    float y;
    float width;
    float height;
    float label;
    float confidence;
};

}  // namespace radar