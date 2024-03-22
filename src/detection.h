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

    /**
     * @brief Stream insertion operator for the Detection structure.
     *
     * This operator overloads the insertion operator to allow for easy
     * streaming of a Detection structure's content to an output stream. It
     * formats the output as a JSON-like string containing the fields of the
     * Detection structure.
     *
     * @param os Reference to the output stream to which the content is to be
     * streamed.
     * @param detection The Detection structure instance to be printed out.
     * @return A reference to the output stream to allow for chaining of stream
     * insertions.
     */
    friend std::ostream& operator<<(std::ostream& os,
                                    const Detection& detection) {
        os << "{ x: " << detection.x << ", y: " << detection.y
           << ", width: " << detection.width << ", height: " << detection.height
           << ", label: " << detection.label
           << ", confidence: " << detection.confidence << " }";
        return os;
    }

    float x;
    float y;
    float width;
    float height;
    float label;
    float confidence;
};

}  // namespace radar