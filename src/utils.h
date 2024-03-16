/**
 * @file utils.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief A utility header file providing common utility functions.
 * @date 2024-03-08
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace radar {

/**
 * @brief Macro that measures the time taken by an operation.
 *
 */
#define TIME_IT(operation)                                              \
    do {                                                                \
        auto start_time = std::chrono::high_resolution_clock::now();    \
        operation;                                                      \
        auto end_time = std::chrono::high_resolution_clock::now();      \
        std::chrono::duration<double> duration = end_time - start_time; \
        std::cout << std::setprecision(2) << "Operation took "          \
                  << duration.count() * 1000 << " ms." << std::endl;    \
    } while (0)

/**
 * @brief Overloaded insertion operator for std::vector.
 *
 * This function inserts
 * the elements of the vector into the output stream within square brackets.
 *
 * @tparam T The type of elements in the vector.
 * @param os The output stream.
 * @param vec The vector to be inserted.
 * @return The output stream after inserting the vector.
 */
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec.at(i);
        if (i < vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief  Overloaded insertion operator for std::map.
 *
 * This function inserts the key-value pairs of the map into the output stream
 * in the format: "Key: value".
 *
 * @tparam K The type of keys in the map.
 * @tparam V The type of values in the map.
 * @param os The output stream.
 * @param map The map to be inserted.
 * @return The output stream after inserting the map.
 */
template <typename K, typename V>
inline std::ostream& operator<<(std::ostream& os, const std::map<K, V>& map) {
    os << "{\n";
    for (const auto& pair : map) {
        os << "Key: " << pair.first << ", value: " << pair.second << "\n";
    }
    os << "}";
    return os;
}

/**
 * @brief Overloaded insertion operator for std::unordered_map.
 * This function inserts the key-value pairs of the unordered_map into the
 * output stream in the format: "Key: value".
 *
 * @tparam K The type of keys in the unordered_map.
 * @tparam V The type of values in the unordered_map.
 * @param os The output stream.
 * @param map The unordered_map to be inserted.
 * @return The output stream after inserting the unordered_map.
 */
template <typename K, typename V>
inline std::ostream& operator<<(std::ostream& os,
                                const std::unordered_map<K, V>& map) {
    os << "{\n";
    for (const auto& pair : map) {
        os << "Key: " << pair.first << ", value: " << pair.second << "\n";
    }
    os << "}";
    return os;
}

inline void writeToFile(std::span<const char> data, std::string_view path) {
    std::ofstream ofs(path.data(), std::ios::out | std::ios::binary);
    ofs.exceptions(ofs.failbit | ofs.badbit);
    ofs.write(data.data(), data.size());
    ofs.close();
}

inline std::pair<std::shared_ptr<char[]>, size_t> loadFromFile(
    std::string_view path) {
    std::ifstream ifs{path.data(), std::ios::binary};
    ifs.exceptions(ifs.failbit | ifs.badbit);
    auto pbuf = ifs.rdbuf();
    auto size = static_cast<size_t>(pbuf->pubseekoff(0, ifs.end, ifs.in));
    pbuf->pubseekpos(0, ifs.in);
    std::shared_ptr<char[]> buffer{new char[size]};
    pbuf->sgetn(buffer.get(), size);
    ifs.close();
    return std::make_pair(buffer, size);
}

}  // namespace radar