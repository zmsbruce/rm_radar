/**
 * @file sentry.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief Definition of the Sentry Protocol for radar communication.
 * @date 2024-11-03
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <cstdint>

// Use no padding bytes to shrink size
#pragma pack(1)

namespace radar::protocol {

/**
 * @brief A structure representing the Sentry Protocol for radar communication.
 *
 * This structure holds the 3D positional data (x, y, z coordinates) for various
 * units in both the red and blue teams, including heroes, engineers, infantry,
 * and sentries. The structure is packed to ensure no padding is added between
 * members, and its size is asserted to be less than 112 bytes.
 */
struct SentryProtocol {
    /**
     * @brief Constructor for the SentryProtocol struct.
     *
     * The constructor includes a static assertion to ensure that the size of
     * the structure is less than 112 bytes. If the assertion fails, it
     * indicates that the structure has grown beyond the specified limit.
     */
    SentryProtocol() {
        static_assert(sizeof(SentryProtocol) < 112,
                      "Size of sentry protocol out of range");
    }

    /// @brief Red team's hero position (x coordinate).
    uint16_t red_hero_x;
    /// @brief Red team's hero position (y coordinate).
    uint16_t red_hero_y;
    /// @brief Red team's hero position (z coordinate).
    uint16_t red_hero_z;

    /// @brief Red team's engineer position (x coordinate).
    uint16_t red_engineer_x;
    /// @brief Red team's engineer position (y coordinate).
    uint16_t red_engineer_y;
    /// @brief Red team's engineer position (z coordinate).
    uint16_t red_engineer_z;

    /// @brief Red team's infantry 3 position (x coordinate).
    uint16_t red_infantry_3_x;
    /// @brief Red team's infantry 3 position (y coordinate).
    uint16_t red_infantry_3_y;
    /// @brief Red team's infantry 3 position (z coordinate).
    uint16_t red_infantry_3_z;

    /// @brief Red team's infantry 4 position (x coordinate).
    uint16_t red_infantry_4_x;
    /// @brief Red team's infantry 4 position (y coordinate).
    uint16_t red_infantry_4_y;
    /// @brief Red team's infantry 4 position (z coordinate).
    uint16_t red_infantry_4_z;

    /// @brief Red team's infantry 5 position (x coordinate).
    uint16_t red_infantry_5_x;
    /// @brief Red team's infantry 5 position (y coordinate).
    uint16_t red_infantry_5_y;
    /// @brief Red team's infantry 5 position (z coordinate).
    uint16_t red_infantry_5_z;

    /// @brief Red team's sentry position (x coordinate).
    uint16_t red_sentry_x;
    /// @brief Red team's sentry position (y coordinate).
    uint16_t red_sentry_y;
    /// @brief Red team's sentry position (z coordinate).
    uint16_t red_sentry_z;

    /// @brief Blue team's hero position (x coordinate).
    uint16_t blue_hero_x;
    /// @brief Blue team's hero position (y coordinate).
    uint16_t blue_hero_y;
    /// @brief Blue team's hero position (z coordinate).
    uint16_t blue_hero_z;

    /// @brief Blue team's engineer position (x coordinate).
    uint16_t blue_engineer_x;
    /// @brief Blue team's engineer position (y coordinate).
    uint16_t blue_engineer_y;
    /// @brief Blue team's engineer position (z coordinate).
    uint16_t blue_engineer_z;

    /// @brief Blue team's infantry 3 position (x coordinate).
    uint16_t blue_infantry_3_x;
    /// @brief Blue team's infantry 3 position (y coordinate).
    uint16_t blue_infantry_3_y;
    /// @brief Blue team's infantry 3 position (z coordinate).
    uint16_t blue_infantry_3_z;

    /// @brief Blue team's infantry 4 position (x coordinate).
    uint16_t blue_infantry_4_x;
    /// @brief Blue team's infantry 4 position (y coordinate).
    uint16_t blue_infantry_4_y;
    /// @brief Blue team's infantry 4 position (z coordinate).
    uint16_t blue_infantry_4_z;

    /// @brief Blue team's infantry 5 position (x coordinate).
    uint16_t blue_infantry_5_x;
    /// @brief Blue team's infantry 5 position (y coordinate).
    uint16_t blue_infantry_5_y;
    /// @brief Blue team's infantry 5 position (z coordinate).
    uint16_t blue_infantry_5_z;

    /// @brief Blue team's sentry position (x coordinate).
    uint16_t blue_sentry_x;
    /// @brief Blue team's sentry position (y coordinate).
    uint16_t blue_sentry_y;
    /// @brief Blue team's sentry position (z coordinate).
    uint16_t blue_sentry_z;
};

// Restore default byte padding
#pragma pack()

}  // namespace radar::protocol