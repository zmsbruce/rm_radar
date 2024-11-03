/**
 * @file referee_system.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief Definitions of referee system serial port protocol
 * @ref
 * https://rm-static.djicdn.com/tem/71710/RoboMaster%20%E8%A3%81%E5%88%A4%E7%B3%BB%E7%BB%9F%E4%B8%B2%E5%8F%A3%E5%8D%8F%E8%AE%AE%E9%99%84%E5%BD%95%20V1.6.3%EF%BC%8820240527%EF%BC%89.pdf
 * @date 2024-11-03
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <cstdint>

namespace radar::protocol {

/**
 * @brief Command code IDs.
 *
 */
enum class CommandCode {
    GameStatus = 0x0001,
    GameResult = 0x0002,
    GameRobotHP = 0x0003,
    Event = 0x0101,
    SupplyProjectileAction = 0x0102,
    RefereeWarning = 0x0104,
    DartInfo = 0x0105,
    RobotStatus = 0x0201,
    RadarMark = 0x020C,
    RadarInfo = 0x020E,
    RobotInteraction = 0x0301,
    MapRobot = 0x0305,
    CustomInfo = 0x0308,
};

/**
 * @brief Sub-content IDs.
 *
 */
enum class SubContentId {
    RobotCommunication = 0x0201,
    RadarToSentry = 0x0209,
    RadarCommand = 0x0121,
    Sentry = 0x0207,  // User defined
};

/**
 * @brief Robot, player client and server IDs
 *
 */
enum class Id {
    HeroRed = 1,
    EngineerRed = 2,
    InfantryThreeRed = 3,
    InfantryFourRed = 4,
    InfantryFiveRed = 5,
    AerialRed = 6,
    SentryRed = 7,
    DartRed = 8,
    RadarRed = 9,
    HeroBlue = 101,
    EngineerBlue = 102,
    InfantryThreeBlue = 103,
    InfantryFourBlue = 104,
    InfantryFiveBlue = 105,
    AerialBlue = 106,
    SentryBlue = 107,
    DartBlue = 108,
    RadarBlue = 109,
    HeroRedPlayer = 0x0101,
    EngineerRedPlayer = 0x0102,
    InfantryThreeRedPlayer = 0x0103,
    InfantryFourRedPlayer = 0x0104,
    InfantryFiveRedPlayer = 0x0105,
    AerialRedPlayer = 0x0106,
    AerialBluePlayer = 0x016A,
    HeroBluePlayer = 0x0165,
    EngineerBluePlayer = 0x0166,
    InfantryThreeBluePlayer = 0x0167,
    InfantryFourBluePlayer = 0x0168,
    InfantryFiveBluePlayer = 0x0169,
    Server = 0x8080,
};

//! The protocol requires no padding bytes
#pragma pack(1)

/**
 * Command code: 0x0001
 *
 * Description: Competition status data
 *
 * Frequency: 1Hz
 *
 * Sender/Receiver: Server/All robots
 */
typedef struct {
    uint8_t game_type : 4;
    uint8_t game_progress : 4;
    uint16_t stage_remain_time;
    uint64_t SyncTimeStamp;
} game_status_t;

/**
 * Command code: 0x0002
 *
 * Description: Competition result data
 *
 * Frequency: Unknown
 *
 * Sender/Receiver: Server/All robots
 */
typedef struct {
    uint8_t winner;
} game_result_t;

/**
 * Command code: 0x0003
 *
 * Description: Robot health data
 *
 * Frequency: 3Hz
 *
 * Sender/Receiver: Server/All robots
 */
typedef struct {
    uint16_t red_1_robot_HP;
    uint16_t red_2_robot_HP;
    uint16_t red_3_robot_HP;
    uint16_t red_4_robot_HP;
    uint16_t red_5_robot_HP;
    uint16_t red_7_robot_HP;
    uint16_t red_outpost_HP;
    uint16_t red_base_HP;
    uint16_t blue_1_robot_HP;
    uint16_t blue_2_robot_HP;
    uint16_t blue_3_robot_HP;
    uint16_t blue_4_robot_HP;
    uint16_t blue_5_robot_HP;
    uint16_t blue_7_robot_HP;
    uint16_t blue_outpost_HP;
    uint16_t blue_base_HP;
} game_robot_HP_t;

/**
 * Command code: 0x0101
 *
 * Description: Site event data
 *
 * Frequency: 1Hz
 *
 * Sender/Receiver: Server/All robots of the own side
 */
typedef struct {
    uint32_t event_data;
} event_data_t;

/**
 * Command code: 0x0102
 *
 * Description: Action identifier data of the Official Projectile Supplier
 *
 * Frequency: When the Official Projectile Supplier releases projectiles
 *
 * Sender/Receiver: Server/All robots of the own side
 */
typedef struct {
    uint8_t reserved;
    uint8_t supply_robot_id;
    uint8_t supply_projectile_step;
    uint8_t supply_projectile_num;
} ext_supply_projectile_action_t;

/**
 * Command code: 0x0104
 *
 * Description: Referee warning data
 *
 * Frequency: When one's team is issued a penalty/forfeiture and at a fixed
 * frequency of 1 Hz in other cases
 *
 * Sender/Receiver: Server/All robots of the penalized team
 */
typedef struct {
    uint8_t level;
    uint8_t offending_robot_id;
    uint8_t count;
} referee_warning_t;

/**
 * Command code: 0x0105
 *
 * Description: Dart launching data
 *
 * Frequency: 1Hz
 *
 * Sender/Receiver: Server/All robots of the own side
 */
typedef struct {
    uint8_t dart_remaining_time;
    uint8_t dart_aim_state;
} dart_info_t;

/**
 * Command code: 0x0201
 *
 * Description: Robot performance system data
 *
 * Frequency: 10Hz
 *
 * Sender/Receiver: Main Controller Module/Corresponding robot
 */
typedef struct {
    uint8_t robot_id;
    uint8_t robot_level;
    uint16_t current_HP;
    uint16_t maximum_HP;
    uint16_t shooter_barrel_cooling_value;
    uint16_t shooter_barrel_heat_limit;
    uint16_t chassis_power_limit;
    uint8_t power_management_gimbal_output : 1;
    uint8_t power_management_chassis_output : 1;
    uint8_t power_management_shooter_output : 1;
} robot_status_t;

/**
 * Command code: 0x020C
 *
 * Description: Radar-marked progress data
 *
 * Frequency: 1Hz
 *
 * Sender/Receiver: Server/Own side's Radar Robots
 */
typedef struct {
    uint8_t mark_hero_progress;
    uint8_t mark_engineer_progress;
    uint8_t mark_standard_3_progress;
    uint8_t mark_standard_4_progress;
    uint8_t mark_standard_5_progress;
    uint8_t mark_sentry_progress;
} radar_mark_data_t;

/**
 * Command code: 0x020E
 *
 * Description: Decision-making data of Radar
 *
 * Frequency: 1Hz
 *
 * Sender/Receiver: Server/Own side's Radar Robots
 */
typedef struct {
    uint8_t radar_info;
} radar_info_t;

/**
 * Command code: 0x0301
 *
 * Description: Robot interaction data
 *
 * Frequency: At a maximum frequency of 10 Hz when triggered by the sender
 *
 * Sender/Receiver: -
 */
typedef struct {
    uint16_t data_cmd_id;
    uint16_t sender_id;
    uint16_t receiver_id;
    uint8_t user_data[112];
} robot_interaction_data_t;

/**
 * Sub-content ID: 0x0121
 *
 * Description: Whether the Radar confirms to trigger the "double vulnerability"
 * effect
 */
typedef struct {
    uint8_t radar_cmd;
} radar_cmd_t;

/**
 * Command code: 0x0305
 *
 * Description: Radar data received by player clients' Small Maps
 *
 * Frequency: At a maximum frequency of 10 Hz
 *
 * Sender/Receiver: Radar/All player clients of the own side
 */
typedef struct {
    uint16_t hero_position_x;
    uint16_t hero_position_y;
    uint16_t engineer_position_x;
    uint16_t engineer_position_y;
    uint16_t infantry_3_position_x;
    uint16_t infantry_3_position_y;
    uint16_t infantry_4_position_x;
    uint16_t infantry_4_position_y;
    uint16_t infantry_5_position_x;
    uint16_t infantry_5_position_y;
    uint16_t sentry_position_x;
    uint16_t sentry_position_y;
} map_robot_data_t;

/**
 * Command code: 0x0308
 *
 * Description: Robot data received by player clients' Small Map
 *
 * Frequency: At a maximum frequency of 3 Hz
 *
 * Sender/Receiver: Radar/All player clients of the own side
 */
typedef struct {
    uint16_t sender_id;
    uint16_t receiver_id;
    uint8_t user_data[30];
} custom_info_t;

// Restore default byte padding
#pragma pack()

}  // namespace radar::protocol