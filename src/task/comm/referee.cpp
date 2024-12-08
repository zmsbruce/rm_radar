/**
 * @file crc.h
 * @author TianranW (529708894@qq.com)
 * @brief This file contains the implementation of the RefereeCommunicator
 * class, which provides methods for decoding and encoding link data, performs
 * full-duplex reading and writing using serial ports, and uses CRC codes to
 * determine the correctness of the data.
 * @date 2024-11-18
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#include "referee.h"

#include <spdlog/spdlog.h>

#include <magic_enum/magic_enum.hpp>

#include "crc.h"

namespace radar {

using namespace protocol;
using namespace serial;

constexpr size_t BUFFER_SIZE = 1024;

RefereeCommunicator::RefereeCommunicator(std::string_view serial_addr)
    : serial_(std::make_unique<Serial>(serial_addr,
                                       LibSerial::BaudRate::BAUD_115200)),
      game_status_(std::make_shared<game_status_t>()),
      game_result_(std::make_shared<game_result_t>()),
      robot_health_point_(std::make_shared<game_robot_HP_t>()),
      event_data_(std::make_shared<event_data_t>()),
      projectile_action_(std::make_shared<ext_supply_projectile_action_t>()),
      referee_warning_(std::make_shared<referee_warning_t>()),
      dart_info_(std::make_shared<dart_info_t>()),
      radar_status_(std::make_shared<robot_status_t>()),
      radar_mark_data_(std::make_shared<radar_mark_data_t>()),
      radar_info_(std::make_shared<radar_info_t>()),
      sentry_data_(std::make_shared<robot_interaction_data_t>()),
      data_buffer_(BUFFER_SIZE, std::byte(0)) {
    spdlog::debug("Initialize referee communicator with serial address: {}",
                  serial_addr);

    std::memset(game_status_.get(), 0, sizeof(game_status_t));
    std::memset(game_result_.get(), 0, sizeof(game_result_t));
    std::memset(robot_health_point_.get(), 0, sizeof(game_robot_HP_t));
    std::memset(event_data_.get(), 0, sizeof(event_data_t));
    std::memset(projectile_action_.get(), 0,
                sizeof(ext_supply_projectile_action_t));
    std::memset(referee_warning_.get(), 0, sizeof(referee_warning_t));
    std::memset(dart_info_.get(), 0, sizeof(dart_info_t));
    std::memset(radar_status_.get(), 0, sizeof(robot_status_t));
    std::memset(radar_mark_data_.get(), 0, sizeof(radar_mark_data_t));
    std::memset(radar_info_.get(), 0, sizeof(radar_info_t));
    std::memset(sentry_data_.get(), 0, sizeof(robot_interaction_data_t));

    this->is_connected_ = serial_->open();

    if (!this->is_connected_) {
        spdlog::error(
            "Failed to init RefereeCommunicator: serial port open failed");
    }

    spdlog::trace("Referee communicator initialized.");
}

bool RefereeCommunicator::reconnect() {
    this->is_connected_ = serial_->open();
    return this->is_connected_;
}

void RefereeCommunicator::sendMapRobot(const std::span<const Robot> robots) {
    spdlog::trace("Try to send map robot data to referee system.");
    map_robot_data_t data;
    std::memset(&data, 0, sizeof(data));

    auto filtered_robots =
        robots | std::ranges::views::filter([](const Robot& robot) {
            return robot.label().has_value() && robot.isLocated();
        });

    for (const auto& robot : filtered_robots) {
        std::optional<Robot::Label> label = robot.label();

        if (!isEnemy(label.value())) {  // 跳过友军
            spdlog::trace("Location of {} is skipped.",
                          magic_enum::enum_name(label.value()));
            continue;
        }

        spdlog::trace("Sending location of {} to referee system: ",
                      magic_enum::enum_name(label.value()));
        cv::Point3f point = robot.location().value();
        uint16_t x = static_cast<uint16_t>(point.x * 100);  // m->cm
        uint16_t y = static_cast<uint16_t>(point.y * 100);

        switch (label.value()) {
            case Robot::Label::BlueHero:
            case Robot::Label::RedHero: {
                data.hero_position_x = x;
                data.hero_position_y = y;
                break;
            }
            case Robot::Label::BlueEngineer:
            case Robot::Label::RedEngineer: {
                data.engineer_position_x = x;
                data.engineer_position_y = y;
                break;
            }
            case Robot::Label::BlueInfantryThree:
            case Robot::Label::RedInfantryThree: {
                data.infantry_3_position_x = x;
                data.infantry_3_position_y = y;
                break;
            }
            case Robot::Label::BlueInfantryFour:
            case Robot::Label::RedInfantryFour: {
                data.infantry_4_position_x = x;
                data.infantry_4_position_y = y;
                break;
            }
            case Robot::Label::BlueInfantryFive:
            case Robot::Label::RedInfantryFive: {
                data.infantry_5_position_x = x;
                data.infantry_5_position_y = y;
                break;
            }
            case Robot::Label::BlueSentry:
            case Robot::Label::RedSentry: {
                data.sentry_position_x = x;
                data.sentry_position_y = y;
                break;
            }
        }
    }

    std::vector<std::byte> sendData;
    const std::byte* start = reinterpret_cast<const std::byte*>(&data);
    const std::byte* end = reinterpret_cast<const std::byte*>(&data + 1);
    sendData.assign(start, end);

    if (!encode(CommandCode::MapRobot, std::move(sendData))) {
        spdlog::error("Failed to encode map robot data");
    };
    spdlog::debug("Send map robot data to referee.");
}

void RefereeCommunicator::sendToPlayer(Id id, std::u16string text) {
    spdlog::trace("Try to send custom message to player of robot with id {}.",
                  magic_enum::enum_name(id));
    std::vector<std::byte> sendData;
    uint16_t send_id = getRadarId();
    uint16_t receive_id = static_cast<uint16_t>(id);
    size_t size = text.length() > 15 ? 15 : text.size();

    sendData.resize(sizeof(custom_info_t));
    memcpy(sendData.data(), &send_id, sizeof(uint16_t));
    memcpy(sendData.data() + sizeof(uint16_t), &receive_id, sizeof(uint16_t));
    memcpy(sendData.data() + 2 * sizeof(uint16_t), text.data(),
           size * sizeof(uint16_t));

    if (!encode(CommandCode::CustomInfo, std::move(sendData))) {
        spdlog::error("Failed to encode custom info data");
    }
    spdlog::debug("Send custom info data to referee.");
}

void RefereeCommunicator::sendCommand() {
    spdlog::trace("Try to send radar command to referee system.");
    static auto lastCommandTime = std::chrono::steady_clock::now();
    static uint8_t radarCommand = 0;

    std::map<int, int> counter;
    for (int val : this->qualifications) {
        counter[val] += 1;
    }
    if (std::max_element(counter.begin(), counter.end(),
                         [](const auto& p1, const auto& p2) {
                             return p1.second < p2.second;
                         })
            ->first == 0) {
        spdlog::trace("No qualification to send radar command.");
        return;
    }

    auto nowTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                        nowTime - lastCommandTime)
                        .count();
    if (duration > 30) {
        if (radarCommand < 2) {
            spdlog::trace("Radar command is activated.");
            radarCommand += 1;
            lastCommandTime = nowTime;
        } else {
            spdlog::trace("Radar command times has been exhausted.");
        }
    }
    for (int i = 0; i < 10; ++i) {
        if (!encode(SubContentId::RadarCommand, Id::Server,
                    std::move(
                        std::vector<std::byte>(1, std::byte(radarCommand))))) {
            spdlog::error("Failed to encode radar command data");
        }
    }
    spdlog::debug("Send radar command to referee.");
}

void RefereeCommunicator::sendToSentry(const std::vector<Robot>& robots,
                                       bool dartWarning) {
    spdlog::trace("Try to send message to sentry.");
    constexpr int max_robot_num = 6;
    constexpr int element_size = 4;
    float mem[max_robot_num * element_size + 1] = {0};

    mem[0] = static_cast<float>(dartWarning);

    int offset = 0;
    for (const auto& robot : robots) {
        if (offset >= max_robot_num) {
            spdlog::trace("Number of robots out of range.");
            continue;
        }
        if (!robot.label().has_value() || !isEnemy(robot.label().value()) ||
            !robot.isLocated()) {
            continue;
        }
        auto location = robot.location().value();
        mem[element_size * offset + 1] =
            static_cast<float>(robot.label().value());
        mem[element_size * offset + 2] = location.x;
        mem[element_size * offset + 3] = location.y;
        mem[element_size * offset + 4] = location.z;
        offset += 1;
    }

    std::vector<std::byte> send_data(112, std::byte(0));
    assert(send_data.size() >= sizeof(mem));

    std::memcpy(send_data.data(), reinterpret_cast<uint8_t*>(mem), sizeof(mem));

    Id sentryId = getRadarId() == static_cast<uint8_t>(Id::RadarBlue)
                      ? Id::SentryRed
                      : Id::SentryBlue;
    if (!encode(SubContentId::RadarToSentry, sentryId, std::move(send_data))) {
        spdlog::error("Failed to encode sentry data");
    }
    spdlog::debug("Send sentry data to referee.");
}

void RefereeCommunicator::sendToRobot(Id robot_id, std::byte data) {
    spdlog::trace("Try to send message to robot with id {}.",
                  magic_enum::enum_name(robot_id));
    std::vector<std::byte> send_data;
    send_data.push_back(data);

    if (encode(SubContentId::RobotCommunication, robot_id,
               std::move(send_data))) {
        spdlog::error("Failed to encode robot communication data");
    }
    spdlog::debug("Send robot communication data to referee.");
}

void RefereeCommunicator::update() {
    if (!this->is_connected_) {
        spdlog::warn("Attempted to update with closed serial.");
        return;
    }
    if (this->serial_->read(this->data_buffer_)) {
        if (!decode()) {
            spdlog::warn("Failed to decode data from serial port.");
        }
        spdlog::debug("Read data from serial port.");
    } else {
        spdlog::error("Failed to read data from serial port.");
    }
}

uint8_t RefereeCommunicator::getRadarId() noexcept {
    std::unique_lock<std::shared_mutex> lock(comm_mutex_);
    spdlog::trace("Get radar ID: {}", radar_status_->robot_id);
    return radar_status_->robot_id;
}

bool RefereeCommunicator::encode(CommandCode cmd,
                                 std::vector<std::byte>&& data) noexcept {
    static const std::vector<CommandCode> CMD4SEND{
        CommandCode::RobotInteraction, CommandCode::MapRobot,
        CommandCode::CustomInfo};

    if (!this->is_connected_) {
        spdlog::warn("Attempted to encode with closed serial.");
        return false;
    } else if (std::find(CMD4SEND.begin(), CMD4SEND.end(), cmd) ==
               CMD4SEND.end()) {
        spdlog::warn("Request for encoding with unknown command.");
        return false;
    }

    spdlog::debug("Encoding data with command: {}", magic_enum::enum_name(cmd));

    int length = (data.size() + 15) * 4;
    std::vector<std::byte> buff(7, std::byte{0x00});
    buff[0] = std::byte{0xA5};
    buff[1] = std::byte{data.size()};
    buff[2] = std::byte{data.size() >> 8u};
    appendCRC8(std::span<std::byte>(buff.data(), 5));
    buff[5] = std::byte{static_cast<uint16_t>(cmd)};
    buff[6] = std::byte{static_cast<uint16_t>(cmd) >> 8u};
    buff.insert(buff.end(), std::make_move_iterator(data.begin()),
                std::make_move_iterator(data.end()));
    buff.resize(buff.size() + 2);
    appendCRC16(buff);
    return this->serial_->write(buff);
}

bool RefereeCommunicator::encode(SubContentId id, Id receiver,
                                 std::vector<std::byte>&& data) noexcept {
    static const std::vector<SubContentId> CONTENT4SEND{
        SubContentId::RobotCommunication, SubContentId::RadarCommand,
        SubContentId::RadarToSentry};

    if (!this->is_connected_) {
        spdlog::warn("Attempted to encode with closed serial.");
        return false;
    } else if (std::find(CONTENT4SEND.begin(), CONTENT4SEND.end(), id) ==
               CONTENT4SEND.end()) {
        spdlog::warn("Request for encoding with unknown subcommand.");
        return false;
    } else if (getRadarId() == 0) {
        spdlog::warn("Radar ID is not set.");
        return false;
    } else if (data.size() > 112) {
        spdlog::warn("Data size exceeds maximum limit.");
        return false;
    }

    spdlog::debug("Encoding data with subcommand: {}", static_cast<int>(id));

    auto cmd = CommandCode::RobotInteraction;
    int radar_id = getRadarId();
    std::vector<std::byte> buff(13, std::byte{0x00});
    buff[0] = std::byte{0xA5};
    buff[1] = std::byte{data.size() + 6};
    buff[2] = std::byte{(data.size() + 6) >> 8u};
    appendCRC8(std::span<std::byte>(buff.data(), 5));
    buff[5] = std::byte{static_cast<uint16_t>(cmd)};
    buff[6] = std::byte{static_cast<uint16_t>(cmd) >> 8u};

    buff[7] = std::byte{static_cast<uint16_t>(id)};
    buff[8] = std::byte{static_cast<uint16_t>(id) >> 8u};
    buff[9] = std::byte{static_cast<uint16_t>(radar_id)};
    buff[10] = std::byte{static_cast<uint16_t>(radar_id) >> 8u};
    buff[11] = std::byte{static_cast<uint16_t>(receiver)};
    buff[12] = std::byte{static_cast<uint16_t>(receiver) >> 8u};
    buff.insert(buff.end(), std::make_move_iterator(data.begin()),
                std::make_move_iterator(data.end()));
    buff.resize(buff.size() + 2);
    appendCRC16(buff);
    return this->serial_->write(buff);
}

bool RefereeCommunicator::decode() noexcept {
    static DecodeStatus status = DecodeStatus::Free;
    static std::vector<std::byte> messageBuffer;
    static size_t dataLength = 0;
    static uint16_t command_id;
    static bool decoded;
    static std::byte beginFlag{0xA5};

    spdlog::trace("Decoding data from serial port.");

    decoded = false;
    for (size_t i = 0; i < this->data_buffer_.size(); i++) {
        std::byte dataByte = this->data_buffer_[i];
        spdlog::trace("Received data: {}", static_cast<uint8_t>(dataByte));

        switch (status) {
            case DecodeStatus::Free: {
                messageBuffer.clear();
                if (dataByte == beginFlag) {
                    spdlog::trace("Received begin flag.");
                    messageBuffer.push_back(dataByte);
                    status = DecodeStatus::Length;
                }
                break;
            }
            case DecodeStatus::Length: {
                messageBuffer.push_back(dataByte);
                if (messageBuffer.size() == 3) {
                    dataLength = static_cast<uint16_t>(messageBuffer[2]) << 8 |
                                 static_cast<uint16_t>(messageBuffer[1]);
                    spdlog::trace("Received data length: {}", dataLength);
                }
                if (messageBuffer.size() == 5) {
                    if (verifyCRC8(messageBuffer)) {
                        status = DecodeStatus::CRC16;
                        spdlog::trace("CRC8 verified.");
                    } else {
                        status = DecodeStatus::Free;
                        spdlog::trace("CRC8 verification failed.");
                    }
                }
                break;
            }
            case DecodeStatus::CRC16: {
                messageBuffer.push_back(dataByte);
                if (messageBuffer.size() == 7) {
                    command_id = static_cast<uint16_t>(messageBuffer[6]) << 8 |
                                 static_cast<uint16_t>(messageBuffer[5]);
                    spdlog::trace("Received command: {}",
                                  magic_enum::enum_name(
                                      static_cast<CommandCode>(command_id)));
                } else if (messageBuffer.size() == 9 + dataLength) {
                    if (verifyCRC16(messageBuffer)) {
                        fetchData(
                            std::span<std::byte>(messageBuffer.data() + 7,
                                                 messageBuffer.size() - 9),
                            static_cast<CommandCode>(command_id));
                        decoded = true;
                        spdlog::trace("CRC16 verified");
                    } else {
                        spdlog::trace("CRC16 verification failed.");
                    }
                    status = DecodeStatus::Free;
                }
                break;
            }
        }
    }
    return decoded;
}

bool RefereeCommunicator::fetchData(std::span<std::byte> data,
                                    CommandCode command_id) noexcept {
    std::unique_lock<std::shared_mutex> lock(comm_mutex_);
    switch (command_id) {
        case CommandCode::GameStatus: {
            spdlog::debug("Fetching data with command: GameStatus");
            std::memcpy(game_status_.get(), data.data(), data.size());
            break;
        }
        case CommandCode::GameResult: {
            spdlog::debug("Fetching data with command: GameResult");
            std::memcpy(game_result_.get(), data.data(), data.size());
            break;
        }
        case CommandCode::GameRobotHP: {
            spdlog::debug("Fetching data with command: GameRobotHP");
            std::memcpy(robot_health_point_.get(), data.data(), data.size());
            break;
        }
        case CommandCode::Event: {
            spdlog::debug("Fetching data with command: Event");
            std::memcpy(event_data_.get(), data.data(), data.size());
            break;
        }
        case CommandCode::SupplyProjectileAction: {
            spdlog::debug("Fetching data with command: SupplyProjectileAction");
            std::memcpy(projectile_action_.get(), data.data(), data.size());
            break;
        }
        case CommandCode::RefereeWarning: {
            spdlog::debug("Fetching data with command: RefereeWarning");
            std::memcpy(referee_warning_.get(), data.data(), data.size());
            break;
        }
        case CommandCode::DartInfo: {
            spdlog::debug("Fetching data with command: DartInfo");
            std::memcpy(dart_info_.get(), data.data(), data.size());
            break;
        }
        case CommandCode::RobotStatus: {
            spdlog::debug("Fetching data with command: RobotStatus");
            std::memcpy(radar_status_.get(), data.data(), data.size());
            break;
        }
        case CommandCode::RadarMark: {
            spdlog::debug("Fetching data with command: RadarMark");
            std::memcpy(radar_mark_data_.get(), data.data(), data.size());
            break;
        }
        case CommandCode::RadarInfo: {
            spdlog::debug("Fetching data with command: RadarInfo");
            std::memcpy(radar_info_.get(), data.data(), data.size());
            this->qualifications.emplace_back(radar_info_->radar_info &
                                              0b00000011);
            if (this->qualifications.size() > 4) {
                this->qualifications.pop_front();
            }
            break;
        }
        case CommandCode::RobotInteraction: {
            spdlog::debug("Fetching data with command: RobotInteraction");
            auto dataCmdId = *reinterpret_cast<SubContentId*>(data.data());
            std::memcpy(sentry_data_.get(), data.data(), data.size());
            break;
        }
        default: {
            break;
        }
    }
    this->last_receive_timestamp_ = std::chrono::system_clock::now();
}

bool RefereeCommunicator::isEnemy(Robot::Label label) noexcept {
    static const std::vector<Robot::Label> BLUE_LABELS{
        Robot::Label::BlueEngineer,      Robot::Label::BlueHero,
        Robot::Label::BlueInfantryFive,  Robot::Label::BlueInfantryFour,
        Robot::Label::BlueInfantryThree, Robot::Label::BlueSentry};

    uint8_t radarId = getRadarId();
    if (radarId == 0) {
        spdlog::warn("Radar id not set.");
        return false;
    }

    bool isBlue = std::find(BLUE_LABELS.begin(), BLUE_LABELS.end(), label) !=
                  BLUE_LABELS.end();
    bool isRed = !isBlue;

    return radarId == static_cast<int>(protocol::Id::RadarBlue) ? isRed
                                                                : isBlue;
}

void RefereeCommunicator::appendCRC8(std::span<std::byte> data) noexcept {
    spdlog::trace("Appending CRC8 to data with length {}.", data.size() - 1);
    auto crc = CRC8_Check_Sum(reinterpret_cast<const uint8_t*>(data.data()),
                              data.size() - 1);
    data[data.size() - 1] = std::byte(crc);
}

void RefereeCommunicator::appendCRC16(std::span<std::byte> data) noexcept {
    spdlog::trace("Appending CRC16 to data with length {}.", data.size() - 2);
    auto crc = CRC16_Check_Sum(reinterpret_cast<const uint8_t*>(data.data()),
                               data.size() - 2);
    data[data.size() - 2] = std::byte(crc);
    data[data.size() - 1] = std::byte(crc >> 8u);
}

bool RefereeCommunicator::verifyCRC8(std::span<const std::byte> data) noexcept {
    spdlog::trace("Verifying CRC8 of data with length {}.", data.size() - 1);
    return CRC8_Check_Sum(reinterpret_cast<const uint8_t*>(data.data()),
                          data.size() - 1) ==
           static_cast<uint8_t>(data[data.size() - 1]);
}

bool RefereeCommunicator::verifyCRC16(
    std::span<const std::byte> data) noexcept {
    spdlog::trace("Verifying CRC16 of data with length {}.", data.size() - 2);
    return CRC16_Check_Sum(reinterpret_cast<const uint8_t*>(data.data()),
                           data.size() - 2) ==
           static_cast<uint16_t>(static_cast<uint16_t>(data[data.size() - 2]) |
                                 static_cast<uint16_t>(data[data.size() - 1])
                                     << 8u);
}

}  // namespace radar