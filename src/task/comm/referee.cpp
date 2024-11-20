#include "referee.h"

#include <spdlog/spdlog.h>

#include <magic_enum/magic_enum.hpp>

#include "crc.h"

namespace radar {

using namespace protocol;
using namespace serial;

constexpr size_t BUFFER_SIZE = 128;

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
        spdlog::warn(
            "Failed to init RefereeCommunicator: serial port open failed");
    }

    spdlog::trace("Referee communicator initialized.");
}

bool RefereeCommunicator::reconnect() {
    if (serial_->isOpen()) {
        return true;
    }
    this->is_connected_ = serial_->open();
    return this->is_connected_;
}

void RefereeCommunicator::sendMapRobot(const std::span<const Robot> robots) {
    map_robot_data_t data;
    std::memset(&data, 0, sizeof(data));

    auto filtered_robots =
        robots | std::ranges::views::filter([](const Robot& robot) {
            return robot.label().has_value() && robot.isLocated();
        });

    for (const auto& robot : filtered_robots) {
        std::optional<Robot::Label> label = robot.label();

        if (!isEnemy(label.value())) {  // 跳过友军
            continue;
        }

        spdlog::debug("Send location of {} to referee system: ",
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

    std::vector<std::byte> sendData(
        reinterpret_cast<std::byte*>(&data),
        reinterpret_cast<std::byte*>(&data) + sizeof(data));
    if (!encode(CommandCode::MapRobot, std::move(sendData))) {
        spdlog::error("Failed to encode map robot data");
    };
    spdlog::debug("Send map robot data to referee.");
}

void RefereeCommunicator::update() {
    if (!this->is_connected_) {
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
    bool isSuccess = this->serial_->write(buff);
    // TODO: 线程调度
    return isSuccess;
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
    }

    auto cmd = CommandCode::RobotInteraction;
    int length = (data.size() + 15) * 4;
    std::vector<std::byte> buff(13, std::byte{0x00});
    buff[0] = std::byte{0xA5};
    buff[1] = std::byte{data.size() + 6};
    buff[2] = std::byte{(data.size() + 6) >> 8u};
    appendCRC8(std::span<std::byte>(buff.data(), 5));
    buff[4] = std::byte{static_cast<uint16_t>(cmd)};
    buff[5] = std::byte{static_cast<uint16_t>(cmd) >> 8u};
    buff[6] = std::byte{static_cast<uint16_t>(id)};
    buff[7] = std::byte{static_cast<uint16_t>(id) >> 8u};
    buff[8] = std::byte{static_cast<uint16_t>(radar_status_->robot_id)};
    buff[9] = std::byte{static_cast<uint16_t>(radar_status_->robot_id) >> 8u};
    buff[10] = std::byte{static_cast<uint16_t>(receiver)};
    buff[11] = std::byte{static_cast<uint16_t>(receiver) >> 8u};
    buff.insert(buff.end(), std::make_move_iterator(data.begin()),
                std::make_move_iterator(data.end()));
    buff.resize(buff.size() + 2);
    appendCRC16(buff);
    bool isSuccess = this->serial_->write(buff);
    // TODO: 线程调度
    return isSuccess;
}

bool RefereeCommunicator::decode() noexcept {
    static DecodeStatus status = DecodeStatus::Free;
    static std::vector<std::byte> messageBuffer;
    static size_t dataLength = 0;
    static uint16_t command_id;
    static bool decoded;
    static std::byte beginFlag{0xA5};

    decoded = false;
    dataLength = 0;
    spdlog::trace("Decoding data from serial port.");
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
                                                 messageBuffer.size() - 7),
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
                                    CommandCode command_id) {
    spdlog::trace("Fetch data with command: {}",
                  magic_enum::enum_name(static_cast<CommandCode>(command_id)));
    switch (command_id) {
        case CommandCode::GameStatus: {
            std::memcpy(&this->game_status_, data.data(), data.size());
            break;
        }
        case CommandCode::GameResult: {
            std::memcpy(&this->game_result_, data.data(), data.size());
            break;
        }
        case CommandCode::GameRobotHP: {
            std::memcpy(&this->robot_health_point_, data.data(), data.size());
            break;
        }
        case CommandCode::Event: {
            std::memcpy(&this->event_data_, data.data(), data.size());
            break;
        }
        case CommandCode::SupplyProjectileAction: {
            std::memcpy(&this->projectile_action_, data.data(), data.size());
            break;
        }
        case CommandCode::RefereeWarning: {
            std::memcpy(&this->referee_warning_, data.data(), data.size());
            break;
        }
        case CommandCode::DartInfo: {
            std::memcpy(&this->dart_info_, data.data(), data.size());
            break;
        }
        case CommandCode::RobotStatus: {
            std::memcpy(&this->radar_status_, data.data(), data.size());
            break;
        }
        case CommandCode::RadarMark: {
            std::memcpy(&this->radar_mark_data_, data.data(), data.size());
            break;
        }
        case CommandCode::RadarInfo: {
            std::memcpy(&this->radar_info_, data.data(), data.size());
            // TODO: 队列确认
            break;
        }
        case CommandCode::RobotInteraction: {
            auto dataCmdId = *reinterpret_cast<SubContentId*>(data.data());
            std::memcpy(&this->sentry_data_, data.data(), data.size());
            break;
        }
        default: {
            spdlog::error("Fetch data with unknown type of command");
            break;
        }
    }
    this->last_receive_timestamp_ = std::chrono::system_clock::now();
}

bool RefereeCommunicator::isEnemy(Robot::Label label) {
    static const std::vector<Robot::Label> BLUE_LABELS{
        Robot::Label::BlueEngineer,      Robot::Label::BlueHero,
        Robot::Label::BlueInfantryFive,  Robot::Label::BlueInfantryFour,
        Robot::Label::BlueInfantryThree, Robot::Label::BlueSentry};

    int radarId = this->radar_status_->robot_id;  // TODO:并发控制
    assert(radarId == static_cast<int>(protocol::Id::RadarBlue) ||
           radarId == static_cast<int>(protocol::Id::RadarRed));

    bool isBlue = std::find(BLUE_LABELS.begin(), BLUE_LABELS.end(), label) !=
                  BLUE_LABELS.end();
    bool isRed = !isBlue;

    return radarId == static_cast<int>(protocol::Id::RadarBlue) ? isRed
                                                                : isBlue;
}

void RefereeCommunicator::appendCRC8(
    std::span<std::byte> data) {  // 裁判系统使用8541多项式
    auto crc = CRC8_Check_Sum(reinterpret_cast<const uint8_t*>(data.data()),
                              data.size() - 1);
    data[data.size() - 1] = std::byte(crc);
}

void RefereeCommunicator::appendCRC16(std::span<std::byte> data) {
    auto crc = CRC16_Check_Sum(reinterpret_cast<const uint8_t*>(data.data()),
                               data.size() - 2);
    data[data.size() - 2] = std::byte(crc);
    data[data.size() - 1] = std::byte(crc >> 8u);
}

bool RefereeCommunicator::verifyCRC8(std::span<const std::byte> data) {
    return CRC8_Check_Sum(reinterpret_cast<const uint8_t*>(data.data()),
                          data.size() - 1) ==
           static_cast<uint8_t>(data[data.size() - 1]);
}

bool RefereeCommunicator::verifyCRC16(std::span<const std::byte> data) {
    return CRC16_Check_Sum(reinterpret_cast<const uint8_t*>(data.data()),
                           data.size() - 2) ==
           static_cast<uint16_t>(static_cast<uint16_t>(data[data.size() - 2]) |
                                 static_cast<uint16_t>(data[data.size() - 1])
                                     << 8u);
}

}  // namespace radar