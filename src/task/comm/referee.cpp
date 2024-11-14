#include "referee.h"

#include <spdlog/spdlog.h>

#include "FastCRC.h"

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

    spdlog::trace("Referee communicator initialized.");
}

bool RefereeCommunicator::init() {
    if (!serial_->open()) {
        spdlog::error("Failed to init RefereeCommunicator.");
    }
    return serial_->isOpen();
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

        cv::Point3f point = robot.location().value();  // 需要从米转换为厘米
        uint16_t x = static_cast<uint16_t>(point.x * 100);
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
}

void RefereeCommunicator::update() {
    if (!this->serial_->isOpen()) {
        return;
    }
    bool suc = this->serial_->read(this->data_buffer_);
    if (suc) {
        decode();
        spdlog::debug("");
    } else {
        spdlog::debug("");
    }
}

bool RefereeCommunicator::encode(CommandCode cmd,
                                 std::vector<std::byte>&& data) {
    static const std::vector<CommandCode> CMD4SEND{
        CommandCode::RobotInteraction, CommandCode::MapRobot,
        CommandCode::CustomInfo};

    if (!this->serial_->isOpen() ||
        std::find(CMD4SEND.begin(), CMD4SEND.end(), cmd) == CMD4SEND.end()) {
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
    appendCRC16(buff);
    bool isSuccess = this->serial_->write(buff);
    // TODO: 线程调度
    return isSuccess;
}

bool RefereeCommunicator::encode(SubContentId id, Id receiver,
                                 std::vector<std::byte>&& data) {
    static const std::vector<SubContentId> CONTENT4SEND{
        SubContentId::RobotCommunication, SubContentId::RadarCommand,
        SubContentId::RadarToSentry};

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
    appendCRC16(buff);
    bool isSuccess = this->serial_->write(buff);
    // TODO: 线程调度
    return isSuccess;
}

bool RefereeCommunicator::decode() {
    static DecodeStatus status = DecodeStatus::Free;
    static std::vector<std::byte> messageBuffer;
    static size_t dataLength = 0;
    static uint16_t commandId;
    static bool decoded;
    static std::byte beginFlag{0xA5};

    decoded = false;
    dataLength = 0;
    for (size_t i = 0; i < this->data_buffer_.size(); i++) {
        std::byte dataByte = this->data_buffer_[i];

        switch (status) {
            case DecodeStatus::Free: {
                messageBuffer.clear();
                if (dataByte == beginFlag) {
                    messageBuffer.push_back(dataByte);
                    status = DecodeStatus::Length;
                }
                break;
            }
            case DecodeStatus::Length: {
                messageBuffer.push_back(dataByte);
                if (messageBuffer.size() == 3) {
                    dataLength =  // TODO:能不能优雅一点
                        static_cast<uint16_t>(messageBuffer[2]) << 8 |
                        static_cast<uint16_t>(messageBuffer[1]);
                }
                if (messageBuffer.size() == 7 + dataLength) {
                    if (verifyCRC8(
                            std::span<std::byte>(messageBuffer.data(), 5))) {
                        status = DecodeStatus::CRC16;
                    } else {
                        status = DecodeStatus::Free;
                    }
                }
                break;
            }
            case DecodeStatus::CRC16: {
                messageBuffer.push_back(dataByte);
                if (messageBuffer.size() == 9 + dataLength) {
                    if (verifyCRC16(messageBuffer)) {
                        commandId =
                            static_cast<uint16_t>(
                                static_cast<uint8_t>(messageBuffer[6]) << 8) |
                            static_cast<uint8_t>(messageBuffer[5]);
                        fetchData(
                            std::span<std::byte>(messageBuffer.data() + 7,
                                                 messageBuffer.size() - 7),
                            static_cast<CommandCode>(commandId));
                        status = DecodeStatus::Free;
                        decoded = true;
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
    switch (command_id) {
        case CommandCode::GameStatus: {
            std::memcpy(&this->game_status_, data.data(), data.size());
            break;
        }
        case CommandCode::GameResult: {
            memcpy(&this->game_result_, data.data(), data.size());
            break;
        }
        case CommandCode::GameRobotHP: {
            memcpy(&this->robot_health_point_, data.data(), data.size());
            break;
        }
        case CommandCode::Event: {
            memcpy(&this->event_data_, data.data(), data.size());
            break;
        }
        case CommandCode::SupplyProjectileAction: {
            memcpy(&this->projectile_action_, data.data(), data.size());
            break;
        }
        case CommandCode::RefereeWarning: {
            memcpy(&this->referee_warning_, data.data(), data.size());
            break;
        }
        case CommandCode::DartInfo: {
            memcpy(&this->dart_info_, data.data(), data.size());
            break;
        }
        case CommandCode::RobotStatus: {
            memcpy(&this->radar_status_, data.data(), data.size());
            break;
        }
        case CommandCode::RadarMark: {
            memcpy(&this->radar_mark_data_, data.data(), data.size());
            break;
        }
        case CommandCode::RadarInfo: {
            memcpy(&this->radar_info_, data.data(), data.size());
            // TODO: 队列确认
            break;
        }
        case CommandCode::RobotInteraction: {
            auto dataCmdId = *reinterpret_cast<SubContentId*>(data.data());
            switch (dataCmdId) {
                case SubContentId::RobotCommunication: {
                    break;
                }
                case SubContentId::Sentry: {
                    break;
                }
            }
            break;
        }
        default: {
            break;
        }
    }
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

void RefereeCommunicator::appendCRC8(std::span<std::byte> data) {
    static FastCRC8 crc8;
    auto crc = crc8.smbus(reinterpret_cast<const uint8_t*>(data.data()),
                          data.size() - 1);
    data[data.size() - 1] = std::byte(crc);
}

void RefereeCommunicator::appendCRC16(std::span<std::byte> data) {
    static FastCRC16 crc16;
    auto crc = crc16.mcrf4xx(reinterpret_cast<const uint8_t*>(data.data()),
                             data.size() - 2);
    data[data.size() - 2] = std::byte(crc);
    data[data.size() - 1] = std::byte(crc >> 8u);
}

bool RefereeCommunicator::verifyCRC8(std::span<const std::byte> data) {
    static FastCRC8 crc8;
    return crc8.smbus(reinterpret_cast<const uint8_t*>(data.data()),
                      data.size() - 1) ==
           static_cast<uint8_t>(data[data.size() - 1]);
}

bool RefereeCommunicator::verifyCRC16(std::span<const std::byte> data) {
    static FastCRC16 crc16;
    return crc16.mcrf4xx(reinterpret_cast<const uint8_t*>(data.data()),
                         data.size() - 2) ==
           static_cast<uint16_t>(static_cast<uint16_t>(data[data.size() - 2]) |
                                 static_cast<uint16_t>(data[data.size() - 1])
                                     << 8u);
}

}  // namespace radar