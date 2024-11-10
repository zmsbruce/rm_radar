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

void RefereeCommunicator::sendMapRobot(const std::span<const Robot> robots) {
    map_robot_data_t data;
    std::memset(&data, 0, sizeof(data));
    for (const auto& robot : robots) {
        std::optional<Robot::Label> label = robot.label();

        if (!label.has_value()) {
            spdlog::debug("Robot without label.");
            continue;
        } else if (!isEnemy(label.value())) {
            continue;
        }

        if (!robot.location().has_value()) {
            // TODO:没有坐标时的处理
            spdlog::debug("Robot without location.");
            continue;
        }

        // TODO: 这里坐标单位还没搞清楚，暂时按原来写
        cv::Point3f point = robot.location().value();
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

void RefereeCommunicator::update() {}

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

}  // namespace radar