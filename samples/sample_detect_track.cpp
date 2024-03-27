#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "detect/detector.h"
#include "robot/robot.h"
#include "track/tracker.h"

using namespace radar;

struct CvColor {
    static inline cv::Scalar Blue = cv::Scalar(255, 0, 0);
    static inline cv::Scalar Red = cv::Scalar(0, 0, 255);
    static inline cv::Scalar White = cv::Scalar(255, 255, 255);
    static inline cv::Scalar Black = cv::Scalar(0, 0, 0);
    static inline cv::Scalar Grey = cv::Scalar(128, 128, 128);
    static inline cv::Scalar LightRed = cv::Scalar(119, 47, 255);
    static inline cv::Scalar LightBlue = cv::Scalar(250, 200, 0);
    static inline cv::Scalar LightGrey = cv::Scalar(192, 192, 192);
};

std::unordered_map<int, std::string> kArmorNameMap{
    {Label::NoneType, "None"},       {Label::BlueHero, "B1"},
    {Label::BlueEngineer, "B2"},     {Label::BlueInfantryThree, "B3"},
    {Label::BlueInfantryFour, "B4"}, {Label::BlueInfantryFive, "B5"},
    {Label::RedHero, "R1"},          {Label::RedEngineer, "R2"},
    {Label::RedInfantryThree, "R3"}, {Label::RedInfantryFour, "R4"},
    {Label::RedInfantryFive, "R5"},  {Label::BlueSentry, "Bs"},
    {Label::RedSentry, "Rs"}};

inline cv::Scalar cv_color(int label) {
    switch (label) {
        case Label::BlueHero:
        case Label::BlueEngineer:
        case Label::BlueInfantryThree:
        case Label::BlueInfantryFour:
        case Label::BlueInfantryFive:
        case Label::BlueSentry:
            return CvColor::Blue;
        case Label::RedHero:
        case Label::RedEngineer:
        case Label::RedInfantryThree:
        case Label::RedInfantryFour:
        case Label::RedInfantryFive:
        case Label::RedSentry:
            return CvColor::Red;
        default:
            return CvColor::Grey;
    }
}

inline cv::Scalar cv_color(int label, TrackState track_state) {
    switch (track_state) {
        case TrackState::Confirmed:
            return cv_color(label);
        case TrackState::Tentative: {
            switch (label) {
                case Label::BlueHero:
                case Label::BlueEngineer:
                case Label::BlueInfantryThree:
                case Label::BlueInfantryFour:
                case Label::BlueInfantryFive:
                case Label::BlueSentry:
                    return CvColor::LightBlue;
                case Label::RedHero:
                case Label::RedEngineer:
                case Label::RedInfantryThree:
                case Label::RedInfantryFour:
                case Label::RedInfantryFive:
                case Label::RedSentry:
                    return CvColor::LightRed;
                default:
                    return CvColor::LightGrey;
            }
        }
        default:
            return CvColor::Black;
    }
}

inline cv::Scalar cv_color(const Detection& detection) {
    return cv_color(detection.label);
}

inline cv::Scalar cv_color(const Robot& robot) {
    return robot.isTracked()
               ? cv_color(robot.label().value(), robot.track_state().value())
           : robot.isDetected() ? cv_color(robot.label().value())
                                : CvColor::Black;
}

inline cv::Rect2f cv_rect(const Detection& detection) {
    return cv::Rect2f(detection.x, detection.y, detection.width,
                      detection.height);
}

inline void drawDetection(cv::Mat& image, std::string_view text,
                          const cv::Rect2f& rect, const cv::Scalar& color,
                          int line_width = 2, float font_scale = 1,
                          int font_thickness = 4) {
    cv::rectangle(image, rect, color, line_width);

    int base_line = 0;
    auto label_size = cv::getTextSize(text.data(), cv::FONT_HERSHEY_SIMPLEX,
                                      font_scale, font_thickness, &base_line);
    cv::Rect2f label_rect(rect.x, rect.y - label_size.height - base_line / 2,
                          label_size.width, label_size.height + base_line);
    cv::rectangle(image, label_rect, color, -1);
    cv::putText(image, text.data(), cv::Point(rect.x, rect.y),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255),
                font_thickness);
}

int main() {
    std::unique_ptr<Detector> car_detector{nullptr}, armor_detector{nullptr};
    car_detector = std::make_unique<Detector>("../models/car.engine", 1, 1);
    armor_detector =
        std::make_unique<Detector>("../models/armor.engine", 12, 20, 4);

    std::unique_ptr<Tracker> tracker{
        std::make_unique<Tracker>(1.5, 100, 0.9, 17, 4)};

    cv::VideoCapture capture("../assets/field_1.mp4");
    if (!capture.isOpened()) {
        throw std::runtime_error("video open failed");
    }

    cv::Mat image, image_detect, image_track;

    std::string window_name_detect{"detect result"};
    cv::namedWindow(window_name_detect, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_detect, cv::Size(1296, 1024));

    std::string window_name_track{"track result"};
    cv::namedWindow(window_name_track, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name_track, cv::Size(1296, 1024));

    while (capture.read(image)) {
        image_detect = image.clone();
        image_track = image.clone();

        // detecting
        std::vector<cv::Mat> car_images;
        auto car_detections = car_detector->detect(image);
        car_images.reserve(car_detections.size());
        std::for_each(
            car_detections.begin(), car_detections.end(),
            [&](const Detection& detection) {
                car_images.push_back(image(cv_rect(detection)).clone());
            });
        auto armor_detections_batch = armor_detector->detect(car_images);

        std::vector<Robot> robots_detect, robots_track;
        robots_detect.reserve(car_detections.size());

        assert(car_detections.size() == armor_detections_batch.size());
        for (size_t i = 0; i < car_detections.size(); ++i) {
            if (armor_detections_batch[i].empty()) {
                continue;
            }
            robots_detect.emplace_back(car_detections[i],
                                       armor_detections_batch[i]);
        }

        // tracking
        tracker->predict();
        tracker->update(robots_detect);
        const auto& tracks{tracker->tracks()};
        robots_track.reserve(tracks.size());
        for (auto&& track : tracks) {
            robots_track.emplace_back(track);
        }

        // visualize detect results
        for (const Robot& robot : robots_detect) {
            if (!robot.isDetected()) {
                continue;
            }
            auto car_text = cv::format(
                "%s %.2f", kArmorNameMap[robot.label().value()].c_str(),
                robot.confidence().value());
            drawDetection(image_detect, car_text, robot.rect().value(),
                          cv_color(robot));

            auto armors = robot.armors().value();
            for (const auto& armor : armors) {
                auto armor_rect = cv_rect(armor);

                auto armor_text =
                    cv::format("%s %.2f", kArmorNameMap[armor.label].c_str(),
                               armor.confidence);
                drawDetection(image_detect, armor_text, armor_rect,
                              cv_color(armor));
            }
        }

        // visualize track results
        for (const Robot& robot : robots_track) {
            if (!robot.isTracked()) {
                continue;
            }

            auto label_text = kArmorNameMap[robot.label().value()].c_str();
            auto track_state_text =
                robot.track_state().value() == TrackState::Confirmed
                    ? "Confirmed"
                : robot.track_state().value() == TrackState::Tentative
                    ? "Tentative"
                    : "Deleted";
            auto text = cv::format("%s %s", label_text, track_state_text);
            drawDetection(image_track, text, robot.rect().value(),
                          cv_color(robot));
        }

        cv::imshow(window_name_detect, image_detect);
        cv::imshow(window_name_track, image_track);
        cv::waitKey(10);
    }

    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}