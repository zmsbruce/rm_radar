#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cmath>
#include <future>
#include <memory>
#include <opencv2/opencv.hpp>
#include <ranges>
#include <string_view>
#include <utility>

#include "detect/detector.h"
#include "locate/locator.h"
#include "robot/robot.h"
#include "track/tracker.h"

namespace radar {

struct CvColor {
    static inline cv::Scalar Blue = cv::Scalar(255, 0, 0);
    static inline cv::Scalar Red = cv::Scalar(0, 0, 255);
    static inline cv::Scalar White = cv::Scalar(255, 255, 255);
    static inline cv::Scalar Black = cv::Scalar(0, 0, 0);
    static inline cv::Scalar Grey = cv::Scalar(128, 128, 128);
    static inline cv::Scalar LightRed = cv::Scalar(119, 47, 255);
    static inline cv::Scalar LightBlue = cv::Scalar(250, 200, 0);
    static inline cv::Scalar LightGrey = cv::Scalar(192, 192, 192);
    static inline cv::Scalar Purple = cv::Scalar(240, 32, 160);
    static inline cv::Scalar Green = cv::Scalar(0, 255, 0);
    static inline cv::Scalar Yellow = cv::Scalar(0, 255, 255);
    static inline cv::Scalar Orange = cv::Scalar(0, 165, 255);
    static inline cv::Scalar Pink = cv::Scalar(203, 192, 255);
};

constexpr int kClasses = 12;
constexpr int kMaxBatchSize = 20;
constexpr int kOptBatchSize = 5;
constexpr float kMaxCosineDistance = 1.5;
constexpr int kNnBudget = 100;
constexpr float kMaxIouDistance = 0.9;
constexpr int kMaxAge = 17;
constexpr int kNInit = 4;
const std::unordered_map<int, std::string> kArmorNames{
    {Label::NoneType, "None"},       {Label::BlueHero, "B1"},
    {Label::BlueEngineer, "B2"},     {Label::BlueInfantryThree, "B3"},
    {Label::BlueInfantryFour, "B4"}, {Label::BlueInfantryFive, "B5"},
    {Label::RedHero, "R1"},          {Label::RedEngineer, "R2"},
    {Label::RedInfantryThree, "R3"}, {Label::RedInfantryFour, "R4"},
    {Label::RedInfantryFive, "R5"},  {Label::BlueSentry, "Bs"},
    {Label::RedSentry, "Rs"}};
const std::vector<cv::Scalar> kPointColors{CvColor::Purple, CvColor::Red,
                                           CvColor::Green,  CvColor::Yellow,
                                           CvColor::Orange, CvColor::Pink};

class Radar {
   public:
    Radar() = delete;

    Radar(std::string_view car_engine_path, std::string_view armor_engine_path,
          int image_width, int image_height, const cv::Matx33f& intrinsic,
          const cv::Matx44f& lidar_to_camera,
          const cv::Matx44f& world_to_camera, bool enable_vis = false);

    ~Radar();

    std::vector<std::pair<Label, cv::Point3f>> runOnce(
        const cv::Mat& image,
        const pcl::PointCloud<pcl::PointXYZ>& point_cloud) noexcept;

    void visualize();

   private:
    void detect(const cv::Mat& image) noexcept;
    void track() noexcept;
    cv::Rect2f cv_rect(const Detection& detection) noexcept;
    static cv::Scalar cv_color(int label) noexcept;
    static cv::Scalar cv_color(int label, TrackState track_state) noexcept;
    static void drawDetection(cv::Mat& image, std::string_view text,
                              const cv::Rect2f& rect, const cv::Scalar& color,
                              int line_width = 2, float font_scale = 1,
                              int font_thickness = 4);
    std::unique_ptr<Detector> car_detector_, armor_detector_;
    std::unique_ptr<Tracker> tracker_;
    std::unique_ptr<Locator> locator_;
    std::vector<cv::Mat> car_images_;
    std::vector<Robot> robots_detect_, robots_track_locate_;
    bool enable_vis_ = false;
    cv::Mat image_detect_, image_track_;
};

Radar::Radar(std::string_view car_engine_path,
             std::string_view armor_engine_path, int image_width,
             int image_height, const cv::Matx33f& intrinsic,
             const cv::Matx44f& lidar_to_camera,
             const cv::Matx44f& world_to_camera, bool enable_vis)
    : enable_vis_{enable_vis} {
    car_detector_ = std::make_unique<Detector>(car_engine_path, 1, 1);
    armor_detector_ = std::make_unique<Detector>(armor_engine_path, kClasses,
                                                 kMaxBatchSize, kOptBatchSize);
    tracker_ = std::make_unique<Tracker>(kMaxCosineDistance, kNnBudget,
                                         kMaxIouDistance, kMaxAge, kNInit);
    locator_ = std::make_unique<Locator>(image_width, image_height, intrinsic,
                                         lidar_to_camera, world_to_camera);

    if (enable_vis) {
        cv::namedWindow("detect", cv::WINDOW_NORMAL);
        cv::namedWindow("track", cv::WINDOW_NORMAL);
        cv::namedWindow("locate", cv::WINDOW_NORMAL);

        cv::resizeWindow("detect", 1296, 1024);
        cv::resizeWindow("track", 1296, 1024);
        cv::resizeWindow("locate", 1296, 1024);
    }
}

Radar::~Radar() {
    if (enable_vis_) {
        cv::destroyWindow("detect");
        cv::destroyWindow("track");
        cv::destroyWindow("locate");
    }
}

auto Radar::runOnce(const cv::Mat& image,
                    const pcl::PointCloud<pcl::PointXYZ>& point_cloud) noexcept
    -> std::vector<std::pair<Label, cv::Point3f>> {
    if (enable_vis_) {
        image_detect_ = image.clone();
        image_track_ = image.clone();
    }

    car_images_.clear();
    robots_detect_.clear();
    robots_track_locate_.clear();

    auto detect_track_future = std::async(std::launch::async, [&] {
        detect(image);
        track();
    });
    auto locate_future = std::async(std::launch::async, [&] {
        locator_->update(point_cloud);
        locator_->cluster();
    });
    detect_track_future.get();
    locate_future.get();

    locator_->search(robots_track_locate_);

    std::vector<std::pair<Label, cv::Point3f>> ret;
    ret.reserve(robots_track_locate_.size());
    for (const auto& robot :
         robots_track_locate_ | std::views::filter([](auto&& robot) {
             return robot.isLocated();
         })) {
        assert(robot.isTracked());
        ret.emplace_back(
            std::make_pair(robot.label().value(), robot.location().value()));
    }

    return ret;
}

void Radar::detect(const cv::Mat& image) noexcept {
    auto car_detections{car_detector_->detect(image)};
    car_images_.reserve(car_detections.size());
    robots_detect_.reserve(car_detections.size());

    std::ranges::for_each(car_detections, [&](const Detection& detection) {
        car_images_.emplace_back(std::move(image(cv_rect(detection)).clone()));
    });

    auto armor_detections_batch = armor_detector_->detect(car_images_);
    for (size_t i = 0; i < car_detections.size(); ++i) {
        if (armor_detections_batch[i].empty()) {
            continue;
        }
        robots_detect_.emplace_back(car_detections[i],
                                    armor_detections_batch[i]);
    }
}

void Radar::track() noexcept {
    tracker_->predict();
    tracker_->update(robots_detect_);
    const auto& tracks{tracker_->tracks()};
    robots_track_locate_.reserve(tracks.size());
    for (auto&& track : tracks) {
        robots_track_locate_.emplace_back(track);
    }
}

void Radar::drawDetection(cv::Mat& image, std::string_view text,
                          const cv::Rect2f& rect, const cv::Scalar& color,
                          int line_width, float font_scale,
                          int font_thickness) {
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

cv::Rect2f Radar::cv_rect(const Detection& detection) noexcept {
    return cv::Rect2f(detection.x, detection.y, detection.width,
                      detection.height);
}

cv::Scalar Radar::cv_color(int label) noexcept {
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

cv::Scalar Radar::cv_color(int label, TrackState track_state) noexcept {
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

void Radar::visualize() {
    if (!enable_vis_) {
        std::cerr << "calling `Radar::visualize()` with `enable_vis` flag off, "
                     "will do nothing."
                  << std::endl;
        return;
    }

    for (const auto& robot : robots_detect_) {
        if (!robot.isDetected()) {
            continue;
        }

        int label = robot.label().value();
        auto car_text = cv::format("%s %.2f", kArmorNames.at(label).c_str(),
                                   robot.confidence().value());
        drawDetection(image_detect_, car_text, robot.rect().value(),
                      cv_color(label));

        for (const auto& detection : robot.armors().value()) {
            auto armor_text =
                cv::format("%s %.2f", kArmorNames.at(detection.label).c_str(),
                           detection.confidence);
            drawDetection(image_detect_, armor_text, cv_rect(detection),
                          cv_color(detection.label));
        }
    }

    for (const auto& robot : robots_track_locate_) {
        if (!robot.isTracked()) {
            continue;
        }

        auto label_text = kArmorNames.at(robot.label().value()).c_str();
        auto track_state_text =
            robot.track_state().value() == TrackState::Confirmed   ? "Confirmed"
            : robot.track_state().value() == TrackState::Tentative ? "Tentative"
                                                                   : "Deleted";
        auto text = cv::format("%s %s", label_text, track_state_text);
        drawDetection(
            image_track_, text, robot.rect().value(),
            cv_color(robot.label().value(), robot.track_state().value()));
    }

    cv::Mat image_locate = locator_->background_depth_image_.clone();
    cv::Mat image_locate_normalized, image_locate_colored;
    cv::normalize(image_locate, image_locate_normalized, 0, 255,
                  cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(image_locate_normalized, image_locate_colored,
                      cv::COLORMAP_JET);
    for (int i = 0; i < locator_->diff_depth_image_.rows; ++i) {
        const float* row = locator_->diff_depth_image_.ptr<float>(i);
        for (int j = 0; j < locator_->diff_depth_image_.cols; ++j) {
            if (!iszero(row[j])) {
                int index =
                    locator_->pixel_index_map_.contains(std::make_pair(i, j))
                        ? -1
                        : locator_->pixel_index_map_.at(std::make_pair(i, j));
                int cluster_id = locator_->index_cluster_map_.contains(index)
                                     ? locator_->index_cluster_map_.at(index)
                                     : -1;
                cv::Scalar color =
                    cluster_id == -1
                        ? CvColor::White
                        : kPointColors[cluster_id % kPointColors.size()];
                cv::circle(image_locate_colored, {j, i}, 2, color, -1);
            }
        }
    }

    cv::imshow("detect", image_detect_);
    cv::imshow("track", image_track_);
    cv::imshow("locate", image_locate_colored);
}

}  // namespace radar