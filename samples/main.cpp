#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "sample_radar.h"

std::vector<cv::Mat> readImages(std::string_view folder_path);

std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr,
          std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>>
readClouds(std::string_view folder_path);

int main() {
    cv::Size image_size(2592, 2048);
    cv::Matx33f intrinsic(1685.51538398561, 0, 1278.99324114319, 0,
                          1685.26471848220, 1037.21273138299, 0, 0, 1);
    cv::Matx44f lidar_to_camera(0, -1, 0, 0.85443, 0, 0, -1, -37.6845, 1, 0, 0,
                                12.2631, 0.0, 0.0, 0.0, 1.0);
    cv::Matx44f world_to_camera(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    cv::Point3f lidar_noise(0.4, 0.4, 0.4);

    SampleRadar radar("../models/car.engine", "../models/armor.engine",
                      image_size, intrinsic, lidar_to_camera, world_to_camera,
                      lidar_noise);

    auto images = readImages("../assets/images");
    auto [background_cloud, clouds] = readClouds("../assets/clouds");
    if (images.size() != clouds.size()) {
        throw std::logic_error("sizes do not match");
    }
    const auto start_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::milliseconds(100);

    radar.updateBackgroundCloud(background_cloud);

    for (size_t i = 0; i < images.size(); ++i) {
        const auto& image = images[i];
        const auto& cloud = clouds[i];
        const auto timestamp = start_time + i * duration;

        Frame frame(image, cloud, timestamp);
        radar.runOnce(frame);
    }

    return EXIT_SUCCESS;
}

std::vector<cv::Mat> readImages(std::string_view folder_path) {
    if (!std::filesystem::exists(folder_path)) {
        throw std::runtime_error(std::string(folder_path) + " does not exist");
    }

    std::vector<cv::Mat> images;
    for (int i = 0; i < 10; ++i) {
        std::string filename =
            std::filesystem::path(folder_path) / (std::to_string(i) + ".jpg");
        if (!std::filesystem::exists(filename)) {
            throw std::runtime_error(filename + " does not exist");
        }
        cv::Mat image = cv::imread(filename);
        images.emplace_back(image);
    }
    return images;
}

auto readClouds(std::string_view folder_path)
    -> std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr,
                 std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>> {
    if (!std::filesystem::exists(folder_path)) {
        throw std::runtime_error(std::string(folder_path) + " does not exist");
    }

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
    for (int i = 0; i < 10; ++i) {
        std::string filename =
            std::filesystem::path(folder_path) / (std::to_string(i) + ".pcd");
        if (!std::filesystem::exists(filename)) {
            throw std::runtime_error(filename + " does not exist");
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZ>());
        pcl::io::loadPCDFile(filename, *cloud);
        clouds.emplace_back(cloud);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr background_cloud(
        new pcl::PointCloud<pcl::PointXYZ>());
    std::string filename =
        std::filesystem::path(folder_path) / "background.pcd";
    if (!std::filesystem::exists(filename)) {
        throw std::runtime_error(filename + " does not exist");
    }
    pcl::io::loadPCDFile(filename, *background_cloud);

    return std::make_pair(background_cloud, clouds);
}
