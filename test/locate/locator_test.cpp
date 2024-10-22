#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>
#include <random>

#define private public
#define protected public

#include "locate/locator.h"
#include "robot/robot.h"

#undef private
#undef protected

class LocatorTest : public ::testing::Test {
   private:
    int image_width = 640;
    int image_height = 480;
    cv::Matx33f intrinsic = cv::Matx33f::eye();
    cv::Matx44f lidar_to_camera = cv::Matx44f::eye();
    cv::Matx44f world_to_camera = cv::Matx44f::eye();
    float zoom_factor = 0.5f;
    size_t queue_size = 5;
    float min_depth_diff = 0.05f;
    float max_depth_diff = 5.0f;
    float cluster_tolerance = 100.f;
    int min_cluster_size = 10;
    int max_cluster_size = 1000;
    float max_distance = 20.0f;

   protected:
    std::unique_ptr<radar::Locator> locator = nullptr;

    virtual void SetUp() {
        locator = std::make_unique<radar::Locator>(
            image_width, image_height, intrinsic, lidar_to_camera,
            world_to_camera, zoom_factor, queue_size, min_depth_diff,
            max_depth_diff, cluster_tolerance, min_cluster_size,
            max_cluster_size, max_distance);
    }
};

TEST_F(LocatorTest, TestZoom) {
    cv::Rect rect(100, 100, 50, 50);
    cv::Rect zoomed_rect = locator->zoom(rect);

    EXPECT_EQ(zoomed_rect.width,
              static_cast<int>(rect.width * locator->zoom_factor_));
    EXPECT_EQ(zoomed_rect.height,
              static_cast<int>(rect.height * locator->zoom_factor_));
}

TEST_F(LocatorTest, TestCoordinateTransform) {
    cv::Point3f lidar_point(1.0f, 2.0f, 3.0f);
    cv::Point3f world_point = locator->lidarToWorld(lidar_point);

    // Since the identity matrix is ​​used, the conversion result should
    // remain unchanged
    EXPECT_FLOAT_EQ(world_point.x, lidar_point.x);
    EXPECT_FLOAT_EQ(world_point.y, lidar_point.y);
    EXPECT_FLOAT_EQ(world_point.z, lidar_point.z);

    cv::Point3f camera_point = locator->lidarToCamera(lidar_point);
    EXPECT_FLOAT_EQ(camera_point.x,
                    lidar_point.x * locator->zoom_factor_ / camera_point.z);
    EXPECT_FLOAT_EQ(camera_point.y,
                    lidar_point.y * locator->zoom_factor_ / camera_point.z);
    EXPECT_FLOAT_EQ(camera_point.z, lidar_point.z);

    cv::Point3f reversed_lidar_point = locator->cameraToLidar(camera_point);
    EXPECT_FLOAT_EQ(reversed_lidar_point.x, lidar_point.x);
    EXPECT_FLOAT_EQ(reversed_lidar_point.y, lidar_point.y);
    EXPECT_FLOAT_EQ(reversed_lidar_point.z, lidar_point.z);
}

TEST_F(LocatorTest, TestCloudCluster) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>());

    const int num_points_per_cloud = 500;

    std::random_device rd;
    std::mt19937 gen(rd());

    // Cluster 1 x-center at 160
    std::normal_distribution<> cloud1_x_dist(160.0, 10.0);
    // Cluster 1 y-center at 120
    std::normal_distribution<> cloud1_y_dist(120.0, 10.0);
    // Cluster 1 depth
    std::uniform_real_distribution<> cloud1_depth_dist(5.0, 6.0);

    // Cluster 2 x-center at 480
    std::normal_distribution<> cloud2_x_dist(80.0, 10.0);
    // Cluster 2 y-center at 360
    std::normal_distribution<> cloud2_y_dist(60.0, 10.0);
    // Cluster 2 depth
    std::uniform_real_distribution<> cloud2_depth_dist(1.0, 2.0);

    for (int i = 0; i < num_points_per_cloud; ++i) {
        int x = std::clamp(static_cast<int>(cloud1_x_dist(gen)), 0,
                           locator->image_width_ - 1);
        int y = std::clamp(static_cast<int>(cloud1_y_dist(gen)), 0,
                           locator->image_height_ - 1);
        float depth = cloud1_depth_dist(gen);
        locator->diff_depth_image_.at<float>(y, x) = depth;
    }

    for (int i = 0; i < num_points_per_cloud; ++i) {
        int x = std::clamp(static_cast<int>(cloud2_x_dist(gen)), 0,
                           locator->image_width_ - 1);
        int y = std::clamp(static_cast<int>(cloud2_y_dist(gen)), 0,
                           locator->image_height_ - 1);
        float depth = cloud2_depth_dist(gen);
        locator->diff_depth_image_.at<float>(y, x) = depth;
    }

    locator->cluster();
    EXPECT_EQ(locator->clusters_.size(), 2);
}

TEST_F(LocatorTest, TestRobotSearch) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>());

    const int num_points_per_cloud = 500;

    std::random_device rd;
    std::mt19937 gen(rd());

    // Cluster 1 x-center at 160
    std::normal_distribution<> cloud1_x_dist(160.0, 10.0);
    // Cluster 1 y-center at 120
    std::normal_distribution<> cloud1_y_dist(120.0, 10.0);
    // Cluster 1 depth
    std::uniform_real_distribution<> cloud1_depth_dist(5.0, 6.0);

    // Cluster 2 x-center at 480
    std::normal_distribution<> cloud2_x_dist(80.0, 10.0);
    // Cluster 2 y-center at 360
    std::normal_distribution<> cloud2_y_dist(60.0, 10.0);
    // Cluster 2 depth
    std::uniform_real_distribution<> cloud2_depth_dist(1.0, 2.0);

    for (int i = 0; i < num_points_per_cloud; ++i) {
        int x = std::clamp(static_cast<int>(cloud1_x_dist(gen)), 0,
                           locator->image_width_ - 1);
        int y = std::clamp(static_cast<int>(cloud1_y_dist(gen)), 0,
                           locator->image_height_ - 1);
        float depth = cloud1_depth_dist(gen);
        locator->diff_depth_image_.at<float>(y, x) = depth;
    }

    for (int i = 0; i < num_points_per_cloud; ++i) {
        int x = std::clamp(static_cast<int>(cloud2_x_dist(gen)), 0,
                           locator->image_width_ - 1);
        int y = std::clamp(static_cast<int>(cloud2_y_dist(gen)), 0,
                           locator->image_height_ - 1);
        float depth = cloud2_depth_dist(gen);
        locator->diff_depth_image_.at<float>(y, x) = depth;
    }

    locator->cluster();

    radar::Robot robot;
    robot.rect_ = cv::Rect2f(140, 100, 40, 40);
    locator->search(robot);
    EXPECT_TRUE(robot.location().has_value());
}