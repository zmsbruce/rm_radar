#include <opencv2/opencv.hpp>
#include <vector>

#include "gtest/gtest.h"
#include "locate/cluster.h"

TEST(LocateTest, TestCluster) {
    std::vector<cv::Point2f> points{
        {0.0, 0.0},  {0.5, 0.0},  {1.0, 0.0},  {0.0, 0.5},  {0.5, 0.5},
        {1.0, 0.5},  {0.0, 1.0},  {0.5, 1.0},  {1.0, 1.0},  {10.0, 0.0},
        {10.5, 0.0}, {11.0, 0.0}, {10.0, 0.5}, {10.5, 0.5}, {11.0, 0.5},
        {10.0, 1.0}, {10.5, 1.0}, {11.0, 1.0}, {20.0, 0.0}, {20.5, 0.0},
        {21.0, 0.0}, {20.0, 0.5}, {20.5, 0.5}, {21.0, 0.5}, {20.0, 1.0},
        {20.5, 1.0}, {21.0, 1.0},
    };

    std::vector<int> cluster_indices{1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                     2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3};

    DBSCAN dbscan(4, 1.0f);
    auto result = dbscan.cluster(std::span(points));

    ASSERT_EQ(cluster_indices.size(), result.size());
    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        ASSERT_EQ(cluster_indices[i], result[i]);
    }
}