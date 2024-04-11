#include "track/auction.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <vector>

using namespace radar::track;

TEST(AuctionTest, EqualNumberOfAgentsAndTasks) {
    Eigen::MatrixXf value_matrix(3, 3);
    value_matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    std::vector<int> expected_assignment = {2, 1, 0};
    int max_iter = 100;

    std::vector<int> result = auction(value_matrix, max_iter);

    EXPECT_EQ(result.size(), expected_assignment.size());
    for (size_t i = 0; i < expected_assignment.size(); ++i) {
        EXPECT_EQ(result[i], expected_assignment[i]);
    }
}

TEST(AuctionTest, MoreAgentsThanTasks) {
    Eigen::MatrixXf value_matrix(4, 3);
    value_matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 4, 7;
    int max_iter = 100;

    std::vector<int> result = auction(value_matrix, max_iter);

    EXPECT_EQ(result.size(), 4);
    // Check that all tasks have been assigned to an agent
    for (int task = 0; task < 3; ++task) {
        EXPECT_TRUE(std::find(result.begin(), result.end(), task) !=
                    result.end());
    }
}

TEST(AuctionTest, MoreTasksThanAgents) {
    Eigen::MatrixXf value_matrix(3, 4);
    value_matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
    int max_iter = 100;

    std::vector<int> result = auction(value_matrix, max_iter);

    EXPECT_EQ(result.size(), 3);
    // Check that each agent has been assigned to a task
    for (int agent = 0; agent < 3; ++agent) {
        EXPECT_NE(result[agent], kNotMatched);
    }
}

TEST(AuctionTest, HandlesZeroIterations) {
    Eigen::MatrixXf value_matrix(3, 3);
    value_matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    int max_iter = 0;

    std::vector<int> result = auction(value_matrix, max_iter);

    // With zero iterations, we expect no assignments.
    for (int agent : result) {
        EXPECT_EQ(agent, kNotMatched);
    }
}