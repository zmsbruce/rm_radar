/**
 * @file auction.h
 * @author zmsbruce (zmsbruce@163.com)
 * @brief This file implements the auction algorithm.
 * @date 2024-04-09
 *
 * @copyright (c) 2024 HITCRT
 * All rights reserved.
 *
 */

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <vector>

namespace radar::track {

constexpr int kNotMatched = -1;

/**
 * @brief Runs the auction algorithm to assign tasks to agents based on the
 * provided value matrix.
 *
 * @param value_matrix A matrix where each row represents an agent and each
 * column represents a task. The element at row i, column j represents the value
 * agent i assigns to task j.
 * @param max_iter The maximum number of iterations to run the algorithm for.
 *
 * @return std::vector<int> A vector where each element i represents the task
 * assigned to agent i. If agent i is assigned a virtual task (in case of more
 * agents than tasks), element i will be -1.
 *
 * @details The algorithm works by iteratively having agents "bid" on tasks
 * based on their value minus the current price of the task. If there are more
 * agents than tasks, virtual tasks with zero value are added to the value
 * matrix to make it a square matrix. The algorithm tries to assign each agent
 * to a task in a way that maximizes the overall value while ensuring that no
 * task is assigned to more than one agent. If the algorithm reaches the maximum
 * number of iterations before all agents are assigned, it will stop and return
 * the current assignment, with any unassigned agents having a task value of -1.
 *
 * @ref Bertsekas D P . The auction algorithm: A distributed relaxation method
 * for the assignment problem[J]. Annals of Operations Research, 1988,
 * 14(1):105-123.
 */
std::vector<int> auction(Eigen::MatrixXf value_matrix, int max_iter) {
    int num_agents = value_matrix.rows();
    int num_tasks = value_matrix.cols();
    int num_tasks_real = num_tasks;

    // If there are more agents than tasks, add virtual tasks with 0 value
    if (num_agents > num_tasks) {
        Eigen::MatrixXf extended_value_matrix =
            Eigen::MatrixXf::Zero(num_agents, num_agents);
        extended_value_matrix.block(0, 0, num_agents, num_tasks) = value_matrix;
        value_matrix = std::move(extended_value_matrix);
        num_tasks = num_agents;  // Now we have a square matrix
    }

    Eigen::VectorXf prices = Eigen::VectorXf::Zero(num_tasks);
    std::vector<int> assignment(num_agents, kNotMatched);
    std::vector<bool> assigned_tasks(num_tasks, false);

    int iterations = 0;

    while (iterations < max_iter) {
        if (std::count_if(assignment.begin(), assignment.end(), [&](int val) {
                return val >= 0 && val <= num_tasks_real;
            }) >= num_agents) {
            break;
        }

        bool any_assignment_changed = false;

        for (int agent = 0; agent < num_agents; ++agent) {
            if (assignment[agent] != kNotMatched) {
                continue;
            }

            int best_task = kNotMatched;
            float best_value = -std::numeric_limits<float>::infinity();
            for (int task = 0; task < num_tasks; ++task) {
                float value = value_matrix(agent, task) - prices(task);
                if (value > best_value) {
                    best_value = value;
                    best_task = task;
                }
            }

            if (best_task != kNotMatched) {
                // Increase the price for the best task
                prices(best_task) += best_value;

                // Reassign any agent currently assigned to best_task
                for (int other_agent = 0; other_agent < num_agents;
                     ++other_agent) {
                    if (assignment[other_agent] == best_task) {
                        assignment[other_agent] = kNotMatched;
                        break;
                    }
                }

                // Assign the best task to the current agent
                assignment[agent] = best_task;
                assigned_tasks[best_task] = true;
                any_assignment_changed = true;
            }
        }

        if (!any_assignment_changed) {  // If no assignment changed, the
                                        // algorithm is stuck
            break;
        }

        iterations++;
    }

    // Set virtual task index to -1
    std::for_each(assignment.begin(), assignment.end(), [&](int& val) {
        val = val >= num_tasks_real ? kNotMatched : val;
    });

    return assignment;
}

}  // namespace radar::track