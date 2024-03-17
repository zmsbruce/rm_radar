#pragma once

#include "data_type.h"
#include "munkres.h"

class HungarianOper {
   public:
    static Eigen::Matrix<float, -1, 2, Eigen::RowMajor> Solve(
        const DYNAMICM &cost_matrix);
};
