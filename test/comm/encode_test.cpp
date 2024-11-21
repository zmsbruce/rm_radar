#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>
#include <random>

#define private public
#define protected public

#include "task/comm/referee.h"

#undef private
#undef protected

class EncoderTest : public ::testing::Test {
    virtual void SetUp() {}
};

TEST_F(EncoderTest, TestEncode) {}

TEST_F(EncoderTest, TestDecode) {}
