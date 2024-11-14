#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>
#include <random>

#define private public
#define protected public

#include "task/comm/referee.h"

#undef private
#undef protected

class EncoderTest : public ::testing::Test {
   private:
   protected:
    std::unique_ptr<radar::RefereeCommunicator> comm = nullptr;

    virtual void SetUp() {
        comm = std::make_unique<radar::RefereeCommunicator>("/dev/ttyUSB0");
    }
};

TEST_F(EncoderTest, TestEncode) {}

TEST_F(EncoderTest, TestDecode) {}
