#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>
#include <random>

#define private public
#define protected public

#include "task/comm/referee.h"

#undef private
#undef protected

class CrcTest : public ::testing::Test {
    virtual void SetUp() {}
};

TEST_F(CrcTest, TestCRC8) {
    std::vector<std::byte> data = {std::byte(0x01), std::byte(0x02),
                                   std::byte(0x03), std::byte(0x00)};
    radar::RefereeCommunicator::appendCRC8(data);
    EXPECT_EQ(data[3], std::byte(0x0A));

    std::vector<std::byte> dataToVerify = {std::byte(0x01), std::byte(0x02),
                                           std::byte(0x03), std::byte(0x0A)};
    EXPECT_TRUE(radar::RefereeCommunicator::verifyCRC8(dataToVerify));
}

TEST_F(CrcTest, TestCRC16) {
    std::vector<std::byte> data = {std::byte(0x01), std::byte(0x02),
                                   std::byte(0x03), std::byte(0x00),
                                   std::byte(0x00)};
    radar::RefereeCommunicator::appendCRC16(data);
    EXPECT_EQ(data[3], std::byte(0xC4));
    EXPECT_EQ(data[4], std::byte(0x62));

    std::vector<std::byte> dataToVerify = {std::byte(0x01), std::byte(0x02),
                                           std::byte(0x03), std::byte(0xC4),
                                           std::byte(0x62)};
    EXPECT_TRUE(radar::RefereeCommunicator::verifyCRC16(dataToVerify));
}