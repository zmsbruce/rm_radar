#include <gtest/gtest.h>

#include <filesystem>

#include "utils.h"

using namespace radar;

namespace {
const std::string testFilePath = "test.bin";
}

// Test writeToFile
TEST(FileIOTest, WriteToFile) {
    const std::string content = "Hello, World!";
    std::span<const char> data(content.data(), content.size());

    // Writing content to file
    writeToFile(data, testFilePath);

    // Checking if file exists and content is correct
    std::ifstream ifs(testFilePath, std::ios::binary);
    std::string fileContent((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
    ifs.close();

    EXPECT_EQ(content, fileContent);

    // Cleanup
    std::remove(testFilePath.c_str());
}

// Test loadFromFile
TEST(FileIOTest, LoadFromFile) {
    // Create a test file
    const std::string content = "Hello, World!";
    std::ofstream ofs(testFilePath, std::ios::binary);
    ofs.write(content.data(), content.size());
    ofs.close();

    // Load content from file
    auto [buffer, size] = loadFromFile(testFilePath);
    std::string fileContent(buffer.get(), size);

    EXPECT_EQ(content.size(), size);
    EXPECT_EQ(content, fileContent);

    // Cleanup
    std::filesystem::remove(testFilePath.c_str());
}

// Test for exception handling
TEST(FileIOTest, ExceptionHandling) {
    // Test write with invalid path
    const std::string content = "Hello, World!";
    std::span<const char> data(content.data(), content.size());
    EXPECT_THROW(writeToFile(data, ""), std::ios_base::failure);

    // Test read with invalid path
    EXPECT_THROW(loadFromFile(""), std::ios_base::failure);
}