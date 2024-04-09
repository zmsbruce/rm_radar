#include "track/features.h"

#include <gtest/gtest.h>

using namespace radar::track;

// Test default constructor
TEST(FeaturesTest, DefaultConstructor) {
    Features features;
    EXPECT_EQ(features.size(), 0);
    EXPECT_EQ(features.capacity(), 0);
}

// Test constructor with feature size
TEST(FeaturesTest, ConstructorWithSize) {
    Features features(5, 10);
    EXPECT_EQ(features.size(), 0);
    EXPECT_EQ(features.capacity(), 10);
}

// Test constructor with initial feature```cpp
// Test constructor with initial feature vector
TEST(FeaturesTest, ConstructorWithInitialFeature) {
    Eigen::VectorXf vec(5);
    vec << 1, 2, 3, 4, 5;
    Features features(vec, 5);
    EXPECT_EQ(features.size(), 1);
    EXPECT_EQ(features.capacity(), 5);
    EXPECT_TRUE(features.get(0).isApprox(vec));
}

// Test push_back and capacity doubling
TEST(FeaturesTest, PushBackAndCapacity) {
    Eigen::VectorXf vec(3);
    vec << 1, 2, 3;
    Features features(3);  // Initial capacity is 1

    // Push back the first vector
    features.push_back(vec);
    EXPECT_EQ(features.size(), 1);
    EXPECT_EQ(features.capacity(), 1);

    // Push back the second vector, capacity should remain
    features.push_back(vec);
    EXPECT_EQ(features.size(), 2);
    EXPECT_EQ(features.capacity(), 2);

    // Push back the third vector, capacity should double
    features.push_back(vec);
    EXPECT_EQ(features.size(), 3);
    EXPECT_EQ(features.capacity(), 4);  // Capacity should have doubled
}

// Test get method with valid and invalid index
TEST(FeaturesTest, GetMethod) {
    Eigen::VectorXf vec(3);
    vec << 1, 2, 3;
    Features features(vec, 5);

    // Valid get
    Eigen::VectorXf retrievedVec = features.get(0);
    EXPECT_TRUE(retrievedVec.isApprox(vec));

    // Invalid get
    EXPECT_THROW(features.get(1), std::out_of_range);
}

// Test size and capacity after clear
TEST(FeaturesTest, ClearMethod) {
    Eigen::VectorXf vec(3);
    vec << 1, 2, 3;
    Features features(vec, 5);
    features.clear();

    EXPECT_EQ(features.size(), 0);
    EXPECT_EQ(features.capacity(),
              5);  // Capacity remains unchanged after clear
    EXPECT_TRUE(features.get()
                    .isZero());  // Check if the internal matrix is set to zero
}

// Test copy assignment
TEST(FeaturesTest, CopyAssignment) {
    Eigen::VectorXf vec(3);
    vec << 1, 2, 3;
    Features features1(vec, 5);
    Features features2;
    features2 = features1;

    EXPECT_EQ(features2.size(), features1.size());
    EXPECT_EQ(features2.capacity(), features1.capacity());
    EXPECT_TRUE(features2.get(0).isApprox(vec));
}

// Test move assignment
TEST(FeaturesTest, MoveAssignment) {
    Eigen::VectorXf vec(3);
    vec << 1, 2, 3;
    Features features1(vec, 5);
    Features features2;
    features2 = std::move(features1);

    EXPECT_EQ(features2.size(), 1);
    EXPECT_EQ(features2.capacity(), 5);
    EXPECT_TRUE(features2.get(0).isApprox(vec));

    // After move, features1 should be in a valid state but with size and
    // capacity 0
    EXPECT_EQ(features1.size(), 0);
    EXPECT_EQ(features1.capacity(), 0);
}
