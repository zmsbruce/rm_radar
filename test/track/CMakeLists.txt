find_package(Eigen3 REQUIRED)

add_executable(track_test
    kf_test.cpp
    ekf_test.cpp
    features_test.cpp
)

target_include_directories(track_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(track_test PRIVATE
    Eigen3::Eigen
    GTest::gtest_main
)

add_test(
    NAME TrackTest
    COMMAND track_test        
)