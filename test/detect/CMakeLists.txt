add_executable(detect_test
    kernel_test.cu
    detector_test.cpp
)

target_link_libraries(detect_test PRIVATE
    detector
    GTest::gtest_main
)

add_test(
    NAME DetectTest 
    COMMAND detect_test 
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)