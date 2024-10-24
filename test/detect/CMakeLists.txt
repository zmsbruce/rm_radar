find_package(TIFF REQUIRED)
find_package(PkgConfig REQUIRED)

add_executable(detect_test
    kernel_test.cu
    detector_test.cpp
)

target_link_libraries(detect_test PRIVATE
    detector
    GTest::gtest_main
    ${TIFF_LIBRARIES}
)

add_test(
    NAME DetectTest 
    COMMAND detect_test 
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)