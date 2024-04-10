find_package(TIFF REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFI REQUIRED IMPORTED_TARGET libffi)

add_executable(detect_test
    kernel_test.cu
    detector_test.cpp
)

target_link_libraries(detect_test PRIVATE
    detector
    GTest::gtest_main
    ${TIFF_LIBRARIES}
    PkgConfig::FFI
)

add_test(
    NAME DetectTest 
    COMMAND detect_test 
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
)