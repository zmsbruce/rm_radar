find_package(OpenCV REQUIRED)

add_executable(comm_test crc_test.cpp)

target_include_directories(comm_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(comm_test PRIVATE
    ${OpenCV_LIBS}
    communicator
    GTest::gtest_main
)

add_test(
    NAME COMM_TEST
    COMMAND comm_test        
)