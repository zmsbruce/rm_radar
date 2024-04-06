find_package(OpenCV REQUIRED)

add_executable(locate_test
    locate_test.cpp
)

target_include_directories(locate_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(locate_test PRIVATE
    ${OpenCV_LIBS} 
    GTest::gtest_main
)

add_test(
    NAME LocateTest
    COMMAND locate_test        
)