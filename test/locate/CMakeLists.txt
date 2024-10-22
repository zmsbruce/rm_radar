find_package(PCL REQUIRED COMPONENTS common io kdtree segmentation)
find_package(OpenCV REQUIRED)

add_executable(locate_test locator_test.cpp)

target_include_directories(locate_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(locate_test PRIVATE
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    locator
    GTest::gtest_main
)

add_test(
    NAME LocateTest
    COMMAND locate_test        
)