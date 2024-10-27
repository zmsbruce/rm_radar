find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io kdtree segmentation)

add_library(locator SHARED
    locate.cpp
)

target_include_directories(locator PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/task/rdlt
)

target_link_libraries(locator PUBLIC
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    robot
    spdlog::spdlog
)