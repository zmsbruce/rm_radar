find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

add_library(robot SHARED
    robot.cpp
)

target_include_directories(robot PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/task/rdlt
)

target_link_libraries(robot PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)