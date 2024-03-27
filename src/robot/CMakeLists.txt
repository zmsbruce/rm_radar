find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(robot SHARED
    robot.cpp
)

target_include_directories(robot PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(robot PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)