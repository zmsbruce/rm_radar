find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

add_library(robot SHARED
    robot.cpp
)

target_include_directories(robot PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_include_directories(robot PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_link_libraries(robot PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)