find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(tracker SHARED
    hungarian.cpp
    kalman_filter.cpp
    linear_assignment.cpp
    nn_matching.cpp
    track.cpp
    tracker.cpp
)

target_include_directories(tracker PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(tracker PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)