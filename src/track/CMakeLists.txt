find_package(OpenCV REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

add_library(tracker SHARED
    tracker.cpp
)

target_include_directories(tracker PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(tracker PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
)