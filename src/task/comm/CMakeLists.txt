find_package(OpenCV REQUIRED)

add_library(communicator SHARED
    referee.cpp
)

target_include_directories(communicator PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(communicator PUBLIC
    ${OpenCV_LIBS}
    robot
    driver
    spdlog::spdlog
)
