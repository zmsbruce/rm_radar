find_package(OpenCV REQUIRED)

add_library(communicator SHARED
    referee.cpp
)

target_include_directories(communicator PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
)

target_link_libraries(communicator PUBLIC
    ${OpenCV_LIBS}
    fastcrc
    robot
    driver
    spdlog::spdlog
)


