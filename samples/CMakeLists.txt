add_executable(sample sample.cpp)

target_link_libraries(sample PRIVATE
    detector
    locator
    tracker
    robot
)

target_include_directories(sample PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/samples
)