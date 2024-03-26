add_executable(sample_detect sample_detect.cpp)

add_executable(sample_detect_track sample_detect_track.cpp)

add_executable(sample_detect_locate sample_detect_locate.cpp)

target_link_libraries(sample_detect PRIVATE
    detector
)

target_link_libraries(sample_detect_track PRIVATE
    detector
    tracker
    robot
)

target_link_libraries(sample_detect_locate PRIVATE
    detector
    robot
    locator
)