add_executable(sample_detect_image sample_detect_image.cpp)

add_executable(sample_detect_video sample_detect_video.cpp)

target_link_libraries(sample_detect_image PRIVATE
    detector
)

target_link_libraries(sample_detect_video PRIVATE
    detector
)