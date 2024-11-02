find_package(Hikvision REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

add_library(driver STATIC
    camera/hik/hik.cpp
    lidar/livox/livox.cpp
)

target_include_directories(driver PUBLIC
    ${Hikvision_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(driver PUBLIC
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Hikvision_LIBS}
    livox_sdk_static
    spdlog::spdlog
    magic_enum::magic_enum
)