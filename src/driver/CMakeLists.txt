find_package(Hikvision REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(LibSerial REQUIRED)

add_library(driver STATIC
    camera/hik/hik.cpp
    lidar/livox/livox.cpp
    serial/serial.cpp
)

target_include_directories(driver PUBLIC
    ${Hikvision_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${LibSerial_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(driver PUBLIC
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Hikvision_LIBS}
    ${LibSerial_LIBRARY}
    livox_sdk_static
    spdlog::spdlog
    magic_enum::magic_enum
)