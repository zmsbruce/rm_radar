find_package(Hikvision REQUIRED)
find_package(OpenCV REQUIRED)

add_library(driver SHARED
    camera/hik/hik.cpp
)

target_include_directories(driver PUBLIC
    ${Hikvision_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(driver PUBLIC
    ${OpenCV_LIBS}
    ${Hikvision_LIBS}
    spdlog::spdlog
    magic_enum::magic_enum
)