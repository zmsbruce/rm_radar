find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK)

add_library(locator SHARED
    locate.cpp
)

target_include_directories(locator PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(locator PUBLIC
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    robot
)