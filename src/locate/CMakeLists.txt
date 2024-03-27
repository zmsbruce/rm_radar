find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK)

add_library(locator SHARED
    locate.cpp
)

target_include_directories(locator PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

target_include_directories(locator PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)

target_link_libraries(locator PUBLIC
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    robot
)