cmake_minimum_required(VERSION 3.19)
project(RMRadar LANGUAGES C CXX CUDA)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Ofast -DNDEBUG -flto -ffast-math -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")

set(CMAKE_CUDA_ARCHITECTURES "52;61;75;86")
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr --std=c++20")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0") 
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(samples)

set(${PROJECT_NAME}_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}")
set(${PROJECT_NAME}_LIBRARIES "${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}")

install(
    TARGETS detector tracker locator robot
    EXPORT ${PROJECT_NAME}Target
    LIBRARY DESTINATION lib/${PROJECT_NAME}
    INCLUDES DESTINATION include/${PROJECT_NAME}
)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/src/
    DESTINATION include/${PROJECT_NAME}
    FILES_MATCHING PATTERN "*.h"
)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/lib/
    DESTINATION lib/${PROJECT_NAME}
)

install(
    EXPORT ${PROJECT_NAME}Target
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake
    PATH_VARS ${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_NAME}_LIBRARIES
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
)

add_custom_target(uninstall COMMAND xargs rm -vf < install_manifest.txt ) 