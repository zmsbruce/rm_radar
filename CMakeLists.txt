cmake_minimum_required(VERSION 3.19)
project(rm_radar LANGUAGES C CXX CUDA)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_RELEASE "-w -Ofast -DNDEBUG -flto -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra")

set(CMAKE_CUDA_ARCHITECTURES 75;80;89)
set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr --std=c++20")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0") 
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# Force all targets to generate position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

enable_testing()

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(third_party)
