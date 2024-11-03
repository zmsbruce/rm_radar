find_package(TensorRT REQUIRED VERSION 8.5)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED VERSION 12.2)
find_package(TBB REQUIRED)

add_library(detector SHARED
    detector.cpp
    detector.cu
)

target_include_directories(detector PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(detector PUBLIC
    ${OpenCV_LIBS}
    CUDA::cudart
    ${TensorRT_LIBRARIES}
    TBB::tbb
    robot
    spdlog::spdlog
)