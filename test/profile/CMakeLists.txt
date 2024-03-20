find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(vtune_test 
    vtune_test.cpp
)

add_executable(ncu_test
    ncu_test.cu
)

target_compile_options(vtune_test PRIVATE -g -O0)

target_compile_options(ncu_test PRIVATE -g -G -lineinfo)

target_link_libraries(vtune_test PRIVATE
    detector
)

target_include_directories(ncu_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(ncu_test PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
)
