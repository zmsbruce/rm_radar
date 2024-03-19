add_executable(vtune_test 
    vtune_test.cpp
)

add_executable(ncu_test
    ncu_test.cu
)

target_compile_options(vtune_test PRIVATE -g -O0)

target_compile_options(ncu_test PRIVATE -g -G -O0)

target_link_libraries(vtune_test PRIVATE
    detector
)

target_link_libraries(ncu_test PRIVATE
    detector
)
