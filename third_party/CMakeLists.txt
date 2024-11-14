add_subdirectory(spdlog)
add_subdirectory(magic_enum)
add_subdirectory(backward-cpp)
add_subdirectory(Livox-SDK)

add_library(fastcrc STATIC
    ${PROJECT_SOURCE_DIR}/third_party/FastCRC/src/FastCRCsw.cpp
)

target_compile_options(fastcrc PRIVATE "-include" "cstddef")

target_include_directories(fastcrc PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/FastCRC/src
)