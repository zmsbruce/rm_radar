if (NOT GTest_FOUND)
    find_package(GTest REQUIRED)
endif()

add_subdirectory(detect)
add_subdirectory(utils)