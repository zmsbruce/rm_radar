add_executable(util_test
    util_test.cpp
)

target_include_directories(util_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(util_test
    GTest::gtest_main
)

add_test(NAME UtilTest COMMAND util_test)