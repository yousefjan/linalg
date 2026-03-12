cmake_minimum_required(VERSION 3.20)

project(linear_algebra_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LINEAR_ALGEBRA_BUILD_TESTS "Build unit tests" ON)
option(LINEAR_ALGEBRA_BUILD_EXAMPLES "Build example programs" ON)

add_library(linear_algebra
    src/vector.cpp
    src/matrix.cpp
)

add_library(linear_algebra::core ALIAS linear_algebra)

target_include_directories(linear_algebra
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(linear_algebra PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(linear_algebra PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
elseif(MSVC)
    target_compile_options(linear_algebra PRIVATE /W4 /permissive-)
endif()

if(LINEAR_ALGEBRA_BUILD_EXAMPLES)
    add_executable(solve_linear_system examples/solve_linear_system.cpp)
    target_link_libraries(solve_linear_system PRIVATE linear_algebra::core)
endif()

if(LINEAR_ALGEBRA_BUILD_TESTS)
    include(FetchContent)

    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.4
    )

    FetchContent_MakeAvailable(Catch2)

    enable_testing()

    add_executable(linear_algebra_tests
        tests/test_vector.cpp
        tests/test_matrix.cpp
    )

    target_link_libraries(linear_algebra_tests
        PRIVATE
            linear_algebra::core
            Catch2::Catch2WithMain
    )

    include(Catch)
    catch_discover_tests(linear_algebra_tests)
endif()
