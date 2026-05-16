cmake_minimum_required(VERSION 3.30)

set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "d0edc3af-4c50-42ea-a356-e2862fe7a444")

if(APPLE AND CMAKE_CXX_COMPILER MATCHES "/opt/homebrew/.*/llvm/")
    set(CMAKE_CXX_COMPILER_ID_ARG1 "-B/opt/homebrew/opt/llvm/lib/c++/")
    set(CMAKE_CXX_FLAGS_INIT "-B/opt/homebrew/opt/llvm/lib/c++/")
endif()

project(linear_algebra_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_MODULE_STD ON)

option(LINEAR_ALGEBRA_BUILD_TESTS "Build unit tests" ON)
set(LINEAR_ALGEBRA_SIMD "AUTO" CACHE STRING "SIMD backend for matmul: AUTO (uses NEON on ARM, scalar otherwise), or NONE")
set_property(CACHE LINEAR_ALGEBRA_SIMD PROPERTY STRINGS AUTO NONE)

add_library(linear_algebra)

add_library(linear_algebra::core ALIAS linear_algebra)

target_sources(linear_algebra
    PUBLIC
    FILE_SET CXX_MODULES FILES
        src/linalgebra.cpp
        src/linalgebra_error.cpp
        src/vector.cpp
        src/matrix.cpp
        src/norms.cpp
        src/triangular_solve.cpp
        src/lu.cpp
        src/qr.cpp
        src/qr_iteration.cpp
        src/cholesky.cpp
        src/sym_eigen.cpp
        src/svd.cpp
        src/iterative.cpp
        src/precond.cpp
        src/expm.cpp
)

target_compile_features(linear_algebra PUBLIC cxx_std_23)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(linear_algebra PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

    if(LINEAR_ALGEBRA_SIMD STREQUAL "NONE")
        target_compile_definitions(linear_algebra PRIVATE LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL=1)
    elseif(LINEAR_ALGEBRA_SIMD STREQUAL "AUTO")
    else()
        message(FATAL_ERROR "Unsupported LINEAR_ALGEBRA_SIMD value: ${LINEAR_ALGEBRA_SIMD}")
    endif()
elseif(MSVC)
    target_compile_options(linear_algebra PRIVATE /W4 /permissive-)
    if(LINEAR_ALGEBRA_SIMD STREQUAL "NONE")
        target_compile_definitions(linear_algebra PRIVATE LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL=1)
    elseif(NOT LINEAR_ALGEBRA_SIMD STREQUAL "AUTO")
        message(FATAL_ERROR "Unsupported LINEAR_ALGEBRA_SIMD value: ${LINEAR_ALGEBRA_SIMD}")
    endif()
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
        tests/test_triangular_solve.cpp
        tests/test_lu.cpp
        tests/test_qr.cpp
        tests/test_qr_iteration.cpp
        tests/test_cholesky.cpp
        tests/test_sym_eigen.cpp
        tests/test_svd.cpp
        tests/test_iterative.cpp
        tests/test_precond.cpp
        tests/test_expm.cpp
    )

    target_link_libraries(linear_algebra_tests
        PRIVATE
            linear_algebra::core
            Catch2::Catch2WithMain
    )

    include(Catch)
    catch_discover_tests(linear_algebra_tests)
endif()
