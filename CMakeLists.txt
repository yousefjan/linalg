cmake_minimum_required(VERSION 3.20)

project(linear_algebra_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(LINEAR_ALGEBRA_BUILD_TESTS "Build unit tests" ON)
set(LINEAR_ALGEBRA_SIMD "AUTO" CACHE STRING "SIMD backend for matmul: AUTO, NONE, AVX, AVX2, AVX512")
set_property(CACHE LINEAR_ALGEBRA_SIMD PROPERTY STRINGS AUTO NONE AVX AVX2 AVX512)

add_library(linear_algebra
    src/vector.cpp
    src/matrix.cpp
    src/norms.cpp
    src/triangular_solve.cpp
    src/lu.cpp
    src/qr.cpp
    src/qr_iteration.cpp
)

add_library(linear_algebra::core ALIAS linear_algebra)

target_include_directories(linear_algebra
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(linear_algebra PUBLIC cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(linear_algebra PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

    if(LINEAR_ALGEBRA_SIMD STREQUAL "AVX")
        target_compile_options(linear_algebra PRIVATE -mavx)
    elseif(LINEAR_ALGEBRA_SIMD STREQUAL "AVX2")
        target_compile_options(linear_algebra PRIVATE -mavx2)
    elseif(LINEAR_ALGEBRA_SIMD STREQUAL "AVX512")
        target_compile_options(linear_algebra PRIVATE -mavx512f)
    elseif(LINEAR_ALGEBRA_SIMD STREQUAL "NONE")
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
        message(WARNING "LINEAR_ALGEBRA_SIMD explicit x86 flags are only wired for Clang/GNU right now.")
    endif()
endif()

option(LINEAR_ALGEBRA_BUILD_EXPERIMENTS "Build experiment programs" ON)
if(LINEAR_ALGEBRA_BUILD_EXPERIMENTS)
    add_executable(pivoting_vs_no_pivoting experiments/pivoting_vs_no_pivoting.cpp)
    target_link_libraries(pivoting_vs_no_pivoting PRIVATE linear_algebra::core)

    add_executable(hilbert_qr experiments/hilbert_qr.cpp)
    target_link_libraries(hilbert_qr PRIVATE linear_algebra::core)

    add_executable(matmul experiments/matmul.cpp)
    target_link_libraries(matmul PRIVATE linear_algebra::core)

#     find_package(BLAS)
#     find_package(LAPACK)
#     if(BLAS_FOUND AND LAPACK_FOUND)
#         add_executable(blas_comparison experiments/blas_comparison.cpp)
#         target_link_libraries(blas_comparison PRIVATE
#             linear_algebra::core
#             ${BLAS_LIBRARIES}
#             ${LAPACK_LIBRARIES}
#         )
#         if(DEFINED BLAS_INCLUDE_DIRS)
#             target_include_directories(blas_comparison PRIVATE ${BLAS_INCLUDE_DIRS})
#         endif()
#         if(APPLE)
#             # Suppress deprecation warnings from Accelerate headers on recent macOS.
#             target_compile_options(blas_comparison PRIVATE -Wno-deprecated-declarations)
#         endif()
#     else()
#         message(STATUS "BLAS/LAPACK not found — skipping blas_comparison experiment")
#     endif()
# endif()

if(LINEAR_ALGEBRA_BUILD_TESTS)
    include(FetchContent)
endif()

if(LINEAR_ALGEBRA_BUILD_TESTS)

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
    )

    target_link_libraries(linear_algebra_tests
        PRIVATE
            linear_algebra::core
            Catch2::Catch2WithMain
    )

    include(Catch)
    catch_discover_tests(linear_algebra_tests)
endif()
