#include "matrix.hpp"

#include <catch2/catch_test_macros.hpp>

#include <type_traits>
#include <utility>

using linalg::Matrix;

TEST_CASE("Matrix constructors initialize dimensions and values", "[matrix]") {
    const Matrix empty;
    CHECK(empty.rows() == 0);
    CHECK(empty.cols() == 0);
    CHECK(empty.empty());

    const Matrix zeroed(2, 3);
    REQUIRE(zeroed.rows() == 2);
    REQUIRE(zeroed.cols() == 3);
    for (std::size_t i = 0; i < zeroed.rows(); ++i) {
        for (std::size_t j = 0; j < zeroed.cols(); ++j) {
            CHECK(zeroed(i, j) == 0.0);
        }
    }

    const Matrix filled(2, 2, 1.5);
    CHECK(filled(0, 0) == 1.5);
    CHECK(filled(0, 1) == 1.5);
    CHECK(filled(1, 0) == 1.5);
    CHECK(filled(1, 1) == 1.5);
}

TEST_CASE("Matrix supports checked element access and fill", "[matrix]") {
    Matrix a(2, 3);
    a(0, 0) = 1.0;
    a(0, 1) = 2.0;
    a(1, 2) = 5.0;

    CHECK(a(0, 0) == 1.0);
    CHECK(a(0, 1) == 2.0);
    CHECK(a(1, 2) == 5.0);

    a.fill(-3.0);
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            CHECK(a(i, j) == -3.0);
        }
    }

    CHECK_THROWS_AS(a(2, 0), std::out_of_range);
    CHECK_THROWS_AS(a(0, 3), std::out_of_range);
}

TEST_CASE("Matrix identity and zero factories work", "[matrix]") {
    const Matrix identity = Matrix::identity(3);
    REQUIRE(identity.rows() == 3);
    REQUIRE(identity.cols() == 3);

    for (std::size_t i = 0; i < identity.rows(); ++i) {
        for (std::size_t j = 0; j < identity.cols(); ++j) {
            const double expected = (i == j) ? 1.0 : 0.0;
            CHECK(identity(i, j) == expected);
        }
    }

    const Matrix zeros = Matrix::zeros(2, 4);
    CHECK(zeros.rows() == 2);
    CHECK(zeros.cols() == 4);
    for (std::size_t i = 0; i < zeros.rows(); ++i) {
        for (std::size_t j = 0; j < zeros.cols(); ++j) {
            CHECK(zeros(i, j) == 0.0);
        }
    }
}

TEST_CASE("Matrix copy, move, and initializer list keeps row-major layout", "[matrix]") {
    static_assert(std::is_nothrow_move_constructible_v<Matrix>);

    Matrix original{{1.0, 2.0}, {3.0, 4.0}};
    Matrix copied = original;
    copied(0, 0) = 10.0;

    CHECK(original(0, 0) == 1.0);
    CHECK(copied(0, 0) == 10.0);

    Matrix moved = std::move(original);
    REQUIRE(moved.rows() == 2);
    REQUIRE(moved.cols() == 2);
    CHECK(moved(0, 0) == 1.0);
    CHECK(moved(0, 1) == 2.0);
    CHECK(moved(1, 0) == 3.0);
    CHECK(moved(1, 1) == 4.0);
}

TEST_CASE("Matrix initializer list rejects unequal row lengths", "[matrix]") {
    CHECK_THROWS_AS((Matrix{{1.0, 2.0}, {3.0}}), std::invalid_argument);
}
