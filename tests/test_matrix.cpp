import linalgebra;

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using linalgebra::Matrix;
using linalgebra::Vector;
using linalgebra::DimensionMismatchError;

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

TEST_CASE("Matrix transpose swaps rows and columns", "[matrix]") {
    const Matrix a{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const Matrix at = linalgebra::transpose(a);

    REQUIRE(at.rows() == 3);
    REQUIRE(at.cols() == 2);
    CHECK(at(0, 0) == 1.0);
    CHECK(at(1, 0) == 2.0);
    CHECK(at(2, 0) == 3.0);
    CHECK(at(0, 1) == 4.0);
    CHECK(at(1, 1) == 5.0);
    CHECK(at(2, 1) == 6.0);
}

TEST_CASE("Matrix addition and subtraction enforce equal shapes", "[matrix]") {
    const Matrix a{{1.0, 2.0}, {3.0, 4.0}};
    const Matrix b{{0.5, -1.0}, {2.0, 1.5}};

    const Matrix sum = a + b;
    CHECK(sum(0, 0) == Catch::Approx(1.5));
    CHECK(sum(0, 1) == Catch::Approx(1.0));
    CHECK(sum(1, 0) == Catch::Approx(5.0));
    CHECK(sum(1, 1) == Catch::Approx(5.5));

    const Matrix diff = a - b;
    CHECK(diff(0, 0) == Catch::Approx(0.5));
    CHECK(diff(0, 1) == Catch::Approx(3.0));
    CHECK(diff(1, 0) == Catch::Approx(1.0));
    CHECK(diff(1, 1) == Catch::Approx(2.5));

    const Matrix wrong_shape(3, 1);
    CHECK_THROWS_AS(a + wrong_shape, DimensionMismatchError);
    CHECK_THROWS_AS(a - wrong_shape, DimensionMismatchError);
}

TEST_CASE("Matrix-vector multiply checks dimensions", "[matrix]") {
    const Matrix a{{1.0, 2.0, 3.0}, {0.0, -1.0, 4.0}};
    const Vector x{2.0, -1.0, 0.5};

    const Vector y = a * x;
    REQUIRE(y.size() == 2);
    CHECK(y[0] == Catch::Approx(1.5));
    CHECK(y[1] == Catch::Approx(3.0));

    const Vector wrong_size{1.0, 2.0};
    CHECK_THROWS_AS(a * wrong_size, DimensionMismatchError);
}

TEST_CASE("Matrix-matrix multiply handles identity and shape checks", "[matrix]") {
    const Matrix a{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    const Matrix identity = Matrix::identity(2);
    const Matrix product = a * identity;

    REQUIRE(product.rows() == a.rows());
    REQUIRE(product.cols() == a.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            CHECK(product(i, j) == Catch::Approx(a(i, j)));
        }
    }

    const Matrix lhs{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    const Matrix rhs{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    const Matrix dense_product = lhs * rhs;

    REQUIRE(dense_product.rows() == 2);
    REQUIRE(dense_product.cols() == 2);
    CHECK(dense_product(0, 0) == Catch::Approx(58.0));
    CHECK(dense_product(0, 1) == Catch::Approx(64.0));
    CHECK(dense_product(1, 0) == Catch::Approx(139.0));
    CHECK(dense_product(1, 1) == Catch::Approx(154.0));

    const Matrix incompatible(4, 1);
    CHECK_THROWS_AS(lhs * incompatible, DimensionMismatchError);
}

TEST_CASE("Matrix-matrix multiply handles SIMD tail dimensions", "[matrix]") {
    const Matrix lhs{
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {6.0, 7.0, 8.0, 9.0, 10.0}
    };
    const Matrix rhs{
        {1.0, 0.0, 2.0},
        {0.0, 1.0, 3.0},
        {1.0, 1.0, 4.0},
        {0.0, 2.0, 5.0},
        {1.0, 0.0, 6.0}
    };

    const Matrix product = lhs * rhs;
    REQUIRE(product.rows() == 2);
    REQUIRE(product.cols() == 3);

    CHECK(product(0, 0) == Catch::Approx(9.0));
    CHECK(product(0, 1) == Catch::Approx(13.0));
    CHECK(product(0, 2) == Catch::Approx(70.0));
    CHECK(product(1, 0) == Catch::Approx(24.0));
    CHECK(product(1, 1) == Catch::Approx(33.0));
    CHECK(product(1, 2) == Catch::Approx(170.0));
}
