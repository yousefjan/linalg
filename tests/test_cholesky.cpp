import linalgebra;

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <random>

using linalgebra::CholeskyResult;
using linalgebra::DimensionMismatchError;
using linalgebra::LinAlgError;
using linalgebra::Matrix;
using linalgebra::Vector;

namespace {

double reconstruction_error(const Matrix& A, const CholeskyResult& chol) {
    const Matrix LLT = chol.L * linalgebra::transpose(chol.L);
    const std::size_t n = A.rows();
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double d = A(i, j) - LLT(i, j);
            err += d * d;
        }
    }
    return std::sqrt(err);
}

Matrix make_spd(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix B(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            B(i, j) = dist(rng);
        }
    }
    Matrix A = linalgebra::transpose(B) * B;
    for (std::size_t i = 0; i < n; ++i) {
        A(i, i) += static_cast<double>(n);
    }
    return A;
}

}  // namespace

TEST_CASE("Cholesky factor 2x2", "[cholesky]") {
    Matrix A{{4.0, 2.0}, {2.0, 3.0}};
    auto chol = linalgebra::cholesky_factor(A);

    REQUIRE(reconstruction_error(A, chol) < 1e-12);

    REQUIRE(chol.L(0, 0) == Catch::Approx(2.0));
    REQUIRE(chol.L(1, 0) == Catch::Approx(1.0));
    REQUIRE(chol.L(1, 1) == Catch::Approx(std::sqrt(2.0)));
    REQUIRE(chol.L(0, 1) == Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("Cholesky factor 3x3", "[cholesky]") {
    Matrix A{{25.0, 15.0, -5.0},
             {15.0, 18.0,  0.0},
             {-5.0,  0.0, 11.0}};
    auto chol = linalgebra::cholesky_factor(A);
    REQUIRE(reconstruction_error(A, chol) < 1e-12);
}

TEST_CASE("Cholesky factor identity", "[cholesky]") {
    auto I = Matrix::identity(5);
    auto chol = linalgebra::cholesky_factor(I);
    REQUIRE(reconstruction_error(I, chol) < 1e-14);
    for (std::size_t i = 0; i < 5; ++i) {
        REQUIRE(chol.L(i, i) == Catch::Approx(1.0));
    }
}

TEST_CASE("Cholesky factor random SPD", "[cholesky]") {
    std::mt19937 rng(42);
    for (std::size_t n : {4, 8, 16, 32}) {
        Matrix A = make_spd(n, rng);
        auto chol = linalgebra::cholesky_factor(A);
        REQUIRE(reconstruction_error(A, chol) < 1e-10);
    }
}

TEST_CASE("Cholesky solve", "[cholesky]") {
    Matrix A{{4.0, 2.0}, {2.0, 3.0}};
    Vector b{8.0, 7.0};
    auto chol = linalgebra::cholesky_factor(A);
    Vector x = linalgebra::cholesky_solve(chol, b);

    double residual = linalgebra::norm2(A * x - b);
    REQUIRE(residual < 1e-12);
}

TEST_CASE("Cholesky solve random SPD", "[cholesky]") {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);

    for (std::size_t n : {5, 10, 20}) {
        Matrix A = make_spd(n, rng);
        Vector b(n);
        for (std::size_t i = 0; i < n; ++i) {
            b[i] = dist(rng);
        }

        auto chol = linalgebra::cholesky_factor(A);
        Vector x = linalgebra::cholesky_solve(chol, b);

        double residual = linalgebra::norm2(A * x - b);
        REQUIRE(residual < 1e-10);
    }
}

TEST_CASE("Cholesky rejects non-square matrix", "[cholesky]") {
    Matrix A(3, 4);
    REQUIRE_THROWS_AS(linalgebra::cholesky_factor(A), DimensionMismatchError);
}

TEST_CASE("Cholesky rejects non-symmetric matrix", "[cholesky]") {
    Matrix A{{1.0, 2.0}, {3.0, 4.0}};
    REQUIRE_THROWS_AS(linalgebra::cholesky_factor(A), LinAlgError);
}

TEST_CASE("Cholesky rejects non-positive-definite matrix", "[cholesky]") {
    Matrix A{{1.0, 0.0}, {0.0, -1.0}};
    REQUIRE_THROWS_AS(linalgebra::cholesky_factor(A), LinAlgError);
}

TEST_CASE("Cholesky solve dimension mismatch", "[cholesky]") {
    Matrix A{{4.0, 2.0}, {2.0, 3.0}};
    auto chol = linalgebra::cholesky_factor(A);
    Vector b{1.0, 2.0, 3.0};
    REQUIRE_THROWS_AS(linalgebra::cholesky_solve(chol, b), DimensionMismatchError);
}
