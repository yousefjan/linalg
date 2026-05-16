import linalgebra;

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>

using linalgebra::DimensionMismatchError;
using linalgebra::Matrix;

namespace {

double frobenius_diff(const Matrix& A, const Matrix& B) {
    double s = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) {
            const double d = A(i, j) - B(i, j);
            s += d * d;
        }
    return std::sqrt(s);
}

// Taylor series
Matrix expm_taylor5(const Matrix& A) {
    const std::size_t n = A.rows();
    Matrix result = Matrix::identity(n);
    Matrix power  = Matrix::identity(n);
    double fact = 1.0;
    for (int k = 1; k <= 5; ++k) {
        power = power * A;
        fact *= static_cast<double>(k);
        const double inv_fact = 1.0 / fact;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                result(i, j) += inv_fact * power(i, j);
    }
    return result;
}

}  // namespace

TEST_CASE("expm: zero matrix gives identity", "[expm]") {
    Matrix Z = Matrix::zeros(3, 3);
    auto E = linalgebra::expm(Z);
    auto I = Matrix::identity(3);
    REQUIRE(frobenius_diff(E, I) < 1e-12);
}

TEST_CASE("expm: scalar multiple of identity", "[expm]") {
    // expm(s*I) = exp(s)*I
    const double s = 2.0;
    Matrix A = Matrix::zeros(3, 3);
    A(0, 0) = s; A(1, 1) = s; A(2, 2) = s;
    auto E = linalgebra::expm(A);

    const double expected = std::exp(s);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(E(i, j) == Catch::Approx(i == j ? expected : 0.0).margin(1e-10));
}

TEST_CASE("expm: 2x2 nilpotent", "[expm]") {
    // A = [[0,1],[0,0]], expm(A) = [[1,1],[0,1]] exactly.
    Matrix A{{0.0, 1.0}, {0.0, 0.0}};
    auto E = linalgebra::expm(A);
    REQUIRE(E(0, 0) == Catch::Approx(1.0).margin(1e-12));
    REQUIRE(E(0, 1) == Catch::Approx(1.0).margin(1e-12));
    REQUIRE(E(1, 0) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(E(1, 1) == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("expm: 2x2 rotation generator", "[expm]") {
    // A = [[0,-t],[t,0]], expm(A) = [[cos(t), -sin(t)],[sin(t), cos(t)]].
    const double t = std::numbers::pi / 4.0;
    Matrix A{{0.0, -t}, {t, 0.0}};
    auto E = linalgebra::expm(A);

    REQUIRE(E(0, 0) == Catch::Approx(std::cos(t)).epsilon(1e-10));
    REQUIRE(E(0, 1) == Catch::Approx(-std::sin(t)).epsilon(1e-10));
    REQUIRE(E(1, 0) == Catch::Approx(std::sin(t)).epsilon(1e-10));
    REQUIRE(E(1, 1) == Catch::Approx(std::cos(t)).epsilon(1e-10));
}

TEST_CASE("expm: diagonal matrix", "[expm]") {
    // A = diag(1, 2), expm(A) = diag(e, e^2).
    Matrix A{{1.0, 0.0}, {0.0, 2.0}};
    auto E = linalgebra::expm(A);
    REQUIRE(E(0, 0) == Catch::Approx(std::exp(1.0)).epsilon(1e-10));
    REQUIRE(E(1, 1) == Catch::Approx(std::exp(2.0)).epsilon(1e-10));
    REQUIRE(E(0, 1) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(E(1, 0) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("expm: comparison with Taylor series (small-norm A)", "[expm]") {
    // For small ||A||, expm(A) ≈ Taylor series to order 5.
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);
    Matrix A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = dist(rng);

    auto E = linalgebra::expm(A);
    auto E_taylor = expm_taylor5(A);
    REQUIRE(frobenius_diff(E, E_taylor) < 1e-10);
}

TEST_CASE("expm: large norm requires scaling (10*I)", "[expm]") {
    // expm(10*I) = exp(10)*I.
    Matrix A = Matrix::zeros(3, 3);
    A(0, 0) = 10.0; A(1, 1) = 10.0; A(2, 2) = 10.0;
    auto E = linalgebra::expm(A);

    const double e10 = std::exp(10.0);  // ≈ 22026.47
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE(E(i, i) == Catch::Approx(e10).epsilon(1e-8));
}

TEST_CASE("expm: non-square throws", "[expm]") {
    Matrix A(2, 3);
    REQUIRE_THROWS_AS(linalgebra::expm(A), DimensionMismatchError);
}

TEST_CASE("expm: symmetric A gives symmetric expm(A)", "[expm]") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix B(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            B(i, j) = dist(rng);
    // Make symmetric: A = B + B^T.
    Matrix A(4, 4, 0.0);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = B(i, j) + B(j, i);

    auto E = linalgebra::expm(A);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            REQUIRE(E(i, j) == Catch::Approx(E(j, i)).margin(1e-9));
}
