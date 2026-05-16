import linalgebra;

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <random>

using linalgebra::DimensionMismatchError;
using linalgebra::LinAlgError;
using linalgebra::Matrix;
using linalgebra::Vector;

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

Matrix make_dd(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Matrix A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            A(i, j) = dist(rng);
            if (i != j) row_sum += std::abs(A(i, j));
        }
        A(i, i) = row_sum + 1.0;  // strictly diagonally dominant
    }
    return A;
}

Matrix make_spd(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix B(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            B(i, j) = dist(rng);
    Matrix A = linalgebra::transpose(B) * B;
    for (std::size_t i = 0; i < n; ++i) A(i, i) += static_cast<double>(n);
    return A;
}

}  // namespace

// arnoldi

TEST_CASE("arnoldi: orthonormality and AQ=QH relation", "[iterative][arnoldi]") {
    Matrix A{{2.0, 1.0, 0.0},
             {1.0, 3.0, 1.0},
             {0.0, 1.0, 4.0}};
    Vector b{1.0, 0.0, 0.0};
    const int k = 2;
    auto res = linalgebra::arnoldi(A, b, k);

    const std::size_t n = 3;
    const auto steps = static_cast<std::size_t>(res.steps_taken);

    // Q columns must be orthonormal.
    for (std::size_t i = 0; i <= steps; ++i) {
        for (std::size_t j = 0; j <= steps; ++j) {
            double dot = 0.0;
            for (std::size_t r = 0; r < n; ++r) dot += res.Q(r, i) * res.Q(r, j);
            const double expected = (i == j) ? 1.0 : 0.0;
            REQUIRE(dot == Catch::Approx(expected).margin(1e-10));
        }
    }

    for (std::size_t j = 0; j < steps; ++j) {
        Vector qj(n);
        for (std::size_t i = 0; i < n; ++i) qj[i] = res.Q(i, j);
        const Vector Aqj = A * qj;

        for (std::size_t i = 0; i <= steps; ++i) {
            double qh = 0.0;
            for (std::size_t r = 0; r < n; ++r) qh += res.Q(r, i) * res.H(i, j);
        }


        // Direct check: ||A*qj - Q*H[:,j]||
        Vector Hcol(steps + 1);
        for (std::size_t i = 0; i <= steps; ++i) Hcol[i] = res.H(i, j);
        double resid = 0.0;
        for (std::size_t row = 0; row < n; ++row) {
            double qh_row = 0.0;
            for (std::size_t i = 0; i <= steps; ++i) qh_row += res.Q(row, i) * Hcol[i];
            const double d = Aqj[row] - qh_row;
            resid += d * d;
        }
        REQUIRE(std::sqrt(resid) < 1e-10);
    }
}

TEST_CASE("arnoldi: breakdown on scaled identity", "[iterative][arnoldi]") {
    // A = 2*I → Krylov space is one-dimensional.
    Matrix A(3, 3, 0.0);
    A(0, 0) = 2.0; A(1, 1) = 2.0; A(2, 2) = 2.0;
    Vector b{1.0, 0.0, 0.0};
    auto res = linalgebra::arnoldi(A, b, 3);
    REQUIRE(res.breakdown == true);
    REQUIRE(res.steps_taken <= 3);
}

TEST_CASE("arnoldi: dimension mismatch throws", "[iterative][arnoldi]") {
    Matrix A(3, 3, 0.0);
    Vector b(4, 0.0);
    REQUIRE_THROWS_AS(linalgebra::arnoldi(A, b, 2), DimensionMismatchError);
}

// solve_cg

TEST_CASE("solve_cg: 2x2 SPD", "[iterative][cg]") {
    Matrix A{{4.0, 1.0}, {1.0, 3.0}};
    Vector b{1.0, 2.0};
    auto res = linalgebra::solve_cg(A, b);
    REQUIRE(linalgebra::norm2(A * res.x - b) < 1e-10);
}

TEST_CASE("solve_cg: identity system converges in 1 iteration", "[iterative][cg]") {
    auto I = Matrix::identity(5);
    Vector b{1.0, 2.0, 3.0, 4.0, 5.0};
    auto res = linalgebra::solve_cg(I, b);
    for (std::size_t i = 0; i < 5; ++i)
        REQUIRE(res.x[i] == Catch::Approx(b[i]).margin(1e-10));
}

TEST_CASE("solve_cg: 5x5 random SPD", "[iterative][cg]") {
    std::mt19937 rng(99);
    Matrix A = make_spd(5, rng);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    Vector b(5);
    for (std::size_t i = 0; i < 5; ++i) b[i] = dist(rng);

    auto res = linalgebra::solve_cg(A, b);
    REQUIRE(linalgebra::norm2(A * res.x - b) < 1e-8);

    // Cross-check vs LU.
    auto lu  = linalgebra::lu_factor(A);
    auto x_lu = linalgebra::lu_solve(lu, b);
    REQUIRE(linalgebra::norm2(res.x - x_lu) < 1e-8);
}

TEST_CASE("solve_cg: non-symmetric throws", "[iterative][cg]") {
    Matrix A{{1.0, 2.0}, {0.0, 1.0}};
    Vector b{1.0, 1.0};
    REQUIRE_THROWS_AS(linalgebra::solve_cg(A, b), LinAlgError);
}

// solve_gmres

TEST_CASE("solve_gmres: 2x2 non-symmetric", "[iterative][gmres]") {
    Matrix A{{2.0, 1.0}, {1.0, 3.0}};
    Vector b{5.0, 7.0};
    auto res = linalgebra::solve_gmres(A, b);
    REQUIRE(linalgebra::norm2(A * res.x - b) < 1e-9);
}

TEST_CASE("solve_gmres: 5x5 diagonally dominant", "[iterative][gmres]") {
    std::mt19937 rng(11);
    Matrix A = make_dd(5, rng);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    Vector b(5);
    for (std::size_t i = 0; i < 5; ++i) b[i] = dist(rng);

    auto res = linalgebra::solve_gmres(A, b);
    REQUIRE(linalgebra::norm2(A * res.x - b) < 1e-8);

    auto lu = linalgebra::lu_factor(A);
    auto x_lu = linalgebra::lu_solve(lu, b);
    REQUIRE(linalgebra::norm2(res.x - x_lu) < 1e-8);
}

TEST_CASE("solve_gmres: zero rhs gives zero solution", "[iterative][gmres]") {
    Matrix A{{2.0, 1.0}, {1.0, 3.0}};
    Vector b(2, 0.0);
    auto res = linalgebra::solve_gmres(A, b);
    REQUIRE(linalgebra::norm2(res.x) < 1e-12);
}


// solve_bicgstab

TEST_CASE("solve_bicgstab: 2x2 non-symmetric", "[iterative][bicgstab]") {
    Matrix A{{2.0, 1.0}, {1.0, 3.0}};
    Vector b{5.0, 7.0};
    auto res = linalgebra::solve_bicgstab(A, b);
    REQUIRE(linalgebra::norm2(A * res.x - b) < 1e-9);
}

TEST_CASE("solve_bicgstab: 5x5 diagonally dominant", "[iterative][bicgstab]") {
    std::mt19937 rng(22);
    Matrix A = make_dd(5, rng);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    Vector b(5);
    for (std::size_t i = 0; i < 5; ++i) b[i] = dist(rng);

    auto res = linalgebra::solve_bicgstab(A, b);
    REQUIRE(linalgebra::norm2(A * res.x - b) < 1e-8);

    auto lu = linalgebra::lu_factor(A);
    auto x_lu = linalgebra::lu_solve(lu, b);
    REQUIRE(linalgebra::norm2(res.x - x_lu) < 1e-8);
}
