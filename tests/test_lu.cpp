#include "linalg_error.hpp"
#include "lu.hpp"
#include "matrix.hpp"
#include "norms.hpp"
#include "vector.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <random>

using linalg::DimensionMismatchError;
using linalg::Matrix;
using linalg::SingularMatrixError;
using linalg::Vector;
using linalg::LUResult;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// ||PA - LU||_F  (Frobenius, computed element-wise via ||vec||_2)
double reconstruction_error(const Matrix& A, const LUResult& lu) {
    const std::size_t n = A.rows();
    // Build PA by permuting rows of A.
    Matrix PA(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            PA(i, j) = A(lu.perm[i], j);
        }
    }
    // Compute LU product.
    const Matrix LU = lu.L * lu.U;
    // Compute Frobenius norm of (PA - LU).
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double d = PA(i, j) - LU(i, j);
            err += d * d;
        }
    }
    return std::sqrt(err);
}

// ||Ax - b||_2
double solve_residual(const Matrix& A, const Vector& x, const Vector& b) {
    return linalg::norm2(A * x - b);
}

// Generate a reproducible random nonsingular n x n matrix.
Matrix random_matrix(std::size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    Matrix M(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            M(i, j) = dist(rng);
        }
    }
    return M;
}

}  // namespace

// ---------------------------------------------------------------------------
// Factorization correctness
// ---------------------------------------------------------------------------

TEST_CASE("LU factorization: 3x3 known system", "[lu]") {
    const Matrix A{
        {2.0, 1.0, -1.0},
        {-3.0, -1.0, 2.0},
        {-2.0, 1.0, 2.0}
    };
    const LUResult lu = linalg::lu_factor(A);

    REQUIRE(lu.L.rows() == 3);
    REQUIRE(lu.U.rows() == 3);
    REQUIRE(lu.perm.size() == 3);

    for (std::size_t i = 0; i < 3; ++i) {
        CHECK(lu.L(i, i) == Catch::Approx(1.0));
    }

    // ||PA - LU|| must be near zero.
    CHECK(reconstruction_error(A, lu) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("LU factorization: identity matrix", "[lu]") {
    const Matrix I = Matrix::identity(4);
    const LUResult lu = linalg::lu_factor(I);
    CHECK(reconstruction_error(I, lu) == Catch::Approx(0.0).margin(1e-14));
    // U should equal I (up to row ordering already handled by PA=LU).
    for (std::size_t i = 0; i < 4; ++i) {
        CHECK(lu.U(i, i) == Catch::Approx(1.0));
    }
}

TEST_CASE("LU factorization: matrix requiring row swaps", "[lu]") {
    // First column entry is zero. No-pivot LU would immediately fail.
    const Matrix A{
        {0.0, 1.0, 2.0},
        {3.0, 4.0, 5.0},
        {6.0, 7.0, 8.0}
    };
    // A is singular (rows are in AP), but check that partial pivoting still
    // proceeds and detects singularity correctly.
    // Row 3 - row 2 = row 2 - row 1, so rank < 3.
    CHECK_THROWS_AS(linalg::lu_factor(A), SingularMatrixError);
}

TEST_CASE("LU factorization: first-column zero, nonsingular", "[lu]") {
    // [[0, 1], [1, 0]] — requires a swap at step 0.
    const Matrix A{{0.0, 1.0}, {1.0, 0.0}};
    const LUResult lu = linalg::lu_factor(A);
    CHECK(reconstruction_error(A, lu) == Catch::Approx(0.0).margin(1e-14));
    // Solving Ax = b: A swaps components.
    const Vector b{3.0, 7.0};
    const Vector x = linalg::lu_solve(lu, b);
    CHECK(solve_residual(A, x, b) == Catch::Approx(0.0).margin(1e-12));
    CHECK(x[0] == Catch::Approx(7.0));
    CHECK(x[1] == Catch::Approx(3.0));
}

TEST_CASE("LU factorization: random nonsingular matrices", "[lu]") {
    for (std::size_t n : {5u, 10u, 20u}) {
        const Matrix A = random_matrix(n, 123u + static_cast<unsigned>(n));
        const LUResult lu = linalg::lu_factor(A);
        CHECK(reconstruction_error(A, lu) == Catch::Approx(0.0).margin(1e-10));
    }
}

// ---------------------------------------------------------------------------
// Solve correctness
// ---------------------------------------------------------------------------

TEST_CASE("LU solve: known 3x3 system", "[lu]") {
    // From Cramer / textbook: solution is x = (2, 3, -1).
    const Matrix A{
        {2.0, 1.0, -1.0},
        {-3.0, -1.0, 2.0},
        {-2.0, 1.0, 2.0}
    };
    const Vector expected{2.0, 3.0, -1.0};
    const Vector b = A * expected;

    const LUResult lu = linalg::lu_factor(A);
    const Vector x = linalg::lu_solve(lu, b);

    CHECK(x[0] == Catch::Approx(expected[0]).epsilon(1e-12));
    CHECK(x[1] == Catch::Approx(expected[1]).epsilon(1e-12));
    CHECK(x[2] == Catch::Approx(expected[2]).epsilon(1e-12));
    CHECK(solve_residual(A, x, b) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("LU solve: random nonsingular systems", "[lu]") {
    for (std::size_t n : {5u, 15u, 30u}) {
        const Matrix A = random_matrix(n, 7u * static_cast<unsigned>(n));
        const LUResult lu = linalg::lu_factor(A);

        // Random rhs.
        std::mt19937 rng(n);
        std::uniform_real_distribution<double> dist(-5.0, 5.0);
        Vector b(n);
        for (std::size_t i = 0; i < n; ++i) b[i] = dist(rng);

        const Vector x = linalg::lu_solve(lu, b);
        CHECK(solve_residual(A, x, b) == Catch::Approx(0.0).margin(1e-9));
    }
}

TEST_CASE("LU solve: diagonal system", "[lu]") {
    // D = diag(2, 3, 4), b = (2, 9, 8), solution = (1, 3, 2).
    const Matrix D{
        {2.0, 0.0, 0.0},
        {0.0, 3.0, 0.0},
        {0.0, 0.0, 4.0}
    };
    const Vector b{2.0, 9.0, 8.0};
    const LUResult lu = linalg::lu_factor(D);
    const Vector x = linalg::lu_solve(lu, b);

    CHECK(x[0] == Catch::Approx(1.0));
    CHECK(x[1] == Catch::Approx(3.0));
    CHECK(x[2] == Catch::Approx(2.0));
}

// ---------------------------------------------------------------------------
// Failure cases
// ---------------------------------------------------------------------------

TEST_CASE("LU factorization: non-square matrix throws", "[lu]") {
    const Matrix A(3, 4);
    CHECK_THROWS_AS(linalg::lu_factor(A), DimensionMismatchError);
}

TEST_CASE("LU factorization: exactly singular matrix throws", "[lu]") {
    // Zero row → singular.
    const Matrix A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {0.0, 0.0, 0.0}
    };
    CHECK_THROWS_AS(linalg::lu_factor(A), SingularMatrixError);
}

TEST_CASE("LU factorization: rank-deficient matrix throws", "[lu]") {
    // Row 2 is a linear combination of rows 0 and 1.
    const Matrix A{
        {1.0, 2.0},
        {2.0, 4.0}
    };
    CHECK_THROWS_AS(linalg::lu_factor(A), SingularMatrixError);
}

TEST_CASE("LU factorization: near-singular matrix throws at default tolerance", "[lu]") {
    // Pivot reduced to ~1e-16, should trip the singularity check.
    const Matrix A{
        {1.0, 1.0},
        {1.0, 1.0 + 1e-16}
    };
    CHECK_THROWS_AS(linalg::lu_factor(A), SingularMatrixError);
}

TEST_CASE("LU solve: mismatched rhs throws", "[lu]") {
    const Matrix A = Matrix::identity(3);
    const LUResult lu = linalg::lu_factor(A);
    const Vector b(5, 1.0);
    CHECK_THROWS_AS(linalg::lu_solve(lu, b), DimensionMismatchError);
}

// ---------------------------------------------------------------------------
// L and U structure
// ---------------------------------------------------------------------------

TEST_CASE("LU factorization: L is unit lower triangular", "[lu]") {
    const Matrix A = random_matrix(6, 999u);
    const LUResult lu = linalg::lu_factor(A);
    const std::size_t n = A.rows();

    for (std::size_t i = 0; i < n; ++i) {
        // Unit diagonal.
        CHECK(lu.L(i, i) == Catch::Approx(1.0));
        // Strict upper triangle is zero.
        for (std::size_t j = i + 1; j < n; ++j) {
            CHECK(lu.L(i, j) == Catch::Approx(0.0).margin(1e-15));
        }
    }
}

TEST_CASE("LU factorization: U is upper triangular", "[lu]") {
    const Matrix A = random_matrix(6, 777u);
    const LUResult lu = linalg::lu_factor(A);
    const std::size_t n = A.rows();

    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            CHECK(lu.U(i, j) == Catch::Approx(0.0).margin(1e-15));
        }
    }
}

// ---------------------------------------------------------------------------
// Permutation sign and determinant
// ---------------------------------------------------------------------------

TEST_CASE("LU factorization: sign of permutation is ±1", "[lu]") {
    const Matrix A = random_matrix(5, 321u);
    const LUResult lu = linalg::lu_factor(A);
    CHECK((lu.sign == 1 || lu.sign == -1));
}

TEST_CASE("LU factorization: determinant via sign * prod(diag(U))", "[lu]") {
    // det([[3,1],[2,4]]) = 12 - 2 = 10
    const Matrix A{{3.0, 1.0}, {2.0, 4.0}};
    const LUResult lu = linalg::lu_factor(A);
    double det = static_cast<double>(lu.sign);
    for (std::size_t i = 0; i < A.rows(); ++i) {
        det *= lu.U(i, i);
    }
    CHECK(det == Catch::Approx(10.0).epsilon(1e-12));
}
