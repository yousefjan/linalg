#include "linalg_error.hpp"
#include "matrix.hpp"
#include "norms.hpp"
#include "qr.hpp"
#include "vector.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <functional>
#include <random>

using linalg::DimensionMismatchError;
using linalg::Matrix;
using linalg::QRResult;
using linalg::SingularMatrixError;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// ||A - QR||_F
double reconstruction_error(const Matrix& A, const QRResult& qr) {
    const Matrix diff = A - qr.Q * qr.R;
    double err = 0.0;
    for (std::size_t i = 0; i < diff.rows(); ++i)
        for (std::size_t j = 0; j < diff.cols(); ++j)
            err += diff(i, j) * diff(i, j);
    return std::sqrt(err);
}

// ||Q^T Q - I||_F  (should be ~0 for orthonormal Q)
double orthogonality_error(const QRResult& qr) {
    const Matrix& Q = qr.Q;
    const std::size_t n = Q.cols();
    // Compute Q^T Q
    Matrix QtQ(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < Q.rows(); ++k) s += Q(k, i) * Q(k, j);
            QtQ(i, j) = s;
        }
    // ||QtQ - I||_F
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double d = QtQ(i, j) - (i == j ? 1.0 : 0.0);
            err += d * d;
        }
    return std::sqrt(err);
}

// R must be upper triangular (strict lower triangle near zero).
bool r_is_upper_triangular(const Matrix& R, double tol = 1e-12) {
    for (std::size_t i = 1; i < R.rows(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (std::abs(R(i, j)) > tol) return false;
    return true;
}

Matrix random_matrix(std::size_t m, std::size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    Matrix M(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            M(i, j) = dist(rng);
    return M;
}

// Run all checks for a given QR function and matrix.
using QRFn = std::function<QRResult(const Matrix&)>;

void check_qr(const Matrix& A, QRFn fn,
              double recon_tol, double ortho_tol,
              const std::string& /*label*/) {
    const QRResult qr = fn(A);
    CHECK(qr.Q.rows() == A.rows());
    CHECK(qr.Q.cols() == A.cols());
    CHECK(qr.R.rows() == A.cols());
    CHECK(qr.R.cols() == A.cols());
    CHECK(reconstruction_error(A, qr) == Catch::Approx(0.0).margin(recon_tol));
    CHECK(orthogonality_error(qr) == Catch::Approx(0.0).margin(ortho_tol));
    CHECK(r_is_upper_triangular(qr.R));
}

}  // namespace

// ---------------------------------------------------------------------------
// Macro to run the same test body for all three methods
// ---------------------------------------------------------------------------

#define FOR_ALL_METHODS(A, recon_tol, ortho_tol)                                  \
    SECTION("classical_gs") {                                                      \
        check_qr(A, [](const Matrix& M) { return linalg::qr_classical_gs(M); },  \
                 recon_tol, ortho_tol, "classical_gs");                            \
    }                                                                              \
    SECTION("modified_gs") {                                                       \
        check_qr(A, [](const Matrix& M) { return linalg::qr_modified_gs(M); },   \
                 recon_tol, ortho_tol, "modified_gs");                             \
    }                                                                              \
    SECTION("householder") {                                                       \
        check_qr(A, [](const Matrix& M) { return linalg::qr_householder(M); },   \
                 recon_tol, ortho_tol, "householder");                             \
    }

// ---------------------------------------------------------------------------
// Basic correctness: square matrices
// ---------------------------------------------------------------------------

TEST_CASE("QR: 3x3 known matrix", "[qr]") {
    const Matrix A{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 10.0}   // not exactly singular
    };
    FOR_ALL_METHODS(A, 1e-12, 1e-12)
}

TEST_CASE("QR: identity matrix", "[qr]") {
    const Matrix I = Matrix::identity(4);
    FOR_ALL_METHODS(I, 1e-14, 1e-14)
}

TEST_CASE("QR: diagonal matrix", "[qr]") {
    const Matrix D{
        {3.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 2.0}
    };
    FOR_ALL_METHODS(D, 1e-14, 1e-14)
}

// ---------------------------------------------------------------------------
// Rectangular (tall) matrices
// ---------------------------------------------------------------------------

TEST_CASE("QR: tall 5x3 random matrix", "[qr]") {
    const Matrix A = random_matrix(5, 3, 7u);
    FOR_ALL_METHODS(A, 1e-12, 1e-12)
}

TEST_CASE("QR: tall 10x4 random matrix", "[qr]") {
    const Matrix A = random_matrix(10, 4, 99u);
    FOR_ALL_METHODS(A, 1e-12, 1e-12)
}

// ---------------------------------------------------------------------------
// Random square matrices
// ---------------------------------------------------------------------------

TEST_CASE("QR: random 6x6", "[qr]") {
    const Matrix A = random_matrix(6, 6, 123u);
    FOR_ALL_METHODS(A, 1e-12, 1e-12)
}

TEST_CASE("QR: random 12x12", "[qr]") {
    const Matrix A = random_matrix(12, 12, 456u);
    FOR_ALL_METHODS(A, 1e-11, 1e-11)
}

// ---------------------------------------------------------------------------
// Nearly dependent columns — GS methods degrade; Householder stays clean
// ---------------------------------------------------------------------------

TEST_CASE("QR: nearly dependent columns", "[qr]") {
    // Column 1 = column 0 + epsilon * e_1.
    // Classical GS will lose most of Q's orthogonality here.
    // Modified GS is better. Householder is unaffected.
    constexpr double eps = 1e-7;
    const Matrix A{
        {1.0,      1.0 + eps, 0.0},
        {1.0,      1.0,       1.0},
        {0.0,      eps,       1.0},
        {0.0,      0.0,       1.0}
    };

    // All three should reconstruct A accurately.
    SECTION("classical_gs reconstruction") {
        const QRResult qr = linalg::qr_classical_gs(A);
        CHECK(reconstruction_error(A, qr) == Catch::Approx(0.0).margin(1e-10));
        // Orthogonality will be poor for classical GS on this input.
        // We only assert it's not catastrophically wrong (< 0.01).
        CHECK(orthogonality_error(qr) < 0.01);
    }
    SECTION("modified_gs reconstruction") {
        const QRResult qr = linalg::qr_modified_gs(A);
        CHECK(reconstruction_error(A, qr) == Catch::Approx(0.0).margin(1e-10));
        CHECK(orthogonality_error(qr) == Catch::Approx(0.0).margin(1e-8));
    }
    SECTION("householder reconstruction") {
        const QRResult qr = linalg::qr_householder(A);
        CHECK(reconstruction_error(A, qr) == Catch::Approx(0.0).margin(1e-13));
        CHECK(orthogonality_error(qr) == Catch::Approx(0.0).margin(1e-13));
    }
}

// ---------------------------------------------------------------------------
// Hilbert-like ill-conditioned matrix
// ---------------------------------------------------------------------------

TEST_CASE("QR: 4x4 Hilbert matrix", "[qr]") {
    // H[i][j] = 1 / (i + j + 1)
    const std::size_t n = 4;
    Matrix H(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            H(i, j) = 1.0 / static_cast<double>(i + j + 1);

    SECTION("classical_gs") {
        const QRResult qr = linalg::qr_classical_gs(H);
        CHECK(reconstruction_error(H, qr) == Catch::Approx(0.0).margin(1e-12));
        // Orthogonality is imperfect on Hilbert matrices with classical GS.
        CHECK(orthogonality_error(qr) < 1e-8);
    }
    SECTION("modified_gs") {
        const QRResult qr = linalg::qr_modified_gs(H);
        CHECK(reconstruction_error(H, qr) == Catch::Approx(0.0).margin(1e-12));
        CHECK(orthogonality_error(qr) == Catch::Approx(0.0).margin(1e-10));
    }
    SECTION("householder") {
        const QRResult qr = linalg::qr_householder(H);
        CHECK(reconstruction_error(H, qr) == Catch::Approx(0.0).margin(1e-13));
        CHECK(orthogonality_error(qr) == Catch::Approx(0.0).margin(1e-13));
    }
}

// ---------------------------------------------------------------------------
// Structure checks
// ---------------------------------------------------------------------------

TEST_CASE("QR: R is upper triangular", "[qr]") {
    const Matrix A = random_matrix(5, 5, 555u);
    CHECK(r_is_upper_triangular(linalg::qr_classical_gs(A).R));
    CHECK(r_is_upper_triangular(linalg::qr_modified_gs(A).R));
    CHECK(r_is_upper_triangular(linalg::qr_householder(A).R));
}

TEST_CASE("QR: Q columns are unit length", "[qr]") {
    const Matrix A = random_matrix(6, 4, 321u);
    for (QRFn fn : {QRFn{[](const Matrix& M) { return linalg::qr_classical_gs(M); }},
                    QRFn{[](const Matrix& M) { return linalg::qr_modified_gs(M); }},
                    QRFn{[](const Matrix& M) { return linalg::qr_householder(M); }}}) {
        const QRResult qr = fn(A);
        for (std::size_t j = 0; j < qr.Q.cols(); ++j) {
            double norm2 = 0.0;
            for (std::size_t i = 0; i < qr.Q.rows(); ++i)
                norm2 += qr.Q(i, j) * qr.Q(i, j);
            CHECK(std::sqrt(norm2) == Catch::Approx(1.0).margin(1e-13));
        }
    }
}

// ---------------------------------------------------------------------------
// Failure cases
// ---------------------------------------------------------------------------

TEST_CASE("QR: fat matrix throws DimensionMismatchError", "[qr]") {
    const Matrix A(3, 5);   // rows < cols
    CHECK_THROWS_AS(linalg::qr_classical_gs(A), DimensionMismatchError);
    CHECK_THROWS_AS(linalg::qr_modified_gs(A), DimensionMismatchError);
    CHECK_THROWS_AS(linalg::qr_householder(A), DimensionMismatchError);
}

TEST_CASE("QR: linearly dependent columns throw from GS methods", "[qr]") {
    const Matrix A{
        {1.0, 2.0, 2.0},   // col 2 = 2 * col 0
        {2.0, 4.0, 4.0},
        {3.0, 6.0, 6.0}
    };
    CHECK_THROWS_AS(linalg::qr_classical_gs(A), SingularMatrixError);
    CHECK_THROWS_AS(linalg::qr_modified_gs(A), SingularMatrixError);
    // Householder handles rank-deficient input gracefully (R gets a zero diagonal entry).
    CHECK_NOTHROW(linalg::qr_householder(A));
}
