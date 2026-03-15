// Tests for qr_iteration.hpp / qr_iteration.cpp
//
// Stage 1: Unshifted QR iteration.
//
// All Stage 1 tests use symmetric matrices (only real eigenvalues) because
// the unshifted algorithm converges to upper-triangular form — not merely
// quasi-upper-triangular — only when all eigenvalues are real.  A matrix
// with a complex-conjugate pair would stall: its 2×2 Schur block keeps a
// non-negligible subdiagonal entry indefinitely, so ||lower(A_k)||_F never
// falls below the tolerance.  Proper handling of complex pairs requires the
// double-shift strategy introduced in Stage 2.
//
// Refs: T&B Lecture 28; GVL §7.3–7.4.

#include "linalg_error.hpp"
#include "matrix.hpp"
#include "qr_iteration.hpp"
#include "vector.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

using linalg::Matrix;
using linalg::NonConvergenceError;
using linalg::QRIterationOptions;
using linalg::QRIterationResult;
using linalg::Vector;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

namespace {

// Sort (real, imag) eigenvalue pairs by real part (ascending), then by imag.
// Returns a std::vector<std::pair<double,double>> — a plain container of
// pairs, not a math vector.
using EigPairs = std::vector<std::pair<double, double>>;

EigPairs to_pairs(const Vector& real_v, const Vector& imag_v) {
    EigPairs out;
    out.reserve(real_v.size());
    for (std::size_t i = 0; i < real_v.size(); ++i)
        out.emplace_back(real_v[i], imag_v[i]);
    std::sort(out.begin(), out.end(),
              [](const std::pair<double, double>& a,
                 const std::pair<double, double>& b) {
                  return a.first != b.first ? a.first < b.first
                                            : a.second < b.second;
              });
    return out;
}

// Return true when every computed eigenvalue is within `tol` of the
// corresponding expected eigenvalue (after sorting both sets).
// `expected` is a plain std::vector of (real, imag) pairs used as test data.
bool eigs_match(const Vector& computed_real, const Vector& computed_imag,
                const EigPairs& expected, double tol) {
    if (computed_real.size() != expected.size()) return false;
    const EigPairs computed = to_pairs(computed_real, computed_imag);
    EigPairs exp_sorted      = expected;
    std::sort(exp_sorted.begin(), exp_sorted.end(),
              [](const std::pair<double, double>& a,
                 const std::pair<double, double>& b) {
                  return a.first != b.first ? a.first < b.first
                                            : a.second < b.second;
              });
    for (std::size_t i = 0; i < computed.size(); ++i) {
        const double dr = computed[i].first  - exp_sorted[i].first;
        const double di = computed[i].second - exp_sorted[i].second;
        if (std::sqrt(dr * dr + di * di) > tol) return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1: 2×2 symmetric matrix with known eigenvalues
// ---------------------------------------------------------------------------
//
// A = | 2  1 |   is symmetric positive definite.
//     | 1  2 |
//
// Characteristic polynomial: (2-λ)^2 - 1 = 0  →  λ = 1, 3.
// Eigenvectors: [1,-1]/√2 (λ=1) and [1,1]/√2 (λ=3).
//
// The unshifted iteration converges at rate |λ_1/λ_2| = 1/3 per step,
// so only a handful of iterations are needed.
// Ref: T&B Theorem 28.2.

TEST_CASE("QR iteration (unshifted): 2x2 symmetric known eigenvalues",
          "[qr_iteration][stage1]") {
    const Matrix A{
        {2.0, 1.0},
        {1.0, 2.0}
    };

    const QRIterationResult res = linalg::eigenvalues_unshifted(A);

    REQUIRE(res.eigenvalues_real.size() == 2);
    REQUIRE(res.eigenvalues_imag.size() == 2);
    REQUIRE(res.iterations > 0);

    // All eigenvalues of a symmetric matrix must be real.
    CHECK(std::abs(res.eigenvalues_imag[0]) < 1e-8);
    CHECK(std::abs(res.eigenvalues_imag[1]) < 1e-8);

    const EigPairs expected = {{1.0, 0.0}, {3.0, 0.0}};
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
}

// ---------------------------------------------------------------------------
// Test 2: 4×4 symmetric tridiagonal matrix — reference eigenvalues
// ---------------------------------------------------------------------------
//
// The n×n symmetric tridiagonal matrix with 2 on the diagonal and -1 on the
// first super- and sub-diagonals has known eigenvalues (discrete Laplacian):
//
//   λ_k = 2 - 2 cos(k π / (n+1)),   k = 1, …, n
//
// Ref: Golub & Van Loan §4.4.2 (discrete sine transform).
//
// For n = 4:
//   λ_1 = 2 - 2 cos(π/5)   ≈ 0.3820
//   λ_2 = 2 - 2 cos(2π/5)  ≈ 1.3820
//   λ_3 = 2 - 2 cos(3π/5)  ≈ 2.6180
//   λ_4 = 2 - 2 cos(4π/5)  ≈ 3.6180

TEST_CASE("QR iteration (unshifted): 4x4 symmetric tridiagonal",
          "[qr_iteration][stage1]") {
    const Matrix A{
        { 2.0, -1.0,  0.0,  0.0},
        {-1.0,  2.0, -1.0,  0.0},
        { 0.0, -1.0,  2.0, -1.0},
        { 0.0,  0.0, -1.0,  2.0}
    };

    const QRIterationResult res = linalg::eigenvalues_unshifted(A);

    REQUIRE(res.eigenvalues_real.size() == 4);
    REQUIRE(res.eigenvalues_imag.size() == 4);

    // All eigenvalues of a symmetric matrix must be real.
    for (std::size_t k = 0; k < 4; ++k)
        CHECK(std::abs(res.eigenvalues_imag[k]) < 1e-8);

    // Compare against the closed-form reference.
    constexpr double pi = 3.14159265358979323846;
    const EigPairs expected = {
        {2.0 - 2.0 * std::cos(      pi / 5.0), 0.0},
        {2.0 - 2.0 * std::cos(2.0 * pi / 5.0), 0.0},
        {2.0 - 2.0 * std::cos(3.0 * pi / 5.0), 0.0},
        {2.0 - 2.0 * std::cos(4.0 * pi / 5.0), 0.0}
    };
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
}

// ---------------------------------------------------------------------------
// Test 3: 5×5 symmetric tridiagonal — convergence history
// ---------------------------------------------------------------------------
//
// Uses a 5×5 symmetric tridiagonal (discrete Laplacian) to guarantee all
// real eigenvalues and predictable linear convergence.  The Frobenius norm
// of the strict lower triangle is printed at every step so the convergence
// rate can be observed directly.
//
// Expected behaviour: ||lower(A_k)||_F decreases geometrically each step
// (linear convergence), with ratio ≈ max_j |λ_{j+1}/λ_j|.
// Ref: T&B Theorem 28.2.
//
// 5×5 tridiagonal eigenvalues: λ_k = 2 - 2cos(kπ/6), k = 1..5.

TEST_CASE("QR iteration (unshifted): 5x5 convergence history",
          "[qr_iteration][stage1]") {
    const Matrix A{
        { 2.0, -1.0,  0.0,  0.0,  0.0},
        {-1.0,  2.0, -1.0,  0.0,  0.0},
        { 0.0, -1.0,  2.0, -1.0,  0.0},
        { 0.0,  0.0, -1.0,  2.0, -1.0},
        { 0.0,  0.0,  0.0, -1.0,  2.0}
    };

    QRIterationOptions opts;
    opts.track_convergence = true;

    const QRIterationResult res = linalg::eigenvalues_unshifted(A, opts);

    REQUIRE_FALSE(res.convergence_history.empty());
    REQUIRE(res.eigenvalues_real.size() == 5);

    // Print convergence history so the linear rate is visible.
    std::cout << "\n=== Stage 1: Unshifted QR — 5x5 convergence history ===\n";
    std::cout << "  Converged in " << res.iterations << " iteration(s)\n";
    for (std::size_t k = 0; k < res.convergence_history.size(); ++k) {
        std::cout << "  iter " << (k + 1)
                  << ": ||lower(A_k)||_F = "
                  << res.convergence_history[k] << "\n";
    }
    std::cout << "=======================================================\n";

    // The final recorded norm must be below the default tolerance.
    CHECK(res.convergence_history.back() < opts.tolerance);
}

// ---------------------------------------------------------------------------
// Test 4: Eigenvalue residuals below 1e-8
// ---------------------------------------------------------------------------
//
// For several symmetric matrices with analytically known eigenvalues, verify
// that every computed eigenvalue is within 1e-8 of its expected value.
//
// Residual means the absolute error |λ_computed - λ_exact| (eigenvalue
// accuracy), not a matrix residual ||A x - λ x||, which would require
// eigenvectors unavailable in Stage 1.

TEST_CASE("QR iteration (unshifted): residuals below 1e-8",
          "[qr_iteration][stage1]") {

    SECTION("2x2: eigenvalues 1 and 3") {
        const Matrix A{{2.0, 1.0}, {1.0, 2.0}};
        const QRIterationResult res = linalg::eigenvalues_unshifted(A);
        const EigPairs expected = {{1.0, 0.0}, {3.0, 0.0}};
        CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
    }

    SECTION("3x3 diagonal: eigenvalues 1, 4, 9") {
        // Diagonal matrix — already in Schur form; converges in one step.
        const Matrix D{
            {1.0, 0.0, 0.0},
            {0.0, 4.0, 0.0},
            {0.0, 0.0, 9.0}
        };
        const QRIterationResult res = linalg::eigenvalues_unshifted(D);
        const EigPairs expected = {{1.0, 0.0}, {4.0, 0.0}, {9.0, 0.0}};
        CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
    }

    SECTION("4x4 tridiagonal: closed-form eigenvalues") {
        const Matrix A{
            { 2.0, -1.0,  0.0,  0.0},
            {-1.0,  2.0, -1.0,  0.0},
            { 0.0, -1.0,  2.0, -1.0},
            { 0.0,  0.0, -1.0,  2.0}
        };
        constexpr double pi = 3.14159265358979323846;
        const EigPairs expected = {
            {2.0 - 2.0 * std::cos(      pi / 5.0), 0.0},
            {2.0 - 2.0 * std::cos(2.0 * pi / 5.0), 0.0},
            {2.0 - 2.0 * std::cos(3.0 * pi / 5.0), 0.0},
            {2.0 - 2.0 * std::cos(4.0 * pi / 5.0), 0.0}
        };
        const QRIterationResult res = linalg::eigenvalues_unshifted(A);
        CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
    }

    SECTION("5x5 identity: all eigenvalues == 1") {
        const Matrix I = Matrix::identity(5);
        const QRIterationResult res = linalg::eigenvalues_unshifted(I);
        REQUIRE(res.eigenvalues_real.size() == 5);
        for (std::size_t k = 0; k < 5; ++k) {
            CHECK(std::abs(res.eigenvalues_real[k] - 1.0) < 1e-8);
            CHECK(std::abs(res.eigenvalues_imag[k])       < 1e-8);
        }
    }
}

// ---------------------------------------------------------------------------
// Failure cases
// ---------------------------------------------------------------------------

TEST_CASE("QR iteration (unshifted): non-square matrix throws",
          "[qr_iteration][stage1]") {
    const Matrix A(3, 4);  // non-square
    CHECK_THROWS_AS(linalg::eigenvalues_unshifted(A),
                    linalg::DimensionMismatchError);
}

TEST_CASE("QR iteration (unshifted): max_iterations exceeded throws",
          "[qr_iteration][stage1]") {
    // Cap at zero iterations — any non-trivial matrix fails immediately.
    const Matrix A{{2.0, 1.0}, {1.0, 2.0}};
    QRIterationOptions opts;
    opts.max_iterations = 0;
    CHECK_THROWS_AS(linalg::eigenvalues_unshifted(A, opts), NonConvergenceError);
}
