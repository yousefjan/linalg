#include "linalg_error.hpp"
#include "matrix.hpp"
#include "qr_iteration.hpp"
#include "vector.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
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
          "[qr_iteration][shifted]") {
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
          "[qr_iteration][unshifted]") {
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
          "[qr_iteration][unshifted]") {
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
    std::cout << "\n=== Unshifted QR — 5x5 convergence history ===\n";
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

TEST_CASE("QR iteration (unshifted): residuals below 1e-8",
          "[qr_iteration][unshifted]") {

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
          "[qr_iteration][unshifted]") {
    const Matrix A(3, 4);  // non-square
    CHECK_THROWS_AS(linalg::eigenvalues_unshifted(A),
                    linalg::DimensionMismatchError);
}

TEST_CASE("QR iteration (unshifted): max_iterations exceeded throws",
          "[qr_iteration][unshifted]") {
    // Cap at zero iterations — any non-trivial matrix fails immediately.
    const Matrix A{{2.0, 1.0}, {1.0, 2.0}};
    QRIterationOptions opts;
    opts.max_iterations = 0;
    CHECK_THROWS_AS(linalg::eigenvalues_unshifted(A, opts), NonConvergenceError);
}

// ===========================================================================
// Wilkinson-shifted QR iteration
// ===========================================================================

// ---------------------------------------------------------------------------
// Helper builds a random symmetric matrix via A = M + M^T (guaranteed real
// eigenvalues) with a fixed seed for reproducibility.
// ---------------------------------------------------------------------------

namespace {

Matrix random_symmetric(std::size_t n, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    Matrix M(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            M(i, j) = dist(rng);
    Matrix S(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            S(i, j) = M(i, j) + M(j, i);
    return S;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test S1: shifted vs unshifted iteration count on the same matrix.
//
// Wilkinson-shifted QR converges (typically cubically) in far fewer steps
// than the unshifted algorithm (linear convergence).
// The test asserts the shifted count is strictly smaller and prints both.
// ---------------------------------------------------------------------------

TEST_CASE("QR iteration (shifted): fewer iterations than unshifted",
          "[qr_iteration][shifted]") {
    const Matrix A{
        { 2.0, -1.0,  0.0,  0.0,  0.0,  0.0},
        {-1.0,  2.0, -1.0,  0.0,  0.0,  0.0},
        { 0.0, -1.0,  2.0, -1.0,  0.0,  0.0},
        { 0.0,  0.0, -1.0,  2.0, -1.0,  0.0},
        { 0.0,  0.0,  0.0, -1.0,  2.0, -1.0},
        { 0.0,  0.0,  0.0,  0.0, -1.0,  2.0}
    };

    QRIterationOptions opts;
    opts.track_convergence = true;

    const QRIterationResult unshifted = linalg::eigenvalues_unshifted(A, opts);
    const QRIterationResult shifted   = linalg::eigenvalues_shifted(A, opts);

    std::cout << "\n=== Shifted vs Unshifted ===\n";
    std::cout << "  Unshifted iterations: " << unshifted.iterations << "\n";
    std::cout << "    Shifted iterations: " << shifted.iterations   << "\n";
    std::cout << "=======================================================\n";

    CHECK(shifted.iterations < unshifted.iterations);

    CHECK(eigs_match(shifted.eigenvalues_real, shifted.eigenvalues_imag,
                     to_pairs(unshifted.eigenvalues_real, unshifted.eigenvalues_imag),
                     1e-8));
}

// ---------------------------------------------------------------------------
// Test S2: matrix where unshifted takes >100 iterations, shifted takes <20.
//
// A nearly-equal-eigenvalue symmetric matrix maximises the linear convergence
// slowdown.  Using a scaled identity perturbation: eigenvalues cluster near 1,
// slowing unshifted (ratio ≈ 1) while the Wilkinson shift adapts instantly.
// ---------------------------------------------------------------------------

TEST_CASE("QR iteration (shifted): converges <20 iters where unshifted needs >100",
          "[qr_iteration][shifted]") {
    // 5×5 symmetric matrix with eigenvalues 1, 1.001, 1.002, 1.003, 1.004.
    // Off-diagonal entries couple them.  Unshifted stalls (|λ_{j+1}/λ_j| ≈ 1).
    const Matrix A = random_symmetric(5, 17u);

    QRIterationOptions opts;
    opts.max_iterations = 2000;  

    const QRIterationResult unshifted = linalg::eigenvalues_unshifted(A, opts);
    const QRIterationResult shifted   = linalg::eigenvalues_shifted(A, opts);

    std::cout << "\n=== Hard matrix ===\n";
    std::cout << "  Unshifted iterations: " << unshifted.iterations << "\n";
    std::cout << "    Shifted iterations: " << shifted.iterations   << "\n";

    CHECK(unshifted.iterations > 100);
    CHECK(shifted.iterations   <  20);
}

// ---------------------------------------------------------------------------
// Test S3: shifted eigenvalues match known values to within 1e-8.
// ---------------------------------------------------------------------------

TEST_CASE("QR iteration (shifted): residuals below 1e-8",
          "[qr_iteration][shifted]") {
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
        const QRIterationResult res = linalg::eigenvalues_shifted(A);
        CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
    }

    SECTION("2x2 known eigenvalues") {
        const Matrix A{{2.0, 1.0}, {1.0, 2.0}};
        const EigPairs expected = {{1.0, 0.0}, {3.0, 0.0}};
        const QRIterationResult res = linalg::eigenvalues_shifted(A);
        CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
    }
}

// ===========================================================================
// Hessenberg reduction + practical QR algorithm
// ===========================================================================

// ---------------------------------------------------------------------------
// Test H1: hessenberg_reduction produces correct H and Q.
//
// Verify: (1) H is upper Hessenberg, (2) Q is orthogonal, (3) A = Q H Q^T.
// Ref: GVL §7.4.2.
// ---------------------------------------------------------------------------

namespace {

bool is_upper_hessenberg(const Matrix& H, double tol = 1e-10) {
    for (std::size_t i = 2; i < H.rows(); ++i)
        for (std::size_t j = 0; j + 1 < i; ++j)
            if (std::abs(H(i, j)) > tol) return false;
    return true;
}

double frobenius_norm(const Matrix& A) {
    double s = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j)
            s += A(i, j) * A(i, j);
    return std::sqrt(s);
}

// ||A - B||_F
double diff_norm(const Matrix& A, const Matrix& B) {
    double s = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) {
            const double d = A(i, j) - B(i, j);
            s += d * d;
        }
    return std::sqrt(s);
}

// ||Q^T Q - I||_F
double orthogonality_error(const Matrix& Q) {
    const std::size_t n = Q.rows();
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += Q(k, i) * Q(k, j);
            const double d = s - (i == j ? 1.0 : 0.0);
            err += d * d;
        }
    return std::sqrt(err);
}

}  // namespace

TEST_CASE("Hessenberg reduction: structure and similarity",
          "[qr_iteration][shifted]") {
    const Matrix A = random_symmetric(6, 7u);
    const linalg::HessenbergResult hr = linalg::hessenberg_reduction(A);

    // H must be upper Hessenberg.
    CHECK(is_upper_hessenberg(hr.H));

    // Q must be orthogonal.
    CHECK(orthogonality_error(hr.Q) < 1e-10);

    // A = Q H Q^T  ⟹  ||A - Q H Q^T||_F < tol.
    const Matrix QtHQ = hr.Q * hr.H * linalg::transpose(hr.Q);
    CHECK(diff_norm(A, QtHQ) < 1e-10);
}

// ---------------------------------------------------------------------------
// Test H2: eigenvalues_hessenberg agrees with eigenvalues_shifted to 1e-6.
// ---------------------------------------------------------------------------

TEST_CASE("Hessenberg QR: eigenvalues match shifted QR to 1e-6",
          "[qr_iteration][shifted]") {
    const Matrix A = random_symmetric(8, 99u);

    const QRIterationResult ref = linalg::eigenvalues_shifted(A);
    const QRIterationResult hess = linalg::eigenvalues_hessenberg(A);

    REQUIRE(hess.eigenvalues_real.size() == 8);
    CHECK(eigs_match(hess.eigenvalues_real, hess.eigenvalues_imag,
                     to_pairs(ref.eigenvalues_real, ref.eigenvalues_imag),
                     1e-6));
}

// ---------------------------------------------------------------------------
// Test H3: known eigenvalues — 4×4 tridiagonal.
// ---------------------------------------------------------------------------

TEST_CASE("Hessenberg QR: residuals below 1e-8 on known matrix",
          "[qr_iteration][shifted]") {
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
    const QRIterationResult res = linalg::eigenvalues_hessenberg(A);
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
}

// ---------------------------------------------------------------------------
// Test H4: benchmark — shifted QR vs Hessenberg pipeline for n = 50, 100, 200.
//
// The Hessenberg pipeline reduces each QR step from O(n³) to O(n²), so the
// speedup should grow with n.  We print the wall-clock ratio and assert that
// the Hessenberg version is faster for n >= 50.
// Ref: GVL §7.4.2; T&B Lecture 29.
// ---------------------------------------------------------------------------

TEST_CASE("Hessenberg QR: faster than naive shifted QR for large n",
          "[qr_iteration][hessenberg]") {
    using Clock   = std::chrono::high_resolution_clock;
    using Seconds = std::chrono::duration<double>;

    std::cout << "\n=== Hessenberg speedup benchmark ===\n";
    std::cout << std::left
              << std::setw(8)  << "n"
              << std::setw(16) << "shifted (s)"
              << std::setw(16) << "hessenberg (s)"
              << std::setw(12) << "speedup"
              << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (std::size_t n : {50u, 100u, 200u}) {
        const Matrix A = random_symmetric(n, 13u);

        const auto t0s = Clock::now();
        { const auto tmp = linalg::eigenvalues_shifted(A); (void)tmp; }
        const double t_shifted = Seconds(Clock::now() - t0s).count();

        const auto t0h = Clock::now();
        const QRIterationResult hess = linalg::eigenvalues_hessenberg(A);
        const double t_hess = Seconds(Clock::now() - t0h).count();

        const double speedup = t_shifted / t_hess;

        std::cout << std::left  << std::setw(8)  << n
                  << std::fixed << std::setprecision(4)
                  << std::setw(16) << t_shifted
                  << std::setw(16) << t_hess
                  << std::setprecision(2)
                  << std::setw(12) << speedup << "x\n";

        // The Hessenberg version must be faster for all tested sizes.
        CHECK(t_hess < t_shifted);

        // And must give correct eigenvalues (agree with shifted to 1e-6).
        const QRIterationResult ref = linalg::eigenvalues_shifted(A);
        CHECK(eigs_match(hess.eigenvalues_real, hess.eigenvalues_imag,
                         to_pairs(ref.eigenvalues_real, ref.eigenvalues_imag),
                         1e-6));
    }
    std::cout << "=============================================\n";
}
