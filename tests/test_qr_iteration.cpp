import linalgebra;

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

using linalgebra::Matrix;
using linalgebra::NonConvergenceError;
using linalgebra::QRIterationOptions;
using linalgebra::QRIterationResult;
using linalgebra::Vector;

namespace {

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

TEST_CASE("QR iteration (unshifted): 2x2 symmetric known eigenvalues",
          "[qr_iteration][shifted]") {
    const Matrix A{
        {2.0, 1.0},
        {1.0, 2.0}
    };

    const QRIterationResult res = linalgebra::eigenvalues_unshifted(A);

    REQUIRE(res.eigenvalues_real.size() == 2);
    REQUIRE(res.eigenvalues_imag.size() == 2);
    REQUIRE(res.iterations > 0);

    CHECK(std::abs(res.eigenvalues_imag[0]) < 1e-8);
    CHECK(std::abs(res.eigenvalues_imag[1]) < 1e-8);

    const EigPairs expected = {{1.0, 0.0}, {3.0, 0.0}};
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
}

TEST_CASE("QR iteration (unshifted): 4x4 symmetric tridiagonal",
          "[qr_iteration][unshifted]") {
    const Matrix A{
        { 2.0, -1.0,  0.0,  0.0},
        {-1.0,  2.0, -1.0,  0.0},
        { 0.0, -1.0,  2.0, -1.0},
        { 0.0,  0.0, -1.0,  2.0}
    };

    const QRIterationResult res = linalgebra::eigenvalues_unshifted(A);

    REQUIRE(res.eigenvalues_real.size() == 4);
    REQUIRE(res.eigenvalues_imag.size() == 4);

    for (std::size_t k = 0; k < 4; ++k)
        CHECK(std::abs(res.eigenvalues_imag[k]) < 1e-8);

    constexpr double pi = 3.14159265358979323846;
    const EigPairs expected = {
        {2.0 - 2.0 * std::cos(      pi / 5.0), 0.0},
        {2.0 - 2.0 * std::cos(2.0 * pi / 5.0), 0.0},
        {2.0 - 2.0 * std::cos(3.0 * pi / 5.0), 0.0},
        {2.0 - 2.0 * std::cos(4.0 * pi / 5.0), 0.0}
    };
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
}

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

    const QRIterationResult res = linalgebra::eigenvalues_unshifted(A, opts);

    REQUIRE_FALSE(res.convergence_history.empty());
    REQUIRE(res.eigenvalues_real.size() == 5);

    std::cout << "\n=== Unshifted QR — 5x5 convergence history ===\n";
    std::cout << "  Converged in " << res.iterations << " iteration(s)\n";
    for (std::size_t k = 0; k < res.convergence_history.size(); ++k) {
        std::cout << "  iter " << (k + 1)
                  << ": ||lower(A_k)||_F = "
                  << res.convergence_history[k] << "\n";
    }
    std::cout << "=======================================================\n";

    CHECK(res.convergence_history.back() < opts.tolerance);
}

TEST_CASE("QR iteration (unshifted): residuals below 1e-8",
          "[qr_iteration][unshifted]") {

    SECTION("2x2: eigenvalues 1 and 3") {
        const Matrix A{{2.0, 1.0}, {1.0, 2.0}};
        const QRIterationResult res = linalgebra::eigenvalues_unshifted(A);
        const EigPairs expected = {{1.0, 0.0}, {3.0, 0.0}};
        CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
    }

    SECTION("3x3 diagonal: eigenvalues 1, 4, 9") {
        const Matrix D{
            {1.0, 0.0, 0.0},
            {0.0, 4.0, 0.0},
            {0.0, 0.0, 9.0}
        };
        const QRIterationResult res = linalgebra::eigenvalues_unshifted(D);
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
        const QRIterationResult res = linalgebra::eigenvalues_unshifted(A);
        CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
    }

    SECTION("5x5 identity: all eigenvalues == 1") {
        const Matrix I = Matrix::identity(5);
        const QRIterationResult res = linalgebra::eigenvalues_unshifted(I);
        REQUIRE(res.eigenvalues_real.size() == 5);
        for (std::size_t k = 0; k < 5; ++k) {
            CHECK(std::abs(res.eigenvalues_real[k] - 1.0) < 1e-8);
            CHECK(std::abs(res.eigenvalues_imag[k])       < 1e-8);
        }
    }
}

TEST_CASE("QR iteration (unshifted): non-square matrix throws",
          "[qr_iteration][unshifted]") {
    const Matrix A(3, 4);
    CHECK_THROWS_AS(linalgebra::eigenvalues_unshifted(A),
                    linalgebra::DimensionMismatchError);
}

TEST_CASE("QR iteration (unshifted): max_iterations exceeded throws",
          "[qr_iteration][unshifted]") {
    const Matrix A{{2.0, 1.0}, {1.0, 2.0}};
    QRIterationOptions opts;
    opts.max_iterations = 0;
    CHECK_THROWS_AS(linalgebra::eigenvalues_unshifted(A, opts), NonConvergenceError);
}

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

    const QRIterationResult unshifted = linalgebra::eigenvalues_unshifted(A, opts);
    const QRIterationResult shifted   = linalgebra::eigenvalues_shifted(A, opts);

    std::cout << "\n=== Shifted vs Unshifted ===\n";
    std::cout << "  Unshifted iterations: " << unshifted.iterations << "\n";
    std::cout << "    Shifted iterations: " << shifted.iterations   << "\n";
    std::cout << "=======================================================\n";

    CHECK(shifted.iterations < unshifted.iterations);

    CHECK(eigs_match(shifted.eigenvalues_real, shifted.eigenvalues_imag,
                     to_pairs(unshifted.eigenvalues_real, unshifted.eigenvalues_imag),
                     1e-8));
}

TEST_CASE("QR iteration (shifted): converges <20 iters where unshifted needs >100",
          "[qr_iteration][shifted]") {
    const Matrix A = random_symmetric(5, 17u);

    QRIterationOptions opts;
    opts.max_iterations = 2000;

    const QRIterationResult unshifted = linalgebra::eigenvalues_unshifted(A, opts);
    const QRIterationResult shifted   = linalgebra::eigenvalues_shifted(A, opts);

    std::cout << "\n=== Hard matrix ===\n";
    std::cout << "  Unshifted iterations: " << unshifted.iterations << "\n";
    std::cout << "    Shifted iterations: " << shifted.iterations   << "\n";

    CHECK(unshifted.iterations > 100);
    CHECK(shifted.iterations   <  20);
}

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
        const QRIterationResult res = linalgebra::eigenvalues_shifted(A);
        CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
    }

    SECTION("2x2 known eigenvalues") {
        const Matrix A{{2.0, 1.0}, {1.0, 2.0}};
        const EigPairs expected = {{1.0, 0.0}, {3.0, 0.0}};
        const QRIterationResult res = linalgebra::eigenvalues_shifted(A);
        CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
    }
}

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

double diff_norm(const Matrix& A, const Matrix& B) {
    double s = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) {
            const double d = A(i, j) - B(i, j);
            s += d * d;
        }
    return std::sqrt(s);
}

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
    const linalgebra::HessenbergResult hr = linalgebra::hessenberg_reduction(A);

    CHECK(is_upper_hessenberg(hr.H));
    CHECK(orthogonality_error(hr.Q) < 1e-10);

    const Matrix QtHQ = hr.Q * hr.H * linalgebra::transpose(hr.Q);
    CHECK(diff_norm(A, QtHQ) < 1e-10);
}

TEST_CASE("Hessenberg QR: eigenvalues match shifted QR to 1e-6",
          "[qr_iteration][shifted]") {
    const Matrix A = random_symmetric(8, 99u);

    const QRIterationResult ref = linalgebra::eigenvalues_shifted(A);
    const QRIterationResult hess = linalgebra::eigenvalues_hessenberg(A);

    REQUIRE(hess.eigenvalues_real.size() == 8);
    CHECK(eigs_match(hess.eigenvalues_real, hess.eigenvalues_imag,
                     to_pairs(ref.eigenvalues_real, ref.eigenvalues_imag),
                     1e-6));
}

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
    const QRIterationResult res = linalgebra::eigenvalues_hessenberg(A);
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
}

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
        { const auto tmp = linalgebra::eigenvalues_shifted(A); (void)tmp; }
        const double t_shifted = Seconds(Clock::now() - t0s).count();

        const auto t0h = Clock::now();
        const QRIterationResult hess = linalgebra::eigenvalues_hessenberg(A);
        const double t_hess = Seconds(Clock::now() - t0h).count();

        const double speedup = t_shifted / t_hess;

        std::cout << std::left  << std::setw(8)  << n
                  << std::fixed << std::setprecision(4)
                  << std::setw(16) << t_shifted
                  << std::setw(16) << t_hess
                  << std::setprecision(2)
                  << std::setw(12) << speedup << "x\n";

        CHECK(t_hess < t_shifted);

        const QRIterationResult ref = linalgebra::eigenvalues_shifted(A);
        CHECK(eigs_match(hess.eigenvalues_real, hess.eigenvalues_imag,
                         to_pairs(ref.eigenvalues_real, ref.eigenvalues_imag),
                         1e-6));
    }
    std::cout << "=============================================\n";
}

// ---------------------------------------------------------------------------
// Francis double-shift QR tests
// ---------------------------------------------------------------------------

TEST_CASE("Francis QR: 2x2 real eigenvalues", "[qr_iteration][francis]") {
    Matrix A{{3.0, 1.0}, {0.0, 2.0}};
    const QRIterationResult res = linalgebra::eigenvalues_francis(A);
    const EigPairs expected = {{3.0, 0.0}, {2.0, 0.0}};
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-10));
}

TEST_CASE("Francis QR: 2x2 complex eigenvalues", "[qr_iteration][francis]") {
    // Rotation matrix — eigenvalues are cos(theta) ± i*sin(theta).
    const double theta = 1.0;
    Matrix A{{std::cos(theta), -std::sin(theta)},
             {std::sin(theta),  std::cos(theta)}};
    const QRIterationResult res = linalgebra::eigenvalues_francis(A);
    const EigPairs expected = {
        {std::cos(theta),  std::sin(theta)},
        {std::cos(theta), -std::sin(theta)}
    };
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-10));
}

TEST_CASE("Francis QR: 3x3 with complex pair", "[qr_iteration][francis]") {
    // Block diagonal: 2x2 rotation (complex pair) + real eigenvalue.
    Matrix A{{0.0, -1.0, 0.0},
             {1.0,  0.0, 0.0},
             {0.0,  0.0, 5.0}};
    const QRIterationResult res = linalgebra::eigenvalues_francis(A);
    const EigPairs expected = {{0.0, 1.0}, {0.0, -1.0}, {5.0, 0.0}};
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-10));
}

TEST_CASE("Francis QR: symmetric matrix (all real)", "[qr_iteration][francis]") {
    const Matrix A = random_symmetric(10, 77u);
    const QRIterationResult ref = linalgebra::eigenvalues_hessenberg(A);
    const QRIterationResult res = linalgebra::eigenvalues_francis(A);
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag,
                     to_pairs(ref.eigenvalues_real, ref.eigenvalues_imag), 1e-8));
}

TEST_CASE("Francis QR: non-symmetric with complex pairs", "[qr_iteration][francis]") {
    // Random non-symmetric matrix — older single-shift methods struggle here,
    // but Francis double-shift handles it natively.
    std::mt19937 rng(99u);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    const std::size_t n = 8;
    Matrix A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = dist(rng);

    const QRIterationResult res = linalgebra::eigenvalues_francis(A);

    // Verify: for each eigenvalue λ, check that the characteristic polynomial
    // product of (λ_i - λ_j) is consistent — i.e., sum of eigenvalues = trace.
    double trace_A = 0.0;
    for (std::size_t i = 0; i < n; ++i) trace_A += A(i, i);

    double trace_eigs = 0.0;
    for (std::size_t i = 0; i < n; ++i) trace_eigs += res.eigenvalues_real[i];
    CHECK(trace_eigs == Catch::Approx(trace_A).margin(1e-6));

    // All imaginary parts should come in conjugate pairs.
    double imag_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) imag_sum += res.eigenvalues_imag[i];
    CHECK(imag_sum == Catch::Approx(0.0).margin(1e-8));
}

TEST_CASE("Francis QR: companion matrix", "[qr_iteration][francis]") {
    // Companion matrix for x^4 - 10x^3 + 35x^2 - 50x + 24 = (x-1)(x-2)(x-3)(x-4).
    Matrix A{{0.0, 0.0, 0.0, -24.0},
             {1.0, 0.0, 0.0,  50.0},
             {0.0, 1.0, 0.0, -35.0},
             {0.0, 0.0, 1.0,  10.0}};
    const QRIterationResult res = linalgebra::eigenvalues_francis(A);
    const EigPairs expected = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}};
    CHECK(eigs_match(res.eigenvalues_real, res.eigenvalues_imag, expected, 1e-8));
}

TEST_CASE("Francis QR: 1x1 matrix", "[qr_iteration][francis]") {
    Matrix A{{7.0}};
    const QRIterationResult res = linalgebra::eigenvalues_francis(A);
    CHECK(res.eigenvalues_real[0] == Catch::Approx(7.0));
    CHECK(res.eigenvalues_imag[0] == Catch::Approx(0.0).margin(1e-15));
}
