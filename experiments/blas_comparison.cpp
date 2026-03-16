// blas_comparison.cpp
//
// Compares every operation in this linalg library against the corresponding
// BLAS / LAPACK reference routine for correctness and performance.
//
// Sections:
//   §1  Level 1 BLAS  : ddot, dnrm2
//   §2  Level 2 BLAS  : dgemv  (matrix–vector multiply, square and rectangular)
//   §3  Level 3 BLAS  : dgemm  (matrix–matrix multiply)
//   §4  Triangular    : dtrsv  (forward and backward substitution)
//   §5  LU solve      : dgesv  (single RHS and multiple RHS)
//   §6  QR            : dgeqrf + dorgqr (vs all three of this library's QR methods)
//   §7  Ill-conditioned: LU solve and QR on Hilbert matrices
//
// Accuracy metric  : compare output to BLAS/LAPACK (or known exact solution).
// Performance metric: minimum wall-clock time over several trials.
//
// "this/blas" ratio < 1 means this library's implementation is faster.
//
// Note on LAPACK timing: calls to lapack_lu_solve() and lapack_qr() include
// to/from column-major conversion overhead because this library's storage is row-major.
// This is the real cost of calling LAPACK from a row-major library.

#ifdef __APPLE__
#  include <Accelerate/Accelerate.h>
   using lapack_int_t = __CLPK_integer;
#else
#  include <cblas.h>
   extern "C" {
       void dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
       void dgeqrf_(int*, int*, double*, int*, double*, double*, int*, int*);
       void dorgqr_(int*, int*, int*, double*, int*, double*, double*, int*, int*);
   }
   using lapack_int_t = int;
#endif

#include "lu.hpp"
#include "matrix.hpp"
#include "norms.hpp"
#include "qr.hpp"
#include "triangular_solve.hpp"
#include "vector.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using linalg::Matrix;
using linalg::Vector;
using Clock   = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;

// Volatile sink prevents the compiler from eliminating timed computations.
static volatile double g_sink = 0.0;

// ===========================================================================
// Random data (fixed seed for reproducibility)
// ===========================================================================

static std::mt19937_64 rng(0xDEADBEEF42ULL);

static double rand_dbl(double lo = -1.0, double hi = 1.0) {
    return std::uniform_real_distribution<double>(lo, hi)(rng);
}

static Vector random_vec(std::size_t n) {
    Vector v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = rand_dbl();
    return v;
}

static Matrix random_mat(std::size_t rows, std::size_t cols) {
    Matrix M(rows, cols);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            M(i, j) = rand_dbl();
    return M;
}

// Lower-triangular with diagonal entries in [1, 2] (well-conditioned).
static Matrix random_lower(std::size_t n) {
    Matrix L(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) L(i, j) = rand_dbl();
        L(i, i) = 1.0 + rand_dbl(0.0, 1.0);
    }
    return L;
}

// Upper-triangular with diagonal entries in [1, 2].
static Matrix random_upper(std::size_t n) {
    Matrix U(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        U(i, i) = 1.0 + rand_dbl(0.0, 1.0);
        for (std::size_t j = i + 1; j < n; ++j) U(i, j) = rand_dbl();
    }
    return U;
}

// Hilbert matrix H[i][j] = 1/(i+j+1).
static Matrix hilbert(std::size_t n) {
    Matrix H(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            H(i, j) = 1.0 / static_cast<double>(i + j + 1);
    return H;
}

// ===========================================================================
// Error metrics
// ===========================================================================

static double vec_l2(const Vector& v) {
    double s = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) s += v[i] * v[i];
    return std::sqrt(s);
}

static double vec_diff_l2(const Vector& a, const Vector& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

static double frob_diff(const Matrix& A, const Matrix& B) {
    double s = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) {
            const double d = A(i, j) - B(i, j);
            s += d * d;
        }
    return std::sqrt(s);
}

static double qr_recon_err(const Matrix& A, const linalg::QRResult& qr) {
    return frob_diff(A, qr.Q * qr.R);
}

static double qr_ortho_err(const linalg::QRResult& qr) {
    const Matrix& Q = qr.Q;
    const std::size_t n = Q.cols();
    const Matrix QtQ = linalg::transpose(Q) * Q;
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double d = QtQ(i, j) - (i == j ? 1.0 : 0.0);
            s += d * d;
        }
    return std::sqrt(s);
}

// ===========================================================================
// Column-major conversion (this library's Matrix is row-major; LAPACK expects col-major)
// ===========================================================================

static std::vector<double> to_col_major(const Matrix& A) {
    const std::size_t m = A.rows(), n = A.cols();
    std::vector<double> buf(m * n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            buf[j * m + i] = A(i, j);
    return buf;
}

static Matrix from_col_major(const std::vector<double>& buf,
                              std::size_t m, std::size_t n) {
    Matrix A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = buf[j * m + i];
    return A;
}

// ===========================================================================
// Timing
// ===========================================================================

template <typename Fn>
static double min_time_s(Fn fn, int trials) {
    double best = 1e30;
    for (int t = 0; t < trials; ++t) {
        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();
        best = std::min(best, Seconds(t1 - t0).count());
    }
    return best;
}

// ===========================================================================
// LAPACK wrappers
// ===========================================================================

// Solve A*x = b using LAPACK dgesv_ (LU with partial pivoting).
// Includes to/from column-major conversion.
static Vector lapack_lu_solve(const Matrix& A, const Vector& b) {
    const std::size_t n = A.rows();
    lapack_int_t ni   = static_cast<lapack_int_t>(n);
    lapack_int_t nrhs = 1;
    lapack_int_t lda  = ni;
    lapack_int_t ldb  = ni;
    lapack_int_t info = 0;

    std::vector<double>       a_cm(to_col_major(A));
    std::vector<double>       b_cm(b.data(), b.data() + n);
    std::vector<lapack_int_t> ipiv(n);

    dgesv_(&ni, &nrhs, a_cm.data(), &lda, ipiv.data(),
           b_cm.data(), &ldb, &info);

    if (info != 0) throw std::runtime_error("dgesv_ failed (info=" +
                                             std::to_string(info) + ")");
    Vector x(n);
    for (std::size_t i = 0; i < n; ++i) x[i] = b_cm[i];
    return x;
}

// Solve A*X = B using LAPACK dgesv_ with multiple RHS columns.
// Returns solution matrix X (n x nrhs), stored row-major.
static Matrix lapack_lu_solve_multi(const Matrix& A, const Matrix& B) {
    const std::size_t n    = A.rows();
    const std::size_t nrhs = B.cols();
    lapack_int_t ni    = static_cast<lapack_int_t>(n);
    lapack_int_t nrhsi = static_cast<lapack_int_t>(nrhs);
    lapack_int_t lda   = ni;
    lapack_int_t ldb   = ni;
    lapack_int_t info  = 0;

    std::vector<double>       a_cm(to_col_major(A));
    // B stored col-major for LAPACK: each RHS is a column
    std::vector<double>       b_cm(to_col_major(B));
    std::vector<lapack_int_t> ipiv(n);

    dgesv_(&ni, &nrhsi, a_cm.data(), &lda, ipiv.data(),
           b_cm.data(), &ldb, &info);

    if (info != 0) throw std::runtime_error("dgesv_ (multi) failed");
    return from_col_major(b_cm, n, nrhs);
}

// QR factorization via LAPACK dgeqrf_ + dorgqr_.
// Returns thin QR (m×n Q, n×n R), including col-major conversion overhead.
static std::optional<linalg::QRResult> lapack_qr(const Matrix& A) {
    const std::size_t m = A.rows(), n = A.cols();
    lapack_int_t mi  = static_cast<lapack_int_t>(m);
    lapack_int_t ni  = static_cast<lapack_int_t>(n);
    lapack_int_t ki  = ni;   // number of reflectors = n for square/tall A
    lapack_int_t lda = mi;   // column-major leading dimension
    lapack_int_t info = 0;

    std::vector<double> a_cm(to_col_major(A));
    std::vector<double> tau(n);

    // --- dgeqrf: workspace query then factorize ---
    {
        lapack_int_t lwork = -1;
        double wq = 0.0;
        dgeqrf_(&mi, &ni, a_cm.data(), &lda, tau.data(), &wq, &lwork, &info);
        if (info != 0) return std::nullopt;
        lwork = static_cast<lapack_int_t>(wq);
        std::vector<double> work(static_cast<std::size_t>(lwork));
        dgeqrf_(&mi, &ni, a_cm.data(), &lda, tau.data(),
                work.data(), &lwork, &info);
        if (info != 0) return std::nullopt;
    }

    // Extract R from the upper triangle of a_cm *before* dorgqr overwrites it.
    Matrix R(n, n, 0.0);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i <= j; ++i)
            R(i, j) = a_cm[j * m + i];

    // --- dorgqr: workspace query then form explicit Q ---
    {
        lapack_int_t lwork = -1;
        double wq = 0.0;
        dorgqr_(&mi, &ni, &ki, a_cm.data(), &lda, tau.data(),
                &wq, &lwork, &info);
        if (info != 0) return std::nullopt;
        lwork = static_cast<lapack_int_t>(wq);
        std::vector<double> work(static_cast<std::size_t>(lwork));
        dorgqr_(&mi, &ni, &ki, a_cm.data(), &lda, tau.data(),
                work.data(), &lwork, &info);
        if (info != 0) return std::nullopt;
    }

    Matrix Q = from_col_major(a_cm, m, n);
    return linalg::QRResult{Q, R};
}

// ===========================================================================
// Formatting helpers
// ===========================================================================

static void separator(char c = '=', int w = 78) {
    std::cout << std::string(static_cast<std::size_t>(w), c) << "\n";
}

static void ratio_col(double r) {
    // Print ratio with a directional note.
    std::cout << std::fixed << std::setprecision(2)
              << std::setw(10) << r
              << (r < 1.0 ? " (faster)\n" : " (slower)\n");
}

// ===========================================================================
// §1  Level 1 BLAS — ddot and dnrm2
// ===========================================================================

static void section_level1() {
    std::cout << "\n"; separator();
    std::cout << "  §1  Level 1 BLAS — dot product (ddot) and L2 norm (dnrm2)\n";
    separator();
    std::cout << "\n";

    const std::vector<std::size_t> sizes = {64, 256, 1024, 4096, 16384, 65536};
    constexpr int trials = 30;

    // ---- ddot ----
    std::cout << "  cblas_ddot  vs  linalg::dot\n\n";
    std::cout << std::left
              << std::setw(10) << "n"
              << std::setw(18) << "|this - blas|"
              << std::setw(14) << "blas µs"
              << std::setw(14) << "this µs"
              << std::setw(10) << "this/blas"
              << "\n";
    separator('-', 66);

    for (std::size_t n : sizes) {
        const Vector x  = random_vec(n);
        const Vector y  = random_vec(n);
        const int    ni = static_cast<int>(n);

        const double dot_blas = cblas_ddot(ni, x.data(), 1, y.data(), 1);
        const double dot_ours = linalg::dot(x, y);

        const double t_blas = min_time_s([&]{
            g_sink += cblas_ddot(ni, x.data(), 1, y.data(), 1);
        }, trials);
        const double t_ours = min_time_s([&]{
            g_sink += linalg::dot(x, y);
        }, trials);

        std::cout << std::left  << std::setw(10) << n
                  << std::scientific << std::setprecision(2)
                  << std::setw(18) << std::abs(dot_ours - dot_blas)
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << t_blas * 1e6
                  << std::setw(14) << t_ours * 1e6;
        ratio_col(t_ours / t_blas);
    }

    // ---- dnrm2 ----
    std::cout << "\n  cblas_dnrm2  vs  linalg::norm2\n\n";
    std::cout << std::left
              << std::setw(10) << "n"
              << std::setw(18) << "|this - blas|"
              << std::setw(14) << "blas µs"
              << std::setw(14) << "this µs"
              << std::setw(10) << "this/blas"
              << "\n";
    separator('-', 66);

    for (std::size_t n : sizes) {
        const Vector x  = random_vec(n);
        const int    ni = static_cast<int>(n);

        const double nrm_blas = cblas_dnrm2(ni, x.data(), 1);
        const double nrm_ours = linalg::norm2(x);

        const double t_blas = min_time_s([&]{
            g_sink += cblas_dnrm2(ni, x.data(), 1);
        }, trials);
        const double t_ours = min_time_s([&]{
            g_sink += linalg::norm2(x);
        }, trials);

        std::cout << std::left  << std::setw(10) << n
                  << std::scientific << std::setprecision(2)
                  << std::setw(18) << std::abs(nrm_ours - nrm_blas)
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << t_blas * 1e6
                  << std::setw(14) << t_ours * 1e6;
        ratio_col(t_ours / t_blas);
    }
}

// ===========================================================================
// §2  Level 2 BLAS — dgemv  (y = A x)
// ===========================================================================

static void section_dgemv() {
    std::cout << "\n"; separator();
    std::cout << "  §2  Level 2 BLAS — matrix–vector multiply (dgemv)\n";
    separator();
    std::cout << "\n";

    constexpr int trials = 20;

    auto run_dgemv = [&](const std::vector<std::size_t>& row_sizes,
                         const std::vector<std::size_t>& col_sizes,
                         const std::string& label) {
        std::cout << "  " << label << "\n\n";
        std::cout << std::left
                  << std::setw(8)  << "rows"
                  << std::setw(8)  << "cols"
                  << std::setw(20) << "||y_this - y_blas||"
                  << std::setw(14) << "blas µs"
                  << std::setw(14) << "this µs"
                  << std::setw(10) << "this/blas"
                  << "\n";
        separator('-', 74);

        for (std::size_t i = 0; i < row_sizes.size(); ++i) {
            const std::size_t m  = row_sizes[i];
            const std::size_t k  = col_sizes[i];
            const Matrix A       = random_mat(m, k);
            const Vector x       = random_vec(k);
            const int    mi      = static_cast<int>(m);
            const int    ki      = static_cast<int>(k);

            Vector y_blas(m, 0.0);
            cblas_dgemv(CblasRowMajor, CblasNoTrans, mi, ki,
                        1.0, A.data(), ki, x.data(), 1,
                        0.0, y_blas.data(), 1);
            const Vector y_ours = A * x;
            const double err    = vec_diff_l2(y_ours, y_blas);

            const double t_blas = min_time_s([&]{
                Vector tmp(m, 0.0);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, mi, ki,
                            1.0, A.data(), ki, x.data(), 1,
                            0.0, tmp.data(), 1);
                g_sink += tmp[0];
            }, trials);
            const double t_ours = min_time_s([&]{
                Vector r = A * x;
                g_sink += r[0];
            }, trials);

            std::cout << std::left  << std::setw(8) << m
                      << std::setw(8) << k
                      << std::scientific << std::setprecision(2)
                      << std::setw(20) << err
                      << std::fixed << std::setprecision(3)
                      << std::setw(14) << t_blas * 1e6
                      << std::setw(14) << t_ours * 1e6;
            ratio_col(t_ours / t_blas);
        }
        std::cout << "\n";
    };

    // Square matrices
    run_dgemv({8, 32, 64, 128, 256, 512, 1024},
              {8, 32, 64, 128, 256, 512, 1024},
              "Square  y = A*x,  A is n×n");

    // Tall matrices (more rows than cols)
    run_dgemv({256, 512, 1024, 2048},
              { 32,  64,  128,  256},
              "Tall  y = A*x,  A is m×k  (m >> k)");

    // Wide matrices (more cols than rows)
    run_dgemv({ 32,  64,  128,  256},
              {256, 512, 1024, 2048},
              "Wide  y = A*x,  A is m×k  (m << k)");
}

// ===========================================================================
// §3  Level 3 BLAS — dgemm  (C = A B)
// ===========================================================================

static void section_dgemm() {
    std::cout << "\n"; separator();
    std::cout << "  §3  Level 3 BLAS — matrix–matrix multiply (dgemm)\n";
    separator();
    std::cout << "\n";

    const std::vector<std::size_t> sizes = {8, 32, 64, 128, 256, 512};
    constexpr std::size_t thresh_small   = 128;
    constexpr int         trials_small   = 10;
    constexpr int         trials_large   = 3;

    std::cout << "  C = A*B,  all matrices n×n\n\n";
    std::cout << std::left
              << std::setw(8)  << "n"
              << std::setw(20) << "||C_this - C_blas||_F"
              << std::setw(12) << "blas ms"
              << std::setw(12) << "this ms"
              << std::setw(14) << "GFLOP/s blas"
              << std::setw(14) << "GFLOP/s this"
              << std::setw(10) << "this/blas"
              << "\n";
    separator('-', 90);

    for (std::size_t n : sizes) {
        const Matrix A  = random_mat(n, n);
        const Matrix B  = random_mat(n, n);
        const int    ni = static_cast<int>(n);
        const int    trials = (n <= thresh_small) ? trials_small : trials_large;

        Matrix C_blas(n, n, 0.0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    ni, ni, ni, 1.0, A.data(), ni, B.data(), ni,
                    0.0, C_blas.data(), ni);

        const Matrix C_ours = A * B;
        const double err    = frob_diff(C_ours, C_blas);

        const double t_blas = min_time_s([&]{
            Matrix tmp(n, n, 0.0);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        ni, ni, ni, 1.0, A.data(), ni, B.data(), ni,
                        0.0, tmp.data(), ni);
            g_sink += tmp(0, 0);
        }, trials);
        const double t_ours = min_time_s([&]{
            Matrix r = A * B;
            g_sink += r(0, 0);
        }, trials);

        const double fp_ops  = 2.0 * static_cast<double>(n)
                                   * static_cast<double>(n)
                                   * static_cast<double>(n);
        const double gf_blas = fp_ops / t_blas / 1e9;
        const double gf_ours = fp_ops / t_ours / 1e9;

        std::cout << std::left  << std::setw(8) << n
                  << std::scientific << std::setprecision(2)
                  << std::setw(20) << err
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << t_blas * 1e3
                  << std::setw(12) << t_ours * 1e3
                  << std::setprecision(2)
                  << std::setw(14) << gf_blas
                  << std::setw(14) << gf_ours;
        ratio_col(t_ours / t_blas);
    }

    // Non-square: C (m×n) = A (m×k) * B (k×n)
    std::cout << "\n  Non-square  C = A*B,  shapes (m×k) * (k×n) -> m×n\n\n";
    std::cout << std::left
              << std::setw(8)  << "m"
              << std::setw(8)  << "k"
              << std::setw(8)  << "n"
              << std::setw(20) << "||C_this - C_blas||_F"
              << std::setw(14) << "blas µs"
              << std::setw(14) << "this µs"
              << std::setw(10) << "this/blas"
              << "\n";
    separator('-', 82);

    const std::vector<std::array<std::size_t,3>> shapes = {
        {64, 32,  128},
        {128, 64, 256},
        {256, 128, 64},
        {512, 32,  256},
    };

    for (const auto& [m, k, nc] : shapes) {
        const Matrix A  = random_mat(m, k);
        const Matrix B  = random_mat(k, nc);
        const int    mi = static_cast<int>(m);
        const int    ki = static_cast<int>(k);
        const int    ni = static_cast<int>(nc);

        Matrix C_blas(m, nc, 0.0);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    mi, ni, ki, 1.0, A.data(), ki, B.data(), ni,
                    0.0, C_blas.data(), ni);

        const Matrix C_ours = A * B;
        const double err    = frob_diff(C_ours, C_blas);

        const double t_blas = min_time_s([&]{
            Matrix tmp(m, nc, 0.0);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        mi, ni, ki, 1.0, A.data(), ki, B.data(), ni,
                        0.0, tmp.data(), ni);
            g_sink += tmp(0, 0);
        }, 10);
        const double t_ours = min_time_s([&]{
            Matrix r = A * B;
            g_sink += r(0, 0);
        }, 10);

        std::cout << std::left  << std::setw(8) << m
                  << std::setw(8) << k
                  << std::setw(8) << nc
                  << std::scientific << std::setprecision(2)
                  << std::setw(20) << err
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << t_blas * 1e6
                  << std::setw(14) << t_ours * 1e6;
        ratio_col(t_ours / t_blas);
    }

    // A^T * B (transposed LHS)
    std::cout << "\n  Transposed  C = A^T * B,  A is k×m, B is k×n -> m×n\n"
              << "  (BLAS uses CblasTrans; this library calls linalg::transpose(A) * B)\n\n";
    std::cout << std::left
              << std::setw(8)  << "k"
              << std::setw(8)  << "m"
              << std::setw(8)  << "n"
              << std::setw(20) << "||C_this - C_blas||_F"
              << std::setw(14) << "blas µs"
              << std::setw(14) << "this µs"
              << std::setw(10) << "this/blas"
              << "\n";
    separator('-', 82);

    for (std::size_t sz : {64UL, 128UL, 256UL}) {
        const std::size_t k  = sz;
        const std::size_t mm = sz / 2;
        const std::size_t nc = sz;
        const Matrix A  = random_mat(k, mm);   // k × m
        const Matrix B  = random_mat(k, nc);   // k × n
        const int    ki = static_cast<int>(k);
        const int    mi = static_cast<int>(mm);
        const int    ni = static_cast<int>(nc);

        // BLAS: C = A^T * B  using CblasTrans for A
        Matrix C_blas(mm, nc, 0.0);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    mi, ni, ki, 1.0, A.data(), mi, B.data(), ni,
                    0.0, C_blas.data(), ni);

        const Matrix C_ours = linalg::transpose(A) * B;
        const double err    = frob_diff(C_ours, C_blas);

        const double t_blas = min_time_s([&]{
            Matrix tmp(mm, nc, 0.0);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        mi, ni, ki, 1.0, A.data(), mi, B.data(), ni,
                        0.0, tmp.data(), ni);
            g_sink += tmp(0, 0);
        }, 10);
        const double t_ours = min_time_s([&]{
            Matrix r = linalg::transpose(A) * B;
            g_sink += r(0, 0);
        }, 10);

        std::cout << std::left  << std::setw(8) << k
                  << std::setw(8) << mm
                  << std::setw(8) << nc
                  << std::scientific << std::setprecision(2)
                  << std::setw(20) << err
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << t_blas * 1e6
                  << std::setw(14) << t_ours * 1e6;
        ratio_col(t_ours / t_blas);
    }
}

// ===========================================================================
// §4  Triangular solve — dtrsv vs forward/backward_substitution
// ===========================================================================

static void section_dtrsv() {
    std::cout << "\n"; separator();
    std::cout << "  §4  Triangular solve — dtrsv vs forward/backward_substitution\n";
    separator();
    std::cout << "\n";

    const std::vector<std::size_t> sizes = {8, 32, 64, 128, 256, 512, 1024};
    constexpr int trials = 20;

    auto print_header = [] {
        std::cout << std::left
                  << std::setw(8)  << "n"
                  << std::setw(20) << "||x_this - x_blas||"
                  << std::setw(14) << "blas µs"
                  << std::setw(14) << "this µs"
                  << std::setw(10) << "this/blas"
                  << "\n";
        separator('-', 66);
    };

    // ---- Forward substitution: Lx = b ----
    std::cout << "  Forward substitution  Lx = b  (L lower triangular, non-unit diagonal)\n\n";
    print_header();

    for (std::size_t n : sizes) {
        const Matrix L  = random_lower(n);
        const Vector b  = random_vec(n);
        const int    ni = static_cast<int>(n);

        Vector x_blas = b;
        cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    ni, L.data(), ni, x_blas.data(), 1);
        const Vector x_ours = linalg::forward_substitution(L, b);
        const double err    = vec_diff_l2(x_ours, x_blas);

        const double t_blas = min_time_s([&]{
            Vector tmp = b;
            cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                        ni, L.data(), ni, tmp.data(), 1);
            g_sink += tmp[0];
        }, trials);
        const double t_ours = min_time_s([&]{
            Vector r = linalg::forward_substitution(L, b);
            g_sink += r[0];
        }, trials);

        std::cout << std::left  << std::setw(8) << n
                  << std::scientific << std::setprecision(2)
                  << std::setw(20) << err
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << t_blas * 1e6
                  << std::setw(14) << t_ours * 1e6;
        ratio_col(t_ours / t_blas);
    }

    // ---- Backward substitution: Ux = b ----
    std::cout << "\n  Backward substitution  Ux = b  (U upper triangular, non-unit diagonal)\n\n";
    print_header();

    for (std::size_t n : sizes) {
        const Matrix U  = random_upper(n);
        const Vector b  = random_vec(n);
        const int    ni = static_cast<int>(n);

        Vector x_blas = b;
        cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    ni, U.data(), ni, x_blas.data(), 1);
        const Vector x_ours = linalg::backward_substitution(U, b);
        const double err    = vec_diff_l2(x_ours, x_blas);

        const double t_blas = min_time_s([&]{
            Vector tmp = b;
            cblas_dtrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        ni, U.data(), ni, tmp.data(), 1);
            g_sink += tmp[0];
        }, trials);
        const double t_ours = min_time_s([&]{
            Vector r = linalg::backward_substitution(U, b);
            g_sink += r[0];
        }, trials);

        std::cout << std::left  << std::setw(8) << n
                  << std::scientific << std::setprecision(2)
                  << std::setw(20) << err
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << t_blas * 1e6
                  << std::setw(14) << t_ours * 1e6;
        ratio_col(t_ours / t_blas);
    }

    // ---- Unit-diagonal forward substitution ----
    std::cout << "\n  Forward substitution  Lx = b  (unit diagonal)\n\n";
    print_header();

    for (std::size_t n : sizes) {
        // Build unit lower triangular
        Matrix L = random_lower(n);
        for (std::size_t i = 0; i < n; ++i) L(i, i) = 1.0;
        const Vector b  = random_vec(n);
        const int    ni = static_cast<int>(n);

        Vector x_blas = b;
        cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit,
                    ni, L.data(), ni, x_blas.data(), 1);
        const Vector x_ours = linalg::forward_substitution(L, b,
                                  /*singular_tolerance=*/1e-12,
                                  /*unit_diagonal=*/true);
        const double err = vec_diff_l2(x_ours, x_blas);

        const double t_blas = min_time_s([&]{
            Vector tmp = b;
            cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit,
                        ni, L.data(), ni, tmp.data(), 1);
            g_sink += tmp[0];
        }, trials);
        const double t_ours = min_time_s([&]{
            Vector r = linalg::forward_substitution(L, b, 1e-12, true);
            g_sink += r[0];
        }, trials);

        std::cout << std::left  << std::setw(8) << n
                  << std::scientific << std::setprecision(2)
                  << std::setw(20) << err
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << t_blas * 1e6
                  << std::setw(14) << t_ours * 1e6;
        ratio_col(t_ours / t_blas);
    }
}

// ===========================================================================
// §5  LU solve — dgesv vs lu_factor + lu_solve
// ===========================================================================

static void section_lu_solve() {
    std::cout << "\n"; separator();
    std::cout << "  §5  LU solve — dgesv vs lu_factor + lu_solve\n";
    separator();
    std::cout << "\n";

    // ---- Single RHS ----
    std::cout << "  Single RHS  Ax = b\n\n";
    std::cout << std::left
              << std::setw(8)  << "n"
              << std::setw(18) << "||x_this-x_blas||"
              << std::setw(18) << "||res_this||"
              << std::setw(18) << "||res_blas||"
              << std::setw(12) << "blas µs"
              << std::setw(12) << "this µs"
              << std::setw(10) << "this/blas"
              << "\n";
    separator('-', 96);

    const std::vector<std::size_t> sizes = {8, 32, 64, 128, 256, 512};
    constexpr std::size_t thresh = 128;
    constexpr int ts = 20, tl = 5;

    for (std::size_t n : sizes) {
        const Matrix A  = random_mat(n, n);
        const Vector b  = random_vec(n);
        const int trials = (n <= thresh) ? ts : tl;

        const Vector x_blas = lapack_lu_solve(A, b);
        const auto   lu     = linalg::lu_factor(A);
        const Vector x_ours = linalg::lu_solve(lu, b);

        const Vector res_ours = (A * x_ours) - b;
        const Vector res_blas = (A * x_blas) - b;

        const double t_blas = min_time_s([&]{
            Vector r = lapack_lu_solve(A, b);
            g_sink += r[0];
        }, trials);
        const double t_ours = min_time_s([&]{
            auto lu2 = linalg::lu_factor(A);
            Vector r = linalg::lu_solve(lu2, b);
            g_sink += r[0];
        }, trials);

        std::cout << std::left  << std::setw(8) << n
                  << std::scientific << std::setprecision(2)
                  << std::setw(18) << vec_diff_l2(x_ours, x_blas)
                  << std::setw(18) << vec_l2(res_ours)
                  << std::setw(18) << vec_l2(res_blas)
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << t_blas * 1e6
                  << std::setw(12) << t_ours * 1e6;
        ratio_col(t_ours / t_blas);
    }

    // ---- Multiple RHS ----
    // LAPACK dgesv handles multiple RHS in one shot.
    // Our lu_solve only handles one vector at a time; we loop.
    std::cout << "\n  Multiple RHS  AX = B  (nrhs = 8)\n"
              << "  BLAS: one dgesv call.  Ours: lu_factor once, lu_solve 8 times.\n\n";
    std::cout << std::left
              << std::setw(8)  << "n"
              << std::setw(22) << "||X_this - X_blas||_F"
              << std::setw(18) << "||res_this||_F"
              << std::setw(18) << "||res_blas||_F"
              << std::setw(12) << "blas µs"
              << std::setw(12) << "this µs"
              << std::setw(10) << "this/blas"
              << "\n";
    separator('-', 100);

    constexpr std::size_t nrhs = 8;

    for (std::size_t n : sizes) {
        const Matrix A  = random_mat(n, n);
        const Matrix B  = random_mat(n, nrhs);
        const int trials = (n <= thresh) ? ts : tl;

        // LAPACK (single call, multiple RHS)
        const Matrix X_blas = lapack_lu_solve_multi(A, B);

        // Ours: factor once, solve per column
        const auto lu = linalg::lu_factor(A);
        Matrix X_ours(n, nrhs);
        for (std::size_t j = 0; j < nrhs; ++j) {
            Vector col_b(n);
            for (std::size_t i = 0; i < n; ++i) col_b[i] = B(i, j);
            const Vector col_x = linalg::lu_solve(lu, col_b);
            for (std::size_t i = 0; i < n; ++i) X_ours(i, j) = col_x[i];
        }

        const double sol_err  = frob_diff(X_ours, X_blas);
        const double res_ours = frob_diff(A * X_ours, B);
        const double res_blas = frob_diff(A * X_blas, B);

        const double t_blas = min_time_s([&]{
            Matrix r = lapack_lu_solve_multi(A, B);
            g_sink += r(0, 0);
        }, trials);
        const double t_ours = min_time_s([&]{
            auto lu2 = linalg::lu_factor(A);
            for (std::size_t j = 0; j < nrhs; ++j) {
                Vector col_b(n);
                for (std::size_t i = 0; i < n; ++i) col_b[i] = B(i, j);
                Vector col_x = linalg::lu_solve(lu2, col_b);
                g_sink += col_x[0];
            }
        }, trials);

        std::cout << std::left  << std::setw(8) << n
                  << std::scientific << std::setprecision(2)
                  << std::setw(22) << sol_err
                  << std::setw(18) << res_ours
                  << std::setw(18) << res_blas
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << t_blas * 1e6
                  << std::setw(12) << t_ours * 1e6;
        ratio_col(t_ours / t_blas);
    }
    std::cout << "  Note: blas timing includes column-major conversion.\n";
}

// ===========================================================================
// §6  QR factorization — dgeqrf+dorgqr vs this library's three methods
// ===========================================================================

static void section_qr() {
    std::cout << "\n"; separator();
    std::cout << "  §6  QR factorization — dgeqrf+dorgqr vs this library's three methods\n";
    separator();
    std::cout << "\n";
    std::cout << "  Metrics per method:\n"
              << "    ||A - QR||_F  : reconstruction error\n"
              << "    ||Q^TQ - I||_F: orthogonality loss\n"
              << "    time µs       : minimum wall-clock time\n"
              << "  (lapack timing includes to/from col-major conversion)\n\n";

    using OurFn = std::function<linalg::QRResult(const Matrix&)>;

    const std::vector<std::pair<std::string, OurFn>> methods = {
        {"lapack",       [](const Matrix& A) -> linalg::QRResult {
                             auto r = lapack_qr(A);
                             if (!r) throw std::runtime_error("lapack_qr failed");
                             return *r;
                         }},
        {"classical_gs", [](const Matrix& A){ return linalg::qr_classical_gs(A); }},
        {"modified_gs",  [](const Matrix& A){ return linalg::qr_modified_gs(A); }},
        {"householder",  [](const Matrix& A){ return linalg::qr_householder(A); }},
    };

    auto run_qr_block = [&](const std::vector<std::size_t>& row_vec,
                             const std::vector<std::size_t>& col_vec,
                             const std::string& label) {
        for (std::size_t idx = 0; idx < row_vec.size(); ++idx) {
            const std::size_t m = row_vec[idx];
            const std::size_t n = col_vec[idx];
            const Matrix A = random_mat(m, n);
            const int trials = (n <= 64) ? 15 : (n <= 128 ? 8 : 4);

            separator('-', 78);
            std::cout << "  " << label << "  m=" << m << "  n=" << n << "\n\n";
            std::cout << std::left
                      << std::setw(16) << "method"
                      << std::setw(16) << "||A-QR||_F"
                      << std::setw(16) << "||QtQ-I||_F"
                      << std::setw(12) << "time µs"
                      << "\n";
            separator('-', 60);

            for (const auto& [name, fn] : methods) {
                try {
                    const linalg::QRResult qr = fn(A);
                    const double re = qr_recon_err(A, qr);
                    const double oe = qr_ortho_err(qr);
                    const double t  = min_time_s([&]{ fn(A); }, trials);

                    std::cout << std::left << std::setw(16) << name
                              << std::scientific << std::setprecision(2)
                              << std::setw(16) << re
                              << std::setw(16) << oe
                              << std::fixed << std::setprecision(2)
                              << std::setw(12) << t * 1e6
                              << "\n";
                } catch (const std::exception& e) {
                    std::cout << std::left << std::setw(16) << name
                              << "  FAILED: " << e.what() << "\n";
                }
            }
            std::cout << "\n";
        }
    };

    run_qr_block({8, 32, 64, 128, 256}, {8, 32, 64, 128, 256},
                 "Square random");

    run_qr_block({128, 256, 512, 256}, {32, 64, 64, 128},
                 "Tall rectangular (m > n)");
}

// ===========================================================================
// §7  Ill-conditioned accuracy — Hilbert matrices
// ===========================================================================

static void section_ill_conditioned() {
    std::cout << "\n"; separator();
    std::cout << "  §7  Accuracy on ill-conditioned systems (Hilbert matrices)\n";
    separator();
    std::cout << "\n";
    std::cout << "  H[i][j] = 1/(i+j+1).  Condition number grows ~exponentially.\n"
              << "  LU: true solution x* = ones  (b = H * ones).\n"
              << "  QR: reconstruction and orthogonality errors.\n\n";

    const std::vector<std::size_t> sizes = {4, 6, 8, 10, 12, 14};

    // ---- LU solve ----
    std::cout << "  LU solve on Hilbert matrices\n\n";
    std::cout << std::left
              << std::setw(6)  << "n"
              << std::setw(20) << "||res_this||"
              << std::setw(20) << "||res_blas||"
              << std::setw(20) << "||x_this - x*||"
              << std::setw(20) << "||x_blas - x*||"
              << "\n";
    separator('-', 86);

    for (std::size_t n : sizes) {
        const Matrix H    = hilbert(n);
        const Vector ones(n, 1.0);
        const Vector b    = H * ones;

        try {
            const auto   lu     = linalg::lu_factor(H);
            const Vector x_ours = linalg::lu_solve(lu, b);
            const Vector x_blas = lapack_lu_solve(H, b);

            const Vector res_ours = (H * x_ours) - b;
            const Vector res_blas = (H * x_blas) - b;
            const Vector err_ours = x_ours - ones;
            const Vector err_blas = x_blas - ones;

            std::cout << std::left  << std::setw(6) << n
                      << std::scientific << std::setprecision(2)
                      << std::setw(20) << vec_l2(res_ours)
                      << std::setw(20) << vec_l2(res_blas)
                      << std::setw(20) << vec_l2(err_ours)
                      << std::setw(20) << vec_l2(err_blas)
                      << "\n";
        } catch (const std::exception& e) {
            std::cout << std::setw(6) << n
                      << "  FAILED: " << e.what() << "\n";
        }
    }

    // ---- QR on Hilbert matrices ----
    std::cout << "\n  QR factorization on Hilbert matrices\n\n";
    std::cout << std::left
              << std::setw(6)  << "n"
              << std::setw(16) << "method"
              << std::setw(18) << "||A-QR||_F"
              << std::setw(18) << "||QtQ-I||_F"
              << "\n";
    separator('-', 58);

    const std::vector<std::size_t> qr_sizes = {4, 6, 8, 10, 12};

    using OurFn2 = std::function<linalg::QRResult(const Matrix&)>;
    const std::vector<std::pair<std::string, OurFn2>> methods2 = {
        {"lapack",       [](const Matrix& A) -> linalg::QRResult {
                             auto r = lapack_qr(A);
                             if (!r) throw std::runtime_error("failed");
                             return *r;
                         }},
        {"classical_gs", [](const Matrix& A){ return linalg::qr_classical_gs(A); }},
        {"modified_gs",  [](const Matrix& A){ return linalg::qr_modified_gs(A); }},
        {"householder",  [](const Matrix& A){ return linalg::qr_householder(A); }},
    };

    for (std::size_t n : qr_sizes) {
        const Matrix H = hilbert(n);
        bool first = true;
        for (const auto& [name, fn] : methods2) {
            try {
                const linalg::QRResult qr = fn(H);
                const double re = qr_recon_err(H, qr);
                const double oe = qr_ortho_err(qr);
                std::cout << std::left
                          << std::setw(6)  << (first ? std::to_string(n) : "")
                          << std::setw(16) << name
                          << std::scientific << std::setprecision(2)
                          << std::setw(18) << re
                          << std::setw(18) << oe
                          << "\n";
            } catch (const std::exception& e) {
                std::cout << std::setw(6)  << (first ? std::to_string(n) : "")
                          << std::setw(16) << name
                          << "  FAILED: " << e.what() << "\n";
            }
            first = false;
        }
        std::cout << "\n";
    }
}

// ===========================================================================
// main
// ===========================================================================

int main() {
    separator('*');
    std::cout << "  BLAS / LAPACK vs linalg";
    separator('*');

    section_level1();
    section_dgemv();
    section_dgemm();
    section_dtrsv();
    section_lu_solve();
    section_qr();
    section_ill_conditioned();

    return 0;
}
