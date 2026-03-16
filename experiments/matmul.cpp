#include "linalg_error.hpp"
#include "matrix.hpp"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__AVX512F__)
#   define MATMUL_BACKEND "AVX512"
#elif !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__AVX2__)
#   define MATMUL_BACKEND "AVX2"
#elif !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__AVX__)
#   define MATMUL_BACKEND "AVX"
#elif !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && \
      defined(__ARM_NEON) && defined(__aarch64__) && \
      defined(__ARM_FEATURE_FP64_VECTOR_ARITHMETIC)
#   define MATMUL_BACKEND "NEON"
#else
#   define MATMUL_BACKEND "scalar"
#endif

using linalg::Matrix;
using Clock   = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;


Matrix naive_matmul(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.cols() != rhs.rows()) {
        throw linalg::DimensionMismatchError(
            "naive_matmul: lhs.cols() != rhs.rows()");
    }
    const std::size_t m = lhs.rows();
    const std::size_t n = rhs.cols();
    const std::size_t k = lhs.cols();

    Matrix result(m, n, 0.0);

    const double* A = lhs.data();
    const double* B = rhs.data();
    double*       C = result.data();

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t p = 0; p < k; ++p)
                C[i * n + j] += A[i * k + p] * B[p * n + j];

    return result;
}

// --- Helpers ---

Matrix make_matrix(std::size_t n) {
    Matrix M(n, n);
    const double inv = 1.0 / static_cast<double>(n + 1);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            M(i, j) = static_cast<double>(i + j + 1) * inv;
    return M;
}

template <typename Fn>
double min_time_s(Fn fn, int trials) {
    double best = 1e30;
    for (int t = 0; t < trials; ++t) {
        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();
        const double elapsed = Seconds(t1 - t0).count();
        if (elapsed < best) best = elapsed;
    }
    return best;
}

double flops(std::size_t n) {
    const double nd = static_cast<double>(n);
    return 2.0 * nd * nd * nd;
}


int main() {
    const std::vector<std::size_t> sizes = {
        8, 16, 32, 64, 128, 256, 512
    };

    constexpr std::size_t small_threshold = 128;
    constexpr int         trials_small    = 9;
    constexpr int         trials_large    = 3;

    std::cout << std::string(72, '*') << "\n";
    std::cout << "  Matmul benchmark: naive (ijk) vs SIMD (" MATMUL_BACKEND
                 ") + transpose\n";
    std::cout << "  C = A * B,  A and B both n×n\n";
    std::cout << std::string(72, '*') << "\n\n";

    std::cout << std::left
              << std::setw(6)  << "n"
              << std::setw(14) << "naive ms"
              << std::setw(14) << "naive GFLOP/s"
              << std::setw(14) << "SIMD ms"
              << std::setw(14) << "SIMD GFLOP/s"
              << std::setw(10) << "speedup"
              << "\n";
    std::cout << std::string(72, '-') << "\n";

    for (const std::size_t n : sizes) {
        const Matrix A = make_matrix(n);
        const Matrix B = make_matrix(n);

        const int trials = (n <= small_threshold) ? trials_small : trials_large;

        volatile double sink_naive = naive_matmul(A, B)(0, 0);
        volatile double sink_simd  = (A * B)(0, 0);
        (void)sink_naive;
        (void)sink_simd;

        const double t_naive = min_time_s([&] { (void)naive_matmul(A, B); }, trials);
        const double t_simd = min_time_s([&] { (void)(A * B); }, trials);

        const double fp = flops(n);
        const double gf_naive = fp / t_naive / 1e9;
        const double gf_simd = fp / t_simd / 1e9;
        const double speedup = t_naive / t_simd;

        std::cout << std::left  << std::setw(6) << n
                  << std::fixed << std::setprecision(3)
                  << std::setw(14) << t_naive * 1e3
                  << std::setprecision(2)
                  << std::setw(14) << gf_naive
                  << std::setprecision(3)
                  << std::setw(14) << t_simd * 1e3
                  << std::setprecision(2)
                  << std::setw(14) << gf_simd
                  << std::setprecision(2) << std::setw(10) << speedup
                  << "x\n";
    }

    std::cout << "\n(each cell = minimum over "
              << trials_small << " trials for n<=" << small_threshold
              << ", " << trials_large << " trials for larger n)\n";

    return 0;
}
