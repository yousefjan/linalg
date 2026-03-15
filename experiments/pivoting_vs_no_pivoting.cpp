// Experiment: partial pivoting vs no-pivot LU
//
// Demonstrates why partial pivoting is essential for numerical stability.
// Run the binary and inspect the residuals printed to stdout.

#include "lu.hpp"
#include "matrix.hpp"
#include "norms.hpp"
#include "triangular_solve.hpp"
#include "vector.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

using linalg::Matrix;
using linalg::Vector;

// ---------------------------------------------------------------------------
// Local no-pivot LU for comparison only.
// This is intentionally naive — it is here to show what breaks without pivoting.
// ---------------------------------------------------------------------------

struct NoPivotLU {
    Matrix L;
    Matrix U;
    bool failed = false;   // true if a zero pivot was encountered
    std::size_t fail_step = 0;
};

NoPivotLU lu_no_pivot(const Matrix& A, double tol = 1e-14) {
    const std::size_t n = A.rows();
    Matrix work = A;
    Matrix L = Matrix::zeros(n, n);
    for (std::size_t i = 0; i < n; ++i) L(i, i) = 1.0;
    Matrix U = Matrix::zeros(n, n);

    for (std::size_t k = 0; k < n; ++k) {
        if (std::abs(work(k, k)) <= tol) {
            return NoPivotLU{std::move(L), std::move(U), true, k};
        }
        for (std::size_t j = k; j < n; ++j) U(k, j) = work(k, j);
        for (std::size_t i = k + 1; i < n; ++i) {
            L(i, k) = work(i, k) / work(k, k);
            for (std::size_t j = k + 1; j < n; ++j) {
                work(i, j) -= L(i, k) * work(k, j);
            }
        }
    }
    return NoPivotLU{std::move(L), std::move(U), false, 0};
}

// Solve using a no-pivot LU (L unit lower triangular, U upper triangular).
// If the factorization failed or U is numerically singular, returns nullopt.
std::optional<Vector> solve_no_pivot(const NoPivotLU& f, const Vector& b) {
    if (f.failed) return std::nullopt;
    try {
        const Vector y = linalg::forward_substitution(f.L, b, 1e-14, /*unit_diagonal=*/true);
        return linalg::backward_substitution(f.U, y);
    } catch (...) {
        return std::nullopt;
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

double solve_residual(const Matrix& A, const Vector& x, const Vector& b) {
    return linalg::norm2(A * x - b);
}

double reconstruction_error(const Matrix& A, const linalg::LUResult& lu) {
    const std::size_t n = A.rows();
    Matrix PA(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            PA(i, j) = A(lu.perm[i], j);
    const Matrix LU_prod = lu.L * lu.U;
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double d = PA(i, j) - LU_prod(i, j);
            err += d * d;
        }
    return std::sqrt(err);
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << std::left
              << std::setw(22) << "Method"
              << std::setw(20) << "||Ax - b||"
              << std::setw(20) << "||PA - LU||"
              << "\n";
    std::cout << std::string(60, '-') << "\n";
}

void report_pivoted(const Matrix& A, const Vector& b) {
    try {
        const linalg::LUResult lu = linalg::lu_factor(A);
        const Vector x = linalg::lu_solve(lu, b);
        std::cout << std::left << std::setw(22) << "Pivoted LU"
                  << std::setw(20) << std::scientific << std::setprecision(3)
                  << solve_residual(A, x, b)
                  << std::setw(20) << reconstruction_error(A, lu)
                  << "\n";
    } catch (const std::exception& e) {
        std::cout << std::left << std::setw(22) << "Pivoted LU"
                  << "FAILED: " << e.what() << "\n";
    }
}

void report_no_pivot(const Matrix& A, const Vector& b) {
    const NoPivotLU f = lu_no_pivot(A);
    if (f.failed) {
        std::cout << std::left << std::setw(22) << "No-pivot LU"
                  << "FAILED at step " << f.fail_step << " (zero pivot)\n";
        return;
    }
    const auto x_opt = solve_no_pivot(f, b);
    if (!x_opt) {
        std::cout << std::left << std::setw(22) << "No-pivot LU"
                  << "FAILED during solve (singular U)\n";
        return;
    }
    // Compute reconstruction error without perm (no-pivot uses A directly).
    const Matrix LU_prod = f.L * f.U;
    double rec_err = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) {
            const double d = A(i, j) - LU_prod(i, j);
            rec_err += d * d;
        }
    rec_err = std::sqrt(rec_err);

    std::cout << std::left << std::setw(22) << "No-pivot LU"
              << std::setw(20) << std::scientific << std::setprecision(3)
              << solve_residual(A, *x_opt, b)
              << std::setw(20) << rec_err
              << "\n";
}

void run_case(const std::string& label, const Matrix& A, const Vector& b) {
    print_header(label);
    report_pivoted(A, b);
    report_no_pivot(A, b);
}

// ---------------------------------------------------------------------------
// Experiment cases
// ---------------------------------------------------------------------------

// 1. Random well-conditioned matrix
void exp_random(std::size_t n = 8) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    Matrix A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = dist(rng);

    Vector b(n);
    for (std::size_t i = 0; i < n; ++i) b[i] = dist(rng);

    run_case("Random 8x8 (well-conditioned)", A, b);
}

// 2. Badly row-scaled matrix
//    Rows differ in magnitude by ~10^14.  Without pivoting, tiny early pivots
//    amplify round-off; with pivoting, the large-row is selected first.
void exp_badly_scaled() {
    const Matrix A{
        {1e-14, 1.0,   2.0  },
        {1.0,   3.0,   4.0  },
        {2.0,   5.0,   7.0  }
    };
    const Vector b{1e-14 + 3.0, 8.0, 14.0};  // b = A * [1, 1, 1]
    run_case("Badly scaled (row norms differ by 10^14)", A, b);
}

// 3. Classic pathological example for no-pivot LU.
//    With epsilon = 1e-15, no-pivot computes a huge multiplier (1/epsilon),
//    which causes catastrophic cancellation in the updated rows.
//    With pivoting, we swap first and the multiplier is bounded by 1.
void exp_epsilon_pathology() {
    constexpr double eps = 1e-15;
    const Matrix A{{eps, 1.0}, {1.0, 2.0}};
    //  True solution of [eps 1; 1 2] * x = [1+eps; 3] is x = [1; 1].
    const Vector b{1.0 + eps, 3.0};
    run_case("Epsilon pathology [[1e-15,1],[1,2]] (classic)", A, b);
    std::cout << "  Note: exact solution is x = [1, 1]\n";
}

// 4. Matrix where no-pivot LU diverges visibly on a 4x4 example.
//    The first pivot is small (0.001) but rows below have entries ~1000.
//    No pivot causes multipliers of magnitude 10^6, annihilating subdiagonal info.
void exp_amplified_multiplier() {
    const Matrix A{
        {0.001, 1.0,   0.0,   0.0  },
        {1.0,   2.0,   1.0,   0.0  },
        {0.0,   1.0,   3.0,   1.0  },
        {0.0,   0.0,   1.0,   4.0  }
    };
    const Vector b = A * Vector{1.0, 2.0, 3.0, 4.0};
    run_case("Amplified multiplier (small (1,1) pivot, 4x4)", A, b);
    std::cout << "  Note: exact solution is x = [1, 2, 3, 4]\n";
}

// 5. Matrix requiring multiple row swaps (permutation is non-trivial).
void exp_permutation() {
    const Matrix A{
        {0.0, 0.0, 3.0},
        {0.0, 2.0, 1.0},
        {5.0, 1.0, 0.0}
    };
    const Vector b = A * Vector{1.0, -1.0, 2.0};
    run_case("Multiple row swaps required (zeros in pivot positions)", A, b);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << std::string(60, '*') << "\n";
    std::cout << "  Pivoting vs No-Pivoting LU Experiment\n";
    std::cout << std::string(60, '*') << "\n";
    std::cout << "Residual  = ||Ax - b||_2   (solve accuracy)\n";
    std::cout << "Recon err = ||PA - LU||_F  (factorization accuracy)\n";

    exp_random();
    exp_badly_scaled();
    exp_epsilon_pathology();
    exp_amplified_multiplier();
    exp_permutation();

    return 0;
}
