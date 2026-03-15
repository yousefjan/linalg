#pragma once

// QR iteration for eigenvalue computation
//
// Refs:
//   Trefethen & Bau, "Numerical Linear Algebra" (T&B)
//     Lecture 25 — Eigenvalue algorithms
//     Lecture 26 — Schur factorisation
//     Lecture 28 — The QR algorithm (unshifted)
//     Lecture 29 — The QR algorithm with shifts
//   Golub & Van Loan, "Matrix Computations" 4th ed. (GVL)
//     §7.3  — The Unshifted QR Algorithm
//     §7.4  — The Shifted QR Algorithm
//     §7.4.2 — Wilkinson shift

#include <vector>

#include "linalg_error.hpp"
#include "matrix.hpp"
#include "vector.hpp"

namespace linalg {

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

// All defaults are consistent with the recommendations in T&B Lecture 28.
struct QRIterationOptions {
    // Convergence threshold.  Iteration halts once the Frobenius norm of the
    // strict lower triangle of A_k falls below this value.
    // Ref: T&B §28; GVL §7.3.
    double tolerance = 1e-10;

    int max_iterations = 1000;

    // When true, the Frobenius norm of the strict lower triangle is recorded
    // after every QR step and returned in QRIterationResult::convergence_history.
    bool track_convergence = false;
};


struct QRIterationResult {
    // Real and imaginary parts of the n eigenvalues.
    // For symmetric inputs all imaginary parts are zero.
    // Complex-conjugate pairs from 2×2 Schur blocks appear as ±imag entries.
    // Both vectors always have length n (the matrix dimension).
    Vector eigenvalues_real;
    Vector eigenvalues_imag;

    // Total number of QR steps performed before convergence or max_iterations.
    int iterations = 0;

    // Populated only when QRIterationOptions::track_convergence is true.
    // Entry k is ||lower(A_k)||_F after the k-th QR step.
    // std::vector is used here because linalg::Vector has no push_back;
    // convergence_history is a plain time-series container, not a math object.
    std::vector<double> convergence_history;
};

// Algorithm (T&B Algorithm 28.1):
//
//   A_0 = A
//   for k = 1, 2, ...:
//       factor  A_{k-1} = Q_k R_k   (Householder QR)
//       set     A_k     = R_k Q_k   (orthogonal similarity: preserves eigenvalues)
//
// The iterates A_k converge to the real Schur form of A: a quasi-upper-
// triangular matrix with 1×1 blocks (real eigenvalue) and 2×2 blocks
// (complex-conjugate pair) on the diagonal.
//
// Convergence rate: linear.  Per-step factor ≈ |lambda_{j+1} / lambda_j|
// for the off-diagonal entries linking eigenvalue clusters j and j+1.
// (T&B Lecture 28, Theorem 28.2)
//
// Throws DimensionMismatchError  if A is not square.
// Throws NonConvergenceError     if convergence is not achieved within
//                                opts.max_iterations steps.
[[nodiscard]] QRIterationResult eigenvalues_unshifted(const Matrix& A,
                                                      QRIterationOptions opts = {});

}  // namespace linalg
