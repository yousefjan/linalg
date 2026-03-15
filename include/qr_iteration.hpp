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
    // For symmetric inputs all imaginary parts are zero.
    // Complex-conjugate pairs from 2×2 Schur blocks appear as +/-imag entries.
    Vector eigenvalues_real;
    Vector eigenvalues_imag;

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
// Convergence rate: linear.  Per-step factor ~= |lambda_{j+1} / lambda_j|
// for the off-diagonal entries linking eigenvalue clusters j and j+1.
// (T&B Lecture 28, Theorem 28.2)
//
// Throws DimensionMismatchError  if A is not square.
// Throws NonConvergenceError     if convergence is not achieved within
//                                opts.max_iterations steps.
[[nodiscard]] QRIterationResult eigenvalues_unshifted(const Matrix& A,
                                                      QRIterationOptions opts = {});

// ---------------------------------------------------------------------------
// Wilkinson-shifted QR iteration
// ---------------------------------------------------------------------------
//
// Same outer loop as Stage 1, but each step applies a shift σ chosen as
// the eigenvalue of the bottom-right 2×2 block of A_{k-1} that is closest
// to the (n,n) entry, then unshifts after the QR step:
//
//   factor  (A_{k-1} - σI) = Q_k R_k
//   set     A_k = R_k Q_k + σI
//
// The Wilkinson shift (T&B Lecture 29; GVL §7.4.2):
//   Given the bottom-right 2×2 block | a  b |
//                                     | b  c |
//   δ = (a - c) / 2
//   σ = c - sign(δ) * b² / (|δ| + sqrt(δ² + b²))
//   equivalently: the eigenvalue of the block closer to c.
//
// Convergence rate: typically cubic near a simple eigenvalue.
// (T&B Lecture 29; GVL §7.5.1)
//
// Same exceptions as eigenvalues_unshifted.
[[nodiscard]] QRIterationResult eigenvalues_shifted(const Matrix& A,
                                                    QRIterationOptions opts = {});

// ---------------------------------------------------------------------------
// Stage 3: Hessenberg reduction algorithm
// ---------------------------------------------------------------------------

// Givens rotation G acting on rows/columns i and i+1:
//
//   G = | c   s |  chosen so that G * [x; y]^T = [r; 0]^T
//       | -s  c |  with c = x/r, s = y/r, r = hypot(x, y)
//
// T&B Lecture 10 (Givens rotations).
struct GivensRotation {
    double      c;  // cos(theta)
    double      s;  // sin(theta)
    std::size_t i;  // first row/column index (second is i+1)

    // Construct the rotation that maps [x, y]^T → [hypot(x,y), 0]^T.
    // Returns the identity (c=1, s=0) when x == y == 0.
    [[nodiscard]] static GivensRotation make(double x, double y,
                                             std::size_t row_index);

    // Apply G from the left to rows i and i+1 of M, columns [col_start, n).
    // M[i:i+2, col_start:] ← G * M[i:i+2, col_start:]
    void apply_left(Matrix& M, std::size_t col_start = 0) const;

    // Apply G^T from the right to columns i and i+1 of M, rows [0, row_end).
    // M[0:row_end, i:i+2] ← M[0:row_end, i:i+2] * G^T
    void apply_right(Matrix& M, std::size_t row_end) const;
};

// Result of reducing A to upper Hessenberg form.
// H is upper Hessenberg: H(i,j) = 0 for all i > j+1.
// Q is orthogonal and A = Q H Q^T.
// Ref: GVL Algorithm 7.4.2; T&B Lecture 26.
struct HessenbergResult {
    Matrix H;  // upper Hessenberg similarity of A
    Matrix Q;  // accumulated orthogonal transformation
};

// Reduce A to upper Hessenberg form via Householder reflectors applied
// from both sides.  Costs O(10n³/3) flops; done once before QR iteration.
// Ref: GVL §7.4.2 (Algorithm 7.4.2).
//
// Throws DimensionMismatchError if A is not square.
[[nodiscard]] HessenbergResult hessenberg_reduction(const Matrix& A);

// Apply one shifted QR step to an upper Hessenberg matrix H in-place,
// using n-1 Givens rotations.  Costs O(n²) vs O(n³) for Householder QR.
// H remains upper Hessenberg after the step.
// Ref: GVL §7.4.2; T&B Lecture 29.
void hessenberg_qr_step(Matrix& H, double sigma);

// Full practical QR algorithm:
//   1. Reduce A to Hessenberg H = Q^T A Q  (O(n³), done once).
//   2. Run Wilkinson-shifted QR on H using Givens steps  (O(n²) each).
// Substantially faster than eigenvalues_shifted for n ≥ 50.
// Ref: T&B Lecture 29.
//
// Same exceptions as eigenvalues_unshifted.
[[nodiscard]] QRIterationResult eigenvalues_hessenberg(const Matrix& A,
                                                       QRIterationOptions opts = {});

}  // namespace linalg
