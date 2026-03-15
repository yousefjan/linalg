#include "qr_iteration.hpp"

#include <cassert>
#include <cmath>
#include <sstream>

#include "linalg_error.hpp"
#include "matrix.hpp"
#include "qr.hpp"
#include "vector.hpp"

// References used throughout this file:
//   T&B  — Trefethen & Bau, "Numerical Linear Algebra"
//   GVL  — Golub & Van Loan, "Matrix Computations" 4th ed.

namespace linalg {

namespace {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Frobenius norm of the strict lower triangle of an n×n matrix.
// This is the standard convergence diagnostic for QR iteration: as A_k
// approaches the real Schur form, all entries below the main diagonal
// (excluding 2×2 block sub-diagonals) tend to zero.
//
// ||lower(A)||_F = sqrt( sum_{i > j} A(i,j)^2 )
//
// Ref: T&B §28; used as the convergence criterion in Algorithm 28.1.
double lower_triangle_norm(const Matrix& A) {
    const std::size_t n = A.rows();
    double s = 0.0;
    for (std::size_t i = 1; i < n; ++i)      // row 1 .. n-1
        for (std::size_t j = 0; j < i; ++j)  // col 0 .. i-1  (strict lower)
            s += A(i, j) * A(i, j);
    return std::sqrt(s);
}

// Extract eigenvalues from a quasi-upper-triangular matrix (real Schur form).
//
// Scans the diagonal from top-left to bottom-right.  At each position i:
//   — |A(i+1, i)| < tol  → 1×1 block: real eigenvalue A(i,i), imag = 0.
//   — otherwise           → 2×2 block [A(i..i+1, i..i+1)]: eigenvalues via
//                           quadratic formula.  When the discriminant is
//                           negative the result is a complex-conjugate pair,
//                           stored as (re, +im) and (re, -im) in the real
//                           and imaginary part Vectors.
//
// Fills positions 0..n-1 of `real_out` and `imag_out` (pre-sized to n).
//
// Ref: T&B Lecture 28; GVL §7.4.1.
void extract_eigenvalues(const Matrix& T, double tol,
                         Vector& real_out, Vector& imag_out) {
    const std::size_t n = T.rows();
    std::size_t out = 0;  // next write position in real_out / imag_out
    std::size_t i   = 0;  // current scan position in T

    while (i < n) {
        const bool is_last   = (i + 1 == n);
        const bool sub_small = is_last || (std::abs(T(i + 1, i)) < tol);

        if (sub_small) {
            // 1×1 block: real eigenvalue.
            real_out[out] = T(i, i);
            imag_out[out] = 0.0;
            ++out;
            ++i;
        } else {
            // 2×2 block:
            //   | a  b |
            //   | c  d |
            // Characteristic polynomial: lambda^2 - (a+d)*lambda + (ad - bc) = 0.
            // Discriminant: (a-d)^2 + 4*b*c.
            // Ref: GVL §7.4.1.
            const double a    = T(i,     i);
            const double b    = T(i,     i + 1);
            const double c    = T(i + 1, i);
            const double d    = T(i + 1, i + 1);
            const double tr   = a + d;
            const double disc = (a - d) * (a - d) + 4.0 * b * c;

            if (disc >= 0.0) {
                // Real eigenvalues — unusual in converged real Schur form, but
                // handled robustly in case the block didn't fully split.
                const double sq   = std::sqrt(disc);
                real_out[out]     = 0.5 * (tr + sq);
                imag_out[out]     = 0.0;
                real_out[out + 1] = 0.5 * (tr - sq);
                imag_out[out + 1] = 0.0;
            } else {
                // Complex-conjugate pair: real part ± imaginary part.
                const double re       = 0.5 * tr;
                const double im       = 0.5 * std::sqrt(-disc);
                real_out[out]         = re;
                imag_out[out]         =  im;
                real_out[out + 1]     = re;
                imag_out[out + 1]     = -im;
            }
            out += 2;
            i   += 2;
        }
    }

    assert(out == n);
}

// Verify that A is square; throw DimensionMismatchError otherwise.
void require_square(const Matrix& A, const char* fname) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << fname << ": requires a square matrix, got "
            << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Stage 1: Unshifted QR iteration
// ---------------------------------------------------------------------------
//
// Each step performs an orthogonal similarity transformation:
//   A_{k-1} = Q_k R_k   (Householder QR; backward-stable)
//   A_k     = R_k Q_k   = Q_k^T A_{k-1} Q_k
//
// Similarity preserves eigenvalues (GVL §7.3.1, Theorem 7.3.1).
// The iterates converge to the real Schur form: a quasi-upper-triangular
// matrix whose 1×1 blocks give real eigenvalues and 2×2 blocks give
// complex-conjugate pairs.
//
// Convergence rate: linear, with per-step reduction factor
//   |lambda_{j+1} / lambda_j|  for the (j, j+1) coupling.
// (T&B Lecture 28, Theorem 28.2; GVL §7.3.2)
//
// Each iteration costs O(n^3) due to full Householder QR; Hessenberg
// reduction (Stage 3) reduces subsequent steps to O(n^2).

QRIterationResult eigenvalues_unshifted(const Matrix& A,
                                        QRIterationOptions opts) {
    require_square(A, "eigenvalues_unshifted");
    const std::size_t n = A.rows();

    // Threshold for classifying a sub-diagonal entry as "zero" when reading
    // eigenvalues out of the converged Schur form.  Using the same value as
    // the convergence tolerance is appropriate; we only reach extraction once
    // ||lower(A_k)||_F < opts.tolerance.  Ref: GVL §7.4.1.
    const double extract_tol = opts.tolerance;

    QRIterationResult result;
    // Pre-size eigenvalue Vectors; they are always length n.
    result.eigenvalues_real = Vector(n, 0.0);
    result.eigenvalues_imag = Vector(n, 0.0);

    if (opts.track_convergence) {
        result.convergence_history.reserve(
            static_cast<std::size_t>(opts.max_iterations));
    }

    // Handle the trivial 1×1 case immediately.
    if (n == 1) {
        result.eigenvalues_real[0] = A(0, 0);
        return result;
    }

    // Working copy; becomes the quasi-upper-triangular Schur form A_k.
    Matrix Ak = A;

    for (int k = 0; k < opts.max_iterations; ++k) {
        // --- QR step ---
        // Factor A_{k-1} = Q R using backward-stable Householder reflections.
        const QRResult qr = qr_householder(Ak);

        // A_k = R Q  (orthogonal similarity: Q^T A_{k-1} Q)
        Ak = qr.R * qr.Q;

        // --- Convergence check ---
        const double lower_norm = lower_triangle_norm(Ak);

        if (opts.track_convergence) {
            result.convergence_history.push_back(lower_norm);
        }
        ++result.iterations;

        if (lower_norm < opts.tolerance) {
            extract_eigenvalues(Ak, extract_tol,
                                result.eigenvalues_real,
                                result.eigenvalues_imag);
            return result;
        }
    }

    // Maximum iterations reached without convergence — fail loudly.
    // Possible causes: eigenvalues too close in magnitude, or complex
    // eigenvalue pairs that require a double shift (see Stage 2).
    std::ostringstream oss;
    oss << "eigenvalues_unshifted: did not converge in "
        << opts.max_iterations << " iterations "
        << "(final ||lower(A_k)||_F = " << lower_triangle_norm(Ak)
        << ", tolerance = " << opts.tolerance << "). "
        << "Try eigenvalues_shifted (Stage 2) or increase max_iterations.";
    throw NonConvergenceError(oss.str());
}

}  // namespace linalg
