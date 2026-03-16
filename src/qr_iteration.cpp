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


// Frobenius norm of the strict lower triangle of an n×n matrix.
// This is the standard convergence diagnostic for QR iteration: as A_k
// approaches the real Schur form, all entries below the main diagonal
// (excluding 2×2 block sub-diagonals) tend to zero.
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
    std::size_t out = 0;  
    std::size_t i   = 0; 

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
            const double a = T(i, i);
            const double b = T(i, i + 1);
            const double c = T(i + 1, i);
            const double d = T(i + 1, i + 1);
            const double tr = a + d;
            const double disc = (a - d) * (a - d) + 4.0 * b * c;

            if (disc >= 0.0) {
                // Real eigenvalues unusual in converged real Schur form, but
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

void require_square(const Matrix& A, const char* fname) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << fname << ": requires a square matrix, got "
            << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }
}

}  // namespace

// --- Unshifted QR iteration ---
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

    const double extract_tol = opts.tolerance;

    QRIterationResult result;
    result.eigenvalues_real = Vector(n, 0.0);
    result.eigenvalues_imag = Vector(n, 0.0);

    if (opts.track_convergence) {
        result.convergence_history.reserve(
            static_cast<std::size_t>(opts.max_iterations));
    }

    if (n == 1) {
        result.eigenvalues_real[0] = A(0, 0);
        return result;
    }

    Matrix Ak = A;

    for (int k = 0; k < opts.max_iterations; ++k) {
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

    std::ostringstream oss;
    oss << "eigenvalues_unshifted: did not converge in "
        << opts.max_iterations << " iterations "
        << "(final ||lower(A_k)||_F = " << lower_triangle_norm(Ak)
        << ", tolerance = " << opts.tolerance << "). "
        << "Try eigenvalues_shifted (Stage 2) or increase max_iterations.";
    throw NonConvergenceError(oss.str());
}

// --- Wilkinson-shifted QR iteration ---
//
// The Wilkinson shift is the eigenvalue of the bottom-right 2×2 block
//   | a  b |
//   | c  d |
// that is closest to d (the trailing diagonal entry).
//
// Exact eigenvalue formula: μ_{1,2} = (a+d)/2 ± sqrt(((a-d)/2)² + b·c)
// We pick the one with |μ - d| smaller.
//
// When the discriminant is negative (complex eigenvalues), fall back to σ = d
// (Rayleigh quotient shift), which still accelerates convergence.
//
// Ref: T&B Lecture 29; GVL §7.4.2.

namespace {

double wilkinson_shift(const Matrix& A) {
    const std::size_t n = A.rows();
    const double a = A(n - 2, n - 2);
    const double b = A(n - 1, n - 2);  // subdiagonal entry only
    const double d = A(n - 1, n - 1);
    const double delta = 0.5 * (a - d);
    const double denom = std::abs(delta) + std::hypot(delta, b);
    if (denom == 0.0) return d;
    const double sgn = (delta >= 0.0) ? 1.0 : -1.0;
    return d - sgn * (b * b) / denom;
}

}  // namespace

// eigenvalues_shifted — Wilkinson-shifted QR with trailing deflation.
//
// After each QR step we check whether the trailing subdiagonal entry of the
// active block is negligible (relative criterion: GVL §7.4.1).  If so, the
// bottom diagonal entry is accepted as a converged eigenvalue and the active
// subproblem shrinks by one.  This "trailing deflation" enables the cubic
// convergence promised by the Wilkinson shift to compound across successive
// eigenvalues rather than stalling on the full lower-triangle norm.
//
// When the active size reaches 2 we extract both eigenvalues analytically
// from the 2×2 block (handling real and complex-conjugate pairs) rather than
// continuing to iterate.  For symmetric inputs this is always a real pair.
//
// Ref: GVL §7.5.1; T&B Lecture 29.

QRIterationResult eigenvalues_shifted(const Matrix& A, QRIterationOptions opts) {
    require_square(A, "eigenvalues_shifted");
    const std::size_t n = A.rows();

    QRIterationResult result;
    result.eigenvalues_real = Vector(n, 0.0);
    result.eigenvalues_imag = Vector(n, 0.0);

    if (opts.track_convergence)
        result.convergence_history.reserve(
            static_cast<std::size_t>(opts.max_iterations));

    if (n == 1) {
        result.eigenvalues_real[0] = A(0, 0);
        return result;
    }

    Matrix Ak = A;

    std::size_t n_found = n;
    std::size_t active  = n;  // live subproblem is rows/cols 0..active-1

    auto store_real = [&](double re) {
        --n_found;
        result.eigenvalues_real[n_found] = re;
        result.eigenvalues_imag[n_found] = 0.0;
    };

    auto store_pair = [&](double re, double im) {
        --n_found; result.eigenvalues_real[n_found] = re; result.eigenvalues_imag[n_found] =  im;
        --n_found; result.eigenvalues_real[n_found] = re; result.eigenvalues_imag[n_found] = -im;
    };

    auto close_2x2 = [&]() {
        const double a    = Ak(active - 2, active - 2);
        const double b    = Ak(active - 2, active - 1);
        const double c    = Ak(active - 1, active - 2);
        const double d    = Ak(active - 1, active - 1);
        const double tr   = a + d;
        const double disc = (a - d) * (a - d) + 4.0 * b * c;
        if (disc >= 0.0) {
            const double sq = std::sqrt(disc);
            store_real(0.5 * (tr + sq));
            store_real(0.5 * (tr - sq));
        } else {
            store_pair(0.5 * tr, 0.5 * std::sqrt(-disc));
        }
        active -= 2;
    };

    for (int k = 0; k < opts.max_iterations; ++k) {
        // --- Deflation sweep ---
        while (active >= 2) {
            const double sub   = std::abs(Ak(active - 1, active - 2));
            const double scale = std::abs(Ak(active - 2, active - 2))
                               + std::abs(Ak(active - 1, active - 1));
            // Relative + absolute floor tolerance (GVL §7.4.1).
            const double deflation_tol =
                opts.tolerance * (scale > 0.0 ? scale : 1.0);
            if (sub > deflation_tol) break;
            Ak(active - 1, active - 2) = 0.0;  // enforce exact zero
            store_real(Ak(active - 1, active - 1));
            --active;
        }

        if (active == 0) break;
        if (active == 1) { store_real(Ak(0, 0)); active = 0; break; }
        if (active == 2) { close_2x2();                       break; }

        // --- Wilkinson-shifted QR step on the active × active subblock ---
        Matrix sub_mat(active, active);
        for (std::size_t i = 0; i < active; ++i)
            for (std::size_t j = 0; j < active; ++j)
                sub_mat(i, j) = Ak(i, j);

        const double sigma = wilkinson_shift(sub_mat);

        for (std::size_t i = 0; i < active; ++i) sub_mat(i, i) -= sigma;
        const QRResult qr = qr_householder(sub_mat);
        sub_mat = qr.R * qr.Q;
        for (std::size_t i = 0; i < active; ++i) sub_mat(i, i) += sigma;

        for (std::size_t i = 0; i < active; ++i)
            for (std::size_t j = 0; j < active; ++j)
                Ak(i, j) = sub_mat(i, j);

        const double lower_norm = lower_triangle_norm(Ak);
        if (opts.track_convergence)
            result.convergence_history.push_back(lower_norm);
        ++result.iterations;
    }

    if (n_found > 0) {
        std::ostringstream oss;
        oss << "eigenvalues_shifted: did not converge in " << opts.max_iterations
            << " iterations (" << n_found << " eigenvalue(s) not yet deflated).";
        throw NonConvergenceError(oss.str());
    }
    return result;
}

// --- Givens rotation ---

GivensRotation GivensRotation::make(double x, double y, std::size_t row_index) {
    const double r = std::hypot(x, y);
    if (r == 0.0) return {1.0, 0.0, row_index};
    return {x / r, y / r, row_index};
}

void GivensRotation::apply_left(Matrix& M, std::size_t col_start) const {
    // Rows i and i+1, columns col_start..n-1.
    // [ c  s] [x]   [cx + sy]
    // [-s  c] [y] = [-sx + cy]
    for (std::size_t j = col_start; j < M.cols(); ++j) {
        const double xi  = M(i,     j);
        const double xi1 = M(i + 1, j);
        M(i,     j) =  c * xi + s * xi1;
        M(i + 1, j) = -s * xi + c * xi1;
    }
}

void GivensRotation::apply_right(Matrix& M, std::size_t row_end) const {
    // Columns i and i+1, rows 0..row_end-1.
    // M * G^T where G^T = [c -s; s c]:
    // new col i   = c * old_i + s * old_{i+1}
    // new col i+1 = -s * old_i + c * old_{i+1}
    for (std::size_t j = 0; j < row_end; ++j) {
        const double xi  = M(j, i);
        const double xi1 = M(j, i + 1);
        M(j, i)     =  c * xi + s * xi1;
        M(j, i + 1) = -s * xi + c * xi1;
    }
}

// --- Hessenberg reduction ---
// For k = 0, 1, ..., n-3:
//   Build a Householder reflector H_k that zeros A[k+2:n, k].
//   Apply from left:  A[k+1:n, k:n] ← H_k * A[k+1:n, k:n]
//   Apply from right: A[0:n, k+1:n] ← A[0:n, k+1:n] * H_k
//   Accumulate Q:     Q[0:n, k+1:n] ← Q[0:n, k+1:n] * H_k
//
// H_k is never formed explicitly; applied via rank-1 update with tau = 2/uᵀu.
// Ref: GVL §7.4.2 (Algorithm 7.4.2).

HessenbergResult hessenberg_reduction(const Matrix& A) {
    require_square(A, "hessenberg_reduction");
    const std::size_t n = A.rows();

    Matrix H = A;
    Matrix Q = Matrix::identity(n);

    for (std::size_t k = 0; k + 2 <= n; ++k) {
        const std::size_t p = n - k - 1;  // p = n - (k+1)
        if (p == 0) break;

        // Build Householder vector u from H[k+1:n, k].
        std::vector<double> u(p);
        for (std::size_t i = 0; i < p; ++i) u[i] = H(k + 1 + i, k);

        double x_norm = 0.0;
        for (double v : u) x_norm += v * v;
        x_norm = std::sqrt(x_norm);

        if (x_norm == 0.0) continue;

        const double sigma = (u[0] >= 0.0 ? 1.0 : -1.0) * x_norm;
        u[0] += sigma;

        double utu = 0.0;
        for (double v : u) utu += v * v;
        const double tau = 2.0 / utu;

        // Apply H_k from the LEFT to H[k+1:n, k:n].
        for (std::size_t j = k; j < n; ++j) {
            double dot = 0.0;
            for (std::size_t i = 0; i < p; ++i) dot += u[i] * H(k + 1 + i, j);
            const double coeff = tau * dot;
            for (std::size_t i = 0; i < p; ++i) H(k + 1 + i, j) -= coeff * u[i];
        }

        // Apply H_k from the RIGHT to H[0:n, k+1:n].
        for (std::size_t j = 0; j < n; ++j) {
            double dot = 0.0;
            for (std::size_t i = 0; i < p; ++i) dot += H(j, k + 1 + i) * u[i];
            const double coeff = tau * dot;
            for (std::size_t i = 0; i < p; ++i) H(j, k + 1 + i) -= coeff * u[i];
        }

        // Accumulate Q: Q[0:n, k+1:n] ← Q[0:n, k+1:n] * H_k.
        for (std::size_t j = 0; j < n; ++j) {
            double dot = 0.0;
            for (std::size_t i = 0; i < p; ++i) dot += Q(j, k + 1 + i) * u[i];
            const double coeff = tau * dot;
            for (std::size_t i = 0; i < p; ++i) Q(j, k + 1 + i) -= coeff * u[i];
        }

        for (std::size_t i = 1; i < p; ++i) H(k + 1 + i, k) = 0.0;
    }

    return HessenbergResult{std::move(H), std::move(Q)};
}

// --- Hessenberg QR step via Givens rotations ---
//
// One shifted QR step on the upper Hessenberg matrix H:
//   1. Shift: H ← H - σI.
//   2. For k = 0..n-2: compute G_k = Givens(H(k,k), H(k+1,k));
//                        apply G_k from left to rows k,k+1 of H,
//                        starting from column k (Hessenberg: H(k+1,j)=0, j<k).
//   3. For k = 0..n-2: apply G_k^T from right to cols k,k+1 of H,
//                        up to row k+2 (exploits upper-triangular structure).
//   4. Unshift: H ← H + σI.
//
// After the step H is again upper Hessenberg (GVL §7.4.2, Theorem 7.4.1).
// Total cost: O(n²).  Ref: GVL §7.4.2.

void hessenberg_qr_step(Matrix& H, double sigma) {
    const std::size_t n = H.rows();

    for (std::size_t j = 0; j < n; ++j) H(j, j) -= sigma;

    std::vector<GivensRotation> gs;
    gs.reserve(n - 1);

    for (std::size_t k = 0; k + 1 < n; ++k) {
        // Eliminate H(k+1, k) via a rotation on rows k and k+1.
        GivensRotation g = GivensRotation::make(H(k, k), H(k + 1, k), k);
        // Left application: rows k, k+1; columns k..n-1.
        // (Hessenberg: H(k+1, j) = 0 for j < k, so starting from col k is exact.)
        g.apply_left(H, k);
        gs.push_back(g);
    }

    for (std::size_t k = 0; k + 1 < n; ++k) {
        gs[k].apply_right(H, std::min(k + 2, n));
    }

    // Unshift.
    for (std::size_t j = 0; j < n; ++j) H(j, j) += sigma;
}

// --- Full QR algorithm ---
//
// Same outer deflation loop as eigenvalues_shifted, but each QR step uses
// hessenberg_qr_step (O(n²) Givens rotations) instead of full Householder QR
// (O(n³)).  After Hessenberg reduction the matrix stays Hessenberg throughout,
// so the O(n²) per-step cost applies for every step after the one-time O(n³)
// reduction.  Total cost is thus O(n³) + O(iterations · n²), which beats
// eigenvalues_shifted's O(iterations · n³) for large n.
//
// Ref: GVL §7.4.2; T&B Lecture 29.

QRIterationResult eigenvalues_hessenberg(const Matrix& A,
                                         QRIterationOptions opts) {
    require_square(A, "eigenvalues_hessenberg");
    const std::size_t n = A.rows();

    QRIterationResult result;
    result.eigenvalues_real = Vector(n, 0.0);
    result.eigenvalues_imag = Vector(n, 0.0);

    if (opts.track_convergence)
        result.convergence_history.reserve(
            static_cast<std::size_t>(opts.max_iterations));

    if (n == 1) {
        result.eigenvalues_real[0] = A(0, 0);
        return result;
    }

    HessenbergResult hr = hessenberg_reduction(A);
    Matrix& H = hr.H;

    std::size_t n_found = n;
    std::size_t active  = n;

    auto store_real = [&](double re) {
        --n_found;
        result.eigenvalues_real[n_found] = re;
        result.eigenvalues_imag[n_found] = 0.0;
    };

    auto store_pair = [&](double re, double im) {
        --n_found; result.eigenvalues_real[n_found] = re; result.eigenvalues_imag[n_found] =  im;
        --n_found; result.eigenvalues_real[n_found] = re; result.eigenvalues_imag[n_found] = -im;
    };

    auto close_2x2 = [&]() {
        const double a    = H(active - 2, active - 2);
        const double b    = H(active - 2, active - 1);
        const double c    = H(active - 1, active - 2);
        const double d    = H(active - 1, active - 1);
        const double tr   = a + d;
        const double disc = (a - d) * (a - d) + 4.0 * b * c;
        if (disc >= 0.0) {
            const double sq = std::sqrt(disc);
            store_real(0.5 * (tr + sq));
            store_real(0.5 * (tr - sq));
        } else {
            store_pair(0.5 * tr, 0.5 * std::sqrt(-disc));
        }
        active -= 2;
    };

    for (int k = 0; k < opts.max_iterations; ++k) {
        // --- Deflation sweep ---
        while (active >= 2) {
            const double sub   = std::abs(H(active - 1, active - 2));
            const double scale = std::abs(H(active - 2, active - 2))
                               + std::abs(H(active - 1, active - 1));
            const double deflation_tol =
                opts.tolerance * (scale > 0.0 ? scale : 1.0);
            if (sub > deflation_tol) break;
            H(active - 1, active - 2) = 0.0;
            store_real(H(active - 1, active - 1));
            --active;
        }

        if (active == 0) break;
        if (active == 1) { store_real(H(0, 0)); active = 0; break; }
        if (active == 2) { close_2x2();                       break; }

        // Wilkinson shift from trailing 2×2 of the active block.
        const double a_w = H(active - 2, active - 2);
        const double b_w = H(active - 1, active - 2);
        const double d_w = H(active - 1, active - 1);
        const double delta = 0.5 * (a_w - d_w);
        const double denom = std::abs(delta) + std::hypot(delta, b_w);
        const double sigma = (denom == 0.0) ? d_w
            : d_w - ((delta >= 0.0) ? 1.0 : -1.0) * (b_w * b_w) / denom;

        // O(n²) Givens step on the active×active Hessenberg subblock.
        Matrix sub_H(active, active);
        for (std::size_t ii = 0; ii < active; ++ii)
            for (std::size_t jj = 0; jj < active; ++jj)
                sub_H(ii, jj) = H(ii, jj);

        hessenberg_qr_step(sub_H, sigma);

        for (std::size_t ii = 0; ii < active; ++ii)
            for (std::size_t jj = 0; jj < active; ++jj)
                H(ii, jj) = sub_H(ii, jj);

        if (opts.track_convergence) {
            double s = 0.0;
            for (std::size_t ii = 1; ii < active; ++ii)
                for (std::size_t jj = 0; jj < ii; ++jj)
                    s += H(ii, jj) * H(ii, jj);
            result.convergence_history.push_back(std::sqrt(s));
        }
        ++result.iterations;
    }

    if (n_found > 0) {
        std::ostringstream oss;
        oss << "eigenvalues_hessenberg: did not converge in "
            << opts.max_iterations << " iterations ("
            << n_found << " eigenvalue(s) not yet deflated).";
        throw NonConvergenceError(oss.str());
    }
    return result;
}

}  // namespace linalg
