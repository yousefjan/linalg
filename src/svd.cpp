export module linalgebra:svd;
import std;
import :error;
import :vector;
import :matrix;
import :qr_iteration;

// References used throughout this file:
//   GVL  — Golub & Van Loan, "Matrix Computations" 4th ed.
//   T&B  — Trefethen & Bau, "Numerical Linear Algebra"

export namespace linalgebra {

struct SVDResult {
    Matrix U;      // m × m orthogonal (left singular vectors)
    Vector sigma;  // min(m,n) singular values, sorted descending
    Matrix Vt;     // n × n orthogonal (V^T, right singular vectors transposed)
};

struct SVDOptions {
    double tolerance     = 1e-12;
    int    max_iterations = 1000;
};

// Golub-Kahan bidiagonalization + Golub-Reinsch QR sweeps.
// Requires rows >= cols; throws DimensionMismatchError otherwise.
// Reference: GVL §8.6; T&B Lecture 31.
[[nodiscard]] SVDResult svd(const Matrix& A, SVDOptions opts = {});

}  // namespace linalgebra

namespace linalgebra {

SVDResult svd(const Matrix& A, SVDOptions opts) {
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    if (m < n) {
        std::ostringstream oss;
        oss << "svd requires rows >= cols, got " << m << "x" << n;
        throw DimensionMismatchError(oss.str());
    }
    if (n == 0) {
        return SVDResult{Matrix::identity(m), Vector(0), Matrix::identity(0)};
    }

    // GK bidiagonalization  A = U_acc * B * Vt_acc
    // where B is m×n upper bidiagonal (nonzero on diagonal and superdiagonal).

    Matrix work  = A;
    Matrix U_acc = Matrix::identity(m);
    Matrix Vt_acc = Matrix::identity(n);

    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t p_left = m - k;  // column length below row k

        // Left Householder: zero work[k+1:, k].
        {
            std::vector<double> u(p_left);
            for (std::size_t i = 0; i < p_left; ++i) u[i] = work(k + i, k);

            double x_norm = 0.0;
            for (double v : u) x_norm += v * v;
            x_norm = std::sqrt(x_norm);

            if (x_norm > 0.0) {
                const double sigma = (u[0] >= 0.0 ? 1.0 : -1.0) * x_norm;
                u[0] += sigma;
                double utu = 0.0;
                for (double v : u) utu += v * v;
                const double tau = 2.0 / utu;

                // Apply to work from left: work[k:, k:] -= tau * u * (u^T work[k:, k:])
                for (std::size_t j = k; j < n; ++j) {
                    double d = 0.0;
                    for (std::size_t i = 0; i < p_left; ++i) d += u[i] * work(k + i, j);
                    const double c = tau * d;
                    for (std::size_t i = 0; i < p_left; ++i) work(k + i, j) -= c * u[i];
                }

                // Accumulate into U_acc from right: U_acc[:, k:] -= tau * (U_acc[:, k:] u) * u^T
                for (std::size_t j = 0; j < m; ++j) {
                    double d = 0.0;
                    for (std::size_t i = 0; i < p_left; ++i) d += U_acc(j, k + i) * u[i];
                    const double c = tau * d;
                    for (std::size_t i = 0; i < p_left; ++i) U_acc(j, k + i) -= c * u[i];
                }
            }
        }

        // Right Householder: zero work[k, k+2:].
        if (k + 2 <= n) {
            const std::size_t p_right = n - k - 1;  // row length after col k+1

            std::vector<double> v(p_right);
            for (std::size_t j = 0; j < p_right; ++j) v[j] = work(k, k + 1 + j);

            double x_norm = 0.0;
            for (double val : v) x_norm += val * val;
            x_norm = std::sqrt(x_norm);

            if (x_norm > 0.0) {
                const double sigma = (v[0] >= 0.0 ? 1.0 : -1.0) * x_norm;
                v[0] += sigma;
                double vtv = 0.0;
                for (double val : v) vtv += val * val;
                const double tau = 2.0 / vtv;

                // Apply to work from right: work[:, k+1:] -= tau * (work[:, k+1:] v) * v^T
                for (std::size_t i = k; i < m; ++i) {
                    double d = 0.0;
                    for (std::size_t j = 0; j < p_right; ++j) d += work(i, k + 1 + j) * v[j];
                    const double c = tau * d;
                    for (std::size_t j = 0; j < p_right; ++j) work(i, k + 1 + j) -= c * v[j];
                }

                // Accumulate into Vt_acc from left: Vt_acc[k+1:, :] -= tau * v * (v^T Vt_acc[k+1:, :])
                for (std::size_t j = 0; j < n; ++j) {
                    double d = 0.0;
                    for (std::size_t i = 0; i < p_right; ++i) d += v[i] * Vt_acc(k + 1 + i, j);
                    const double c = tau * d;
                    for (std::size_t i = 0; i < p_right; ++i) Vt_acc(k + 1 + i, j) -= c * v[i];
                }
            }
        }
    }

    // Extract bidiagonal: d[i] = diag, e[i] = superdiag.
    std::vector<double> d(n), e(n > 1 ? n - 1 : 0, 0.0);
    for (std::size_t i = 0; i < n; ++i) d[i] = work(i, i);
    for (std::size_t i = 0; i + 1 < n; ++i) e[i] = work(i, i + 1);

    // GR QR sweeps on the bidiagonal.
    // Accumulate left rotations into U_f, right rotations into Vt_f.

    Matrix U_f  = Matrix::identity(n);
    Matrix Vt_f = Matrix::identity(n);

    std::size_t active = n;
    int total_iters = 0;

    while (active > 1) {
        if (total_iters >= opts.max_iterations) {
            std::ostringstream oss;
            oss << "svd: did not converge in " << opts.max_iterations << " iterations";
            throw NonConvergenceError(oss.str());
        }
        ++total_iters;

        // Deflate small superdiagonals.
        while (active > 1) {
            const std::size_t i = active - 2;
            if (std::abs(e[i]) <= opts.tolerance * (std::abs(d[i]) + std::abs(d[active - 1]))) {
                e[i] = 0.0;
                --active;
            } else {
                break;
            }
        }
        if (active <= 1) break;

        // Handle zero on diagonal: if d[k] == 0 for k < active-1, chase the
        // nonzero e[k] to zero using a sequence of left Givens rotations.
        bool zero_diag = false;
        for (std::size_t k = 0; k + 1 < active; ++k) {
            if (std::abs(d[k]) <= opts.tolerance) {
                zero_diag = true;
                // Chase e[k] to zero using left Givens rotations in rows k and k+1..active-1.
                double f = e[k];
                e[k] = 0.0;
                for (std::size_t j = k + 1; j < active && f != 0.0; ++j) {
                    const double g = d[j];
                    const double r = std::hypot(f, g);
                    const double c = g / r;
                    const double s = -f / r;
                    d[j] = r;
                    if (j + 1 < active) {
                        f = s * e[j];
                        e[j] *= c;
                    }
                    // Accumulate into U_f (left rotation on rows k and j).
                    // Apply a rotation between rows k and j manually on U_f columns.
                    for (std::size_t col = 0; col < n; ++col) {
                        const double uk = U_f(k, col);
                        const double uj = U_f(j, col);
                        U_f(k, col) =  c * uk - s * uj;
                        U_f(j, col) =  s * uk + c * uj;
                    }
                }
                break;
            }
        }
        if (zero_diag) continue;

        // Wilkinson shift from bottom 2×2 of B^T B.
        // B^T B bottom-right 2×2 (indices active-2, active-1):
        //   [d[a-2]^2 + e[a-3]^2,  d[a-2]*e[a-2]]
        //   [d[a-2]*e[a-2],          d[a-1]^2 + e[a-2]^2]  (if a>=2)
        const double a = active >= 2 ? d[active - 2] : 0.0;
        const double b_val = active >= 2 ? e[active - 2] : 0.0;
        const double c_val = d[active - 1];
        const double t11 = a * a + (active >= 3 ? e[active - 3] * e[active - 3] : 0.0);
        const double t12 = a * b_val;
        const double t22 = c_val * c_val + b_val * b_val;
        const double delta = 0.5 * (t11 - t22);
        const double denom = std::abs(delta) + std::hypot(delta, t12);
        const double mu = (denom == 0.0) ? t22
                        : t22 - (delta >= 0.0 ? 1.0 : -1.0) * (t12 * t12) / denom;

        // Golub-Reinsch implicit QR step.
        double f = d[0] * d[0] - mu;
        double g = d[0] * e[0];

        for (std::size_t i = 0; i + 1 < active; ++i) {
            // Right Givens: eliminate g from (f, g) in columns i and i+1.
            {
                const double r  = std::hypot(f, g);
                const double cr = (r == 0.0) ? 1.0 : f / r;
                const double sr = (r == 0.0) ? 0.0 : g / r;

                if (i > 0) e[i - 1] = r;

                f =  cr * d[i] + sr * e[i];
                e[i] = -sr * d[i] + cr * e[i];
                g = sr * d[i + 1];
                d[i + 1] *= cr;

                // Accumulate right rotation into Vt_f (acts on cols i and i+1 of V, i.e., rows of Vt_f).
                for (std::size_t row = 0; row < n; ++row) {
                    const double vi  = Vt_f(i,     row);
                    const double vi1 = Vt_f(i + 1, row);
                    Vt_f(i,     row) =  cr * vi + sr * vi1;
                    Vt_f(i + 1, row) = -sr * vi + cr * vi1;
                }
            }

            // Left Givens: eliminate g from (f, g) in rows i and i+1.
            {
                const double r  = std::hypot(f, g);
                const double cl = (r == 0.0) ? 1.0 : f / r;
                const double sl = (r == 0.0) ? 0.0 : g / r;

                d[i] = r;

                f =  cl * e[i] + sl * d[i + 1];
                d[i + 1] = -sl * e[i] + cl * d[i + 1];
                e[i] = f;

                if (i + 2 < active) {
                    g = sl * e[i + 1];
                    e[i + 1] *= cl;
                }

                // Accumulate left rotation into U_f (acts on rows i and i+1).
                for (std::size_t col = 0; col < n; ++col) {
                    const double ui  = U_f(i,     col);
                    const double ui1 = U_f(i + 1, col);
                    U_f(i,     col) =  cl * ui + sl * ui1;
                    U_f(i + 1, col) = -sl * ui + cl * ui1;
                }
            }
        }
        // Set the last diagonal update.
        e[active - 2] = f;
    }

    // U = U_acc * U_f^T  (U_f accumulates row operations on B,
    // which correspond to left singular vectors relative to U_acc).
    // Vt = Vt_f * Vt_acc  (Vt_f accumulates row ops on Vt_acc).
    // sigma = |d[i]|, flip signs into U.

    // U_f stores row-wise left rotations applied to the bidiagonal's rows.
    // The actual left factor is U_acc * U_f^T (since each left Givens G was
    // applied as B <- G B, meaning U_acc absorbs G^T from the right).
    const Matrix Uf_t = transpose(U_f);
    // U_acc is m×m, Uf_t is n×n.  We need m×m U; embed Uf_t into top-left.
    Matrix U_full = Matrix::zeros(m, m);
    // Copy U_acc * Uf_t into first n columns; remaining m-n columns of U_acc unchanged.
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t l = 0; l < n; ++l) s += U_acc(i, l) * Uf_t(l, j);
            U_full(i, j) = s;
        }
        for (std::size_t j = n; j < m; ++j) U_full(i, j) = U_acc(i, j);
    }

    Matrix Vt_full = Vt_f * Vt_acc;

    Vector sigma(n);
    for (std::size_t i = 0; i < n; ++i) {
        sigma[i] = std::abs(d[i]);
        if (d[i] < 0.0) {
            // Negate corresponding column of U (row of U^T) to keep sigma positive.
            for (std::size_t j = 0; j < m; ++j) U_full(j, i) = -U_full(j, i);
        }
    }

    // Sort singular values descending, applying same permutation to U cols and Vt rows.
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::sort(idx.begin(), idx.end(),
              [&](std::size_t a, std::size_t b) { return sigma[a] > sigma[b]; });

    Vector sigma_sorted(n);
    Matrix U_sorted(m, m);
    Matrix Vt_sorted(n, n);

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = n; j < m; ++j)
            U_sorted(i, j) = U_full(i, j);

    for (std::size_t rank = 0; rank < n; ++rank) {
        const std::size_t src = idx[rank];
        sigma_sorted[rank] = sigma[src];
        for (std::size_t i = 0; i < m; ++i) U_sorted(i, rank) = U_full(i, src);
        for (std::size_t j = 0; j < n; ++j) Vt_sorted(rank, j) = Vt_full(src, j);
    }

    return SVDResult{std::move(U_sorted), std::move(sigma_sorted), std::move(Vt_sorted)};
}

}  // namespace linalgebra
