module;
#include <cassert>

export module linalgebra:qr_iteration;
import std;
import :error;
import :vector;
import :matrix;
import :qr;

// References used throughout this file:
//   T&B  — Trefethen & Bau, "Numerical Linear Algebra"
//   GVL  — Golub & Van Loan, "Matrix Computations" 4th ed.

export namespace linalgebra {

// Options

struct QRIterationOptions {
    double tolerance = 1e-10;
    int max_iterations = 1000;
    bool track_convergence = false;
};

struct QRIterationResult {
    Vector eigenvalues_real;
    Vector eigenvalues_imag;
    int iterations = 0;
    // std::vector is used here because linalgebra::Vector has no push_back;
    // convergence_history is a plain time-series container, not a math object.
    std::vector<double> convergence_history;
};

[[nodiscard]] QRIterationResult eigenvalues_unshifted(const Matrix& A,
                                                      QRIterationOptions opts = {});

[[nodiscard]] QRIterationResult eigenvalues_shifted(const Matrix& A,
                                                    QRIterationOptions opts = {});

// Givens rotation G acting on rows/columns i and i+1:
//
//   G = | c   s |  chosen so that G * [x; y]^T = [r; 0]^T
//       | -s  c |  with c = x/r, s = y/r, r = hypot(x, y)
struct GivensRotation {
    double      c;
    double      s;
    std::size_t i;

    [[nodiscard]] static GivensRotation make(double x, double y, std::size_t row_index);
    void apply_left(Matrix& M, std::size_t col_start = 0) const;
    void apply_right(Matrix& M, std::size_t row_end) const;
};

struct HessenbergResult {
    Matrix H;
    Matrix Q;
};

[[nodiscard]] HessenbergResult hessenberg_reduction(const Matrix& A);

void hessenberg_qr_step(Matrix& H, double sigma);

[[nodiscard]] QRIterationResult eigenvalues_hessenberg(const Matrix& A,
                                                       QRIterationOptions opts = {});

// Francis double-shift QR — implicit bulge chasing on upper Hessenberg form.
// Handles real matrices with complex conjugate eigenvalue pairs without
// complex arithmetic.  Uses robust deflation (subdiagonal + 2×2 block).
// Reference: GVL §7.5, T&B Lecture 29.
[[nodiscard]] QRIterationResult eigenvalues_francis(const Matrix& A,
                                                    QRIterationOptions opts = {});

}  // namespace linalgebra

namespace {

double lower_triangle_norm(const linalgebra::Matrix& A) {
    const std::size_t n = A.rows();
    double s = 0.0;
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j)
            s += A(i, j) * A(i, j);
    return std::sqrt(s);
}

void extract_eigenvalues(const linalgebra::Matrix& T, double tol,
                         linalgebra::Vector& real_out, linalgebra::Vector& imag_out) {
    const std::size_t n = T.rows();
    std::size_t out = 0;
    std::size_t i   = 0;

    while (i < n) {
        const bool is_last   = (i + 1 == n);
        const bool sub_small = is_last || (std::abs(T(i + 1, i)) < tol);

        if (sub_small) {
            real_out[out] = T(i, i);
            imag_out[out] = 0.0;
            ++out;
            ++i;
        } else {
            const double a = T(i, i);
            const double b = T(i, i + 1);
            const double c = T(i + 1, i);
            const double d = T(i + 1, i + 1);
            const double tr = a + d;
            const double disc = (a - d) * (a - d) + 4.0 * b * c;

            if (disc >= 0.0) {
                const double sq   = std::sqrt(disc);
                real_out[out]     = 0.5 * (tr + sq);
                imag_out[out]     = 0.0;
                real_out[out + 1] = 0.5 * (tr - sq);
                imag_out[out + 1] = 0.0;
            } else {
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

void require_square(const linalgebra::Matrix& A, const char* fname) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << fname << ": requires a square matrix, got "
            << A.rows() << "x" << A.cols();
        throw linalgebra::DimensionMismatchError(oss.str());
    }
}

double wilkinson_shift(const linalgebra::Matrix& A) {
    const std::size_t n = A.rows();
    const double a = A(n - 2, n - 2);
    const double b = A(n - 1, n - 2);
    const double d = A(n - 1, n - 1);
    const double delta = 0.5 * (a - d);
    const double denom = std::abs(delta) + std::hypot(delta, b);
    if (denom == 0.0) return d;
    const double sgn = (delta >= 0.0) ? 1.0 : -1.0;
    return d - sgn * (b * b) / denom;
}

}  // namespace

namespace linalgebra {

QRIterationResult eigenvalues_unshifted(const Matrix& A, QRIterationOptions opts) {
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
        const QRResult qr = qr_householder(Ak);
        Ak = qr.R * qr.Q;

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
        while (active >= 2) {
            const double sub   = std::abs(Ak(active - 1, active - 2));
            const double scale = std::abs(Ak(active - 2, active - 2))
                               + std::abs(Ak(active - 1, active - 1));
            const double deflation_tol =
                opts.tolerance * (scale > 0.0 ? scale : 1.0);
            if (sub > deflation_tol) break;
            Ak(active - 1, active - 2) = 0.0;
            store_real(Ak(active - 1, active - 1));
            --active;
        }

        if (active == 0) break;
        if (active == 1) { store_real(Ak(0, 0)); active = 0; break; }
        if (active == 2) { close_2x2();                       break; }

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

GivensRotation GivensRotation::make(double x, double y, std::size_t row_index) {
    const double r = std::hypot(x, y);
    if (r == 0.0) return {1.0, 0.0, row_index};
    return {x / r, y / r, row_index};
}

void GivensRotation::apply_left(Matrix& M, std::size_t col_start) const {
    for (std::size_t j = col_start; j < M.cols(); ++j) {
        const double xi  = M(i,     j);
        const double xi1 = M(i + 1, j);
        M(i,     j) =  c * xi + s * xi1;
        M(i + 1, j) = -s * xi + c * xi1;
    }
}

void GivensRotation::apply_right(Matrix& M, std::size_t row_end) const {
    for (std::size_t j = 0; j < row_end; ++j) {
        const double xi  = M(j, i);
        const double xi1 = M(j, i + 1);
        M(j, i)     =  c * xi + s * xi1;
        M(j, i + 1) = -s * xi + c * xi1;
    }
}

HessenbergResult hessenberg_reduction(const Matrix& A) {
    require_square(A, "hessenberg_reduction");
    const std::size_t n = A.rows();

    Matrix H = A;
    Matrix Q = Matrix::identity(n);

    for (std::size_t k = 0; k + 2 <= n; ++k) {
        const std::size_t p = n - k - 1;
        if (p == 0) break;

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

        for (std::size_t j = k; j < n; ++j) {
            double d = 0.0;
            for (std::size_t i = 0; i < p; ++i) d += u[i] * H(k + 1 + i, j);
            const double coeff = tau * d;
            for (std::size_t i = 0; i < p; ++i) H(k + 1 + i, j) -= coeff * u[i];
        }

        for (std::size_t j = 0; j < n; ++j) {
            double d = 0.0;
            for (std::size_t i = 0; i < p; ++i) d += H(j, k + 1 + i) * u[i];
            const double coeff = tau * d;
            for (std::size_t i = 0; i < p; ++i) H(j, k + 1 + i) -= coeff * u[i];
        }

        for (std::size_t j = 0; j < n; ++j) {
            double d = 0.0;
            for (std::size_t i = 0; i < p; ++i) d += Q(j, k + 1 + i) * u[i];
            const double coeff = tau * d;
            for (std::size_t i = 0; i < p; ++i) Q(j, k + 1 + i) -= coeff * u[i];
        }

        for (std::size_t i = 1; i < p; ++i) H(k + 1 + i, k) = 0.0;
    }

    return HessenbergResult{std::move(H), std::move(Q)};
}

void hessenberg_qr_step(Matrix& H, double sigma) {
    const std::size_t n = H.rows();

    for (std::size_t j = 0; j < n; ++j) H(j, j) -= sigma;

    std::vector<GivensRotation> gs;
    gs.reserve(n - 1);

    for (std::size_t k = 0; k + 1 < n; ++k) {
        GivensRotation g = GivensRotation::make(H(k, k), H(k + 1, k), k);
        g.apply_left(H, k);
        gs.push_back(g);
    }

    for (std::size_t k = 0; k + 1 < n; ++k) {
        gs[k].apply_right(H, std::min(k + 2, n));
    }

    for (std::size_t j = 0; j < n; ++j) H(j, j) += sigma;
}

QRIterationResult eigenvalues_hessenberg(const Matrix& A, QRIterationOptions opts) {
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

        const double a_w = H(active - 2, active - 2);
        const double b_w = H(active - 1, active - 2);
        const double d_w = H(active - 1, active - 1);
        const double delta = 0.5 * (a_w - d_w);
        const double denom = std::abs(delta) + std::hypot(delta, b_w);
        const double sigma = (denom == 0.0) ? d_w
            : d_w - ((delta >= 0.0) ? 1.0 : -1.0) * (b_w * b_w) / denom;

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

// Francis double-shift QR with implicit bulge chasing

QRIterationResult eigenvalues_francis(const Matrix& A, QRIterationOptions opts) {
    require_square(A, "eigenvalues_francis");
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

    if (n == 2) {
        const double a = A(0, 0), b = A(0, 1), c = A(1, 0), d = A(1, 1);
        const double tr = a + d;
        const double disc = (a - d) * (a - d) + 4.0 * b * c;
        if (disc >= 0.0) {
            const double sq = std::sqrt(disc);
            result.eigenvalues_real[0] = 0.5 * (tr + sq);
            result.eigenvalues_real[1] = 0.5 * (tr - sq);
        } else {
            result.eigenvalues_real[0] = 0.5 * tr;
            result.eigenvalues_imag[0] = 0.5 * std::sqrt(-disc);
            result.eigenvalues_real[1] = 0.5 * tr;
            result.eigenvalues_imag[1] = -0.5 * std::sqrt(-disc);
        }
        return result;
    }

    HessenbergResult hr = hessenberg_reduction(A);
    Matrix& H = hr.H;

    std::size_t n_found = n;
    std::size_t active = n;

    auto store_real = [&](double re) {
        --n_found;
        result.eigenvalues_real[n_found] = re;
        result.eigenvalues_imag[n_found] = 0.0;
    };

    auto store_pair = [&](double re, double im) {
        --n_found; result.eigenvalues_real[n_found] = re; result.eigenvalues_imag[n_found] = im;
        --n_found; result.eigenvalues_real[n_found] = re; result.eigenvalues_imag[n_found] = -im;
    };

    // Robust deflation: checks both subdiagonal magnitude and 2x2 block.
    auto deflation_tol = [&](std::size_t i) -> double {
        const double scale = std::abs(H(i - 1, i - 1)) + std::abs(H(i, i));
        return opts.tolerance * (scale > 0.0 ? scale : 1.0);
    };

    auto close_2x2 = [&]() {
        const double a = H(active - 2, active - 2);
        const double b = H(active - 2, active - 1);
        const double c = H(active - 1, active - 2);
        const double d = H(active - 1, active - 1);
        const double tr = a + d;
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

    // Find the start of the active unreduced block (split from the top).
    auto find_block_start = [&]() -> std::size_t {
        for (std::size_t i = active - 1; i >= 1; --i) {
            if (std::abs(H(i, i - 1)) < deflation_tol(i)) {
                H(i, i - 1) = 0.0;
                return i;
            }
        }
        return 0;
    };

    int exceptional_shift_count = 0;

    for (int k = 0; k < opts.max_iterations; ++k) {
        // Deflate converged eigenvalues from bottom.
        while (active >= 2) {
            if (std::abs(H(active - 1, active - 2)) < deflation_tol(active - 1)) {
                H(active - 1, active - 2) = 0.0;
                store_real(H(active - 1, active - 1));
                --active;
            } else {
                break;
            }
        }

        if (active == 0) break;
        if (active == 1) { store_real(H(0, 0)); active = 0; break; }
        if (active == 2) { close_2x2(); break; }

        // Check for 2x2 block deflation (complex pair at bottom).
        if (active >= 3 && std::abs(H(active - 2, active - 3)) < deflation_tol(active - 2)) {
            H(active - 2, active - 3) = 0.0;
            // The bottom 2x2 has converged.
            const double a = H(active - 2, active - 2);
            const double b = H(active - 2, active - 1);
            const double c = H(active - 1, active - 2);
            const double d = H(active - 1, active - 1);
            const double tr = a + d;
            const double disc = (a - d) * (a - d) + 4.0 * b * c;
            if (disc >= 0.0) {
                const double sq = std::sqrt(disc);
                store_real(0.5 * (tr + sq));
                store_real(0.5 * (tr - sq));
            } else {
                store_pair(0.5 * tr, 0.5 * std::sqrt(-disc));
            }
            active -= 2;
            exceptional_shift_count = 0;
            continue;
        }

        std::size_t block_start = find_block_start();

        // Compute Francis double shift from bottom 2x2 of active block.
        const double a11 = H(active - 2, active - 2);
        const double a12 = H(active - 2, active - 1);
        const double a21 = H(active - 1, active - 2);
        const double a22 = H(active - 1, active - 1);
        double s = a11 + a22;  // trace of bottom 2x2
        double t = a11 * a22 - a12 * a21;  // determinant of bottom 2x2

        // Exceptional shift (Wilkinson's ad hoc) every 10 iterations to break stalls.
        if (exceptional_shift_count > 0 && exceptional_shift_count % 10 == 0) {
            const double w = std::abs(H(active - 1, active - 2))
                           + std::abs(H(block_start + 1, block_start));
            s = 1.5 * w;
            t = w * w;
        }

        // First column of M = H^2 - sH + tI (implicit).
        const double h00 = H(block_start, block_start);
        const double h01 = H(block_start, block_start + 1);
        const double h10 = H(block_start + 1, block_start);
        const double h11 = H(block_start + 1, block_start + 1);
        const double h21 = (block_start + 2 < active) ? H(block_start + 2, block_start + 1) : 0.0;

        double x = h00 * h00 + h01 * h10 - s * h00 + t;
        double y = h10 * (h00 + h11 - s);
        double z = h10 * h21;

        // Chase the bulge through the Hessenberg matrix.
        for (std::size_t i = block_start; i + 2 < active; ++i) {
            // Determine Householder reflector P such that P * [x; y; z]^T = [*; 0; 0]^T.
            const std::size_t p = (i + 3 <= active) ? 3 : 2;

            double norm_v = std::sqrt(x * x + y * y + (p == 3 ? z * z : 0.0));
            if (norm_v == 0.0) break;

            const double sign = (x >= 0.0) ? 1.0 : -1.0;
            double v0 = x + sign * norm_v;
            double v1 = y;
            double v2 = (p == 3) ? z : 0.0;

            const double vdot = v0 * v0 + v1 * v1 + v2 * v2;
            const double tau = 2.0 / vdot;

            // Apply P from left to H rows [i, i+p-1], columns [max(i-1,0), active-1].
            const std::size_t col_start = (i > 0) ? i - 1 : 0;
            for (std::size_t j = col_start; j < active; ++j) {
                double d = v0 * H(i, j) + v1 * H(i + 1, j);
                if (p == 3) d += v2 * H(i + 2, j);
                const double coeff = tau * d;
                H(i, j) -= coeff * v0;
                H(i + 1, j) -= coeff * v1;
                if (p == 3) H(i + 2, j) -= coeff * v2;
            }

            // Apply P from right to H rows [0, min(i+p, active-1)], columns [i, i+p-1].
            const std::size_t row_end = std::min(i + p + 1, active);
            for (std::size_t j = 0; j < row_end; ++j) {
                double d = v0 * H(j, i) + v1 * H(j, i + 1);
                if (p == 3) d += v2 * H(j, i + 2);
                const double coeff = tau * d;
                H(j, i) -= coeff * v0;
                H(j, i + 1) -= coeff * v1;
                if (p == 3) H(j, i + 2) -= coeff * v2;
            }

            // Prepare for next bulge step.
            if (i + 3 < active) {
                x = H(i + 1, i);
                y = H(i + 2, i);
                z = (i + 3 < active) ? H(i + 3, i) : 0.0;
            }
        }

        // Final 2x2 reflector to restore Hessenberg form at bottom.
        {
            const std::size_t i = active - 2;
            const double xi = H(i, i - 1);
            const double yi = H(i + 1, i - 1);
            const double r = std::hypot(xi, yi);
            if (r > 0.0) {
                const double c = xi / r;
                const double s_val = yi / r;
                // Apply Givens from left.
                for (std::size_t j = i - 1; j < active; ++j) {
                    const double t0 = H(i, j);
                    const double t1 = H(i + 1, j);
                    H(i, j) = c * t0 + s_val * t1;
                    H(i + 1, j) = -s_val * t0 + c * t1;
                }
                // Apply Givens from right.
                for (std::size_t j = 0; j < std::min(i + 3, active); ++j) {
                    const double t0 = H(j, i);
                    const double t1 = H(j, i + 1);
                    H(j, i) = c * t0 + s_val * t1;
                    H(j, i + 1) = -s_val * t0 + c * t1;
                }
            }
        }

        ++exceptional_shift_count;

        if (opts.track_convergence) {
            double s_norm = 0.0;
            for (std::size_t ii = 1; ii < active; ++ii)
                s_norm += H(ii, ii - 1) * H(ii, ii - 1);
            result.convergence_history.push_back(std::sqrt(s_norm));
        }
        ++result.iterations;
    }

    if (n_found > 0) {
        std::ostringstream oss;
        oss << "eigenvalues_francis: did not converge in "
            << opts.max_iterations << " iterations ("
            << n_found << " eigenvalue(s) not yet deflated).";
        throw NonConvergenceError(oss.str());
    }
    return result;
}

}  // namespace linalgebra
