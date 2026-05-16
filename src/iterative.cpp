export module linalgebra:iterative;
import std;
import :error;
import :vector;
import :matrix;
import :norms;
import :triangular_solve;

// References used throughout this file:
//   T&B  — Trefethen & Bau, "Numerical Linear Algebra"
//   GVL  — Golub & Van Loan, "Matrix Computations" 4th ed.

export namespace linalgebra {

// Arnoldi iteration builds an orthonormal Krylov basis and the
// corresponding upper Hessenberg matrix.
// Reference: T&B Algorithm 33.1; GVL §6.3

struct ArnoldiResult {
    Matrix Q;          // n × (steps_taken + 1) orthonormal columns
    Matrix H;          // (steps_taken + 1) × steps_taken upper Hessenberg
    int    steps_taken;
    bool   breakdown;  // true if invariant subspace found early
};

struct ArnoldiOptions {
    double breakdown_tolerance = 1e-14;
};

[[nodiscard]] ArnoldiResult arnoldi(const Matrix& A, const Vector& b, int k,
                                    ArnoldiOptions opts = {});

// Conjugate Gradient — for symmetric positive definite systems
// Reference: T&B Algorithm 38.1; GVL §11.3

struct CGOptions {
    double tolerance     = 1e-10;
    int    max_iterations = 1000;
};

struct CGResult {
    Vector x;
    int    iterations;
    double final_residual;
};

[[nodiscard]] CGResult solve_cg(const Matrix& A, const Vector& b, CGOptions opts = {});

// Restarted GMRES — for general square systems
// Reference: T&B Algorithm 35.1; GVL §11.4.2

struct GMRESOptions {
    double tolerance      = 1e-10;
    int    max_iterations = 200;
    int    restart        = 50;
};

struct GMRESResult {
    Vector x;
    int    iterations;
    double final_residual;
};

[[nodiscard]] GMRESResult solve_gmres(const Matrix& A, const Vector& b,
                                      GMRESOptions opts = {});

// BiCGSTAB — for general square systems (van der Vorst 1992)
// Reference: GVL §11.5.3

struct BiCGSTABOptions {
    double tolerance     = 1e-10;
    int    max_iterations = 1000;
};

struct BiCGSTABResult {
    Vector x;
    int    iterations;
    double final_residual;
};

[[nodiscard]] BiCGSTABResult solve_bicgstab(const Matrix& A, const Vector& b,
                                             BiCGSTABOptions opts = {});

}  // namespace linalgebra

namespace linalgebra {

ArnoldiResult arnoldi(const Matrix& A, const Vector& b, int k, ArnoldiOptions opts) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "arnoldi requires a square matrix, got " << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }
    const std::size_t n = A.rows();
    if (b.size() != n) {
        std::ostringstream oss;
        oss << "arnoldi: b size " << b.size() << " must match matrix dimension " << n;
        throw DimensionMismatchError(oss.str());
    }
    if (k <= 0) throw std::invalid_argument("arnoldi: k must be >= 1");

    const auto kk = static_cast<std::size_t>(k);

    Matrix Q(n, kk + 1, 0.0);
    Matrix H(kk + 1, kk, 0.0);

    const double b_norm = norm2(b);
    if (b_norm == 0.0) {
        return ArnoldiResult{std::move(Q), std::move(H), 0, true};
    }

    // q_0 = b / ||b||
    for (std::size_t i = 0; i < n; ++i) Q(i, 0) = b[i] / b_norm;

    int steps = 0;
    for (std::size_t j = 0; j < kk; ++j) {
        // z = A * Q[:, j]
        Vector qj(n);
        for (std::size_t i = 0; i < n; ++i) qj[i] = Q(i, j);
        Vector z = A * qj;

        // Modified Gram-Schmidt orthogonalization.
        for (std::size_t i = 0; i <= j; ++i) {
            double h = 0.0;
            for (std::size_t row = 0; row < n; ++row) h += Q(row, i) * z[row];
            H(i, j) = h;
            for (std::size_t row = 0; row < n; ++row) z[row] -= h * Q(row, i);
        }

        const double z_norm = norm2(z);
        H(j + 1, j) = z_norm;
        ++steps;

        if (z_norm < opts.breakdown_tolerance) {
            // Lucky breakdown: invariant subspace found.
            return ArnoldiResult{std::move(Q), std::move(H), steps, true};
        }

        for (std::size_t i = 0; i < n; ++i) Q(i, j + 1) = z[i] / z_norm;
    }

    return ArnoldiResult{std::move(Q), std::move(H), steps, false};
}

CGResult solve_cg(const Matrix& A, const Vector& b, CGOptions opts) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "solve_cg requires a square matrix, got " << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }
    const std::size_t n = A.rows();
    if (b.size() != n) {
        std::ostringstream oss;
        oss << "solve_cg: rhs size " << b.size() << " does not match dimension " << n;
        throw DimensionMismatchError(oss.str());
    }

    // Check symmetry.
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            if (std::abs(A(i, j) - A(j, i)) > 1e-10 * (std::abs(A(i, j)) + 1.0))
                throw LinAlgError("solve_cg: matrix is not symmetric");

    Vector x(n, 0.0);
    Vector r = b;  // r = b - A*x0, x0 = 0
    Vector p = r;
    double rr = dot(r, r);

    if (std::sqrt(rr) < opts.tolerance) {
        return CGResult{std::move(x), 0, std::sqrt(rr)};
    }

    for (int iter = 1; iter <= opts.max_iterations; ++iter) {
        const Vector Ap = A * p;
        const double pAp = dot(p, Ap);
        if (std::abs(pAp) == 0.0) break;
        const double alpha = rr / pAp;

        for (std::size_t i = 0; i < n; ++i) x[i] += alpha * p[i];
        for (std::size_t i = 0; i < n; ++i) r[i] -= alpha * Ap[i];

        const double rr_new = dot(r, r);
        const double res = std::sqrt(rr_new);

        if (res < opts.tolerance) {
            return CGResult{std::move(x), iter, res};
        }

        const double beta = rr_new / rr;
        for (std::size_t i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rr = rr_new;
    }

    const double final_res = norm2(A * x - b);
    std::ostringstream oss;
    oss << "solve_cg: did not converge in " << opts.max_iterations
        << " iterations (final residual = " << final_res << ")";
    throw NonConvergenceError(oss.str());
}

GMRESResult solve_gmres(const Matrix& A, const Vector& b, GMRESOptions opts) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "solve_gmres requires a square matrix, got " << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }
    const std::size_t n = A.rows();
    if (b.size() != n) {
        std::ostringstream oss;
        oss << "solve_gmres: rhs size " << b.size() << " does not match dimension " << n;
        throw DimensionMismatchError(oss.str());
    }

    const int m = std::min(opts.restart, static_cast<int>(n));

    Vector x(n, 0.0);
    int total_iters = 0;

    while (total_iters < opts.max_iterations) {
        // Compute residual r = b - A*x.
        const Vector Ax = A * x;
        Vector r(n);
        for (std::size_t i = 0; i < n; ++i) r[i] = b[i] - Ax[i];
        const double beta = norm2(r);

        if (beta < opts.tolerance) {
            return GMRESResult{std::move(x), total_iters, beta};
        }

        // Arnoldi to build Krylov basis.
        const ArnoldiResult ar = arnoldi(A, r, m);
        const int steps = ar.steps_taken;

        if (steps == 0) break;

        // Solve least-squares problem: min ||beta*e1 - H_hat * y||
        // where H_hat is (steps+1) x steps.
        // Apply Givens rotations to reduce H_hat to upper triangular.
        const auto sz = static_cast<std::size_t>(steps);

        // Work on a copy of the relevant submatrix of H and rhs g.
        std::vector<std::vector<double>> Hwork(sz + 1, std::vector<double>(sz, 0.0));
        for (std::size_t i = 0; i <= sz; ++i)
            for (std::size_t j = 0; j < sz; ++j)
                Hwork[i][j] = ar.H(i, j);

        std::vector<double> g(sz + 1, 0.0);
        g[0] = beta;

        // Accumulated Givens rotations.
        std::vector<double> cs(sz), sn(sz);

        for (std::size_t j = 0; j < sz; ++j) {
            // Givens to zero H[j+1, j].
            const double f = Hwork[j][j];
            const double hh = Hwork[j + 1][j];
            const double r_val = std::hypot(f, hh);
            if (r_val == 0.0) { cs[j] = 1.0; sn[j] = 0.0; continue; }
            cs[j] =  f / r_val;
            sn[j] = hh / r_val;

            // Apply to column j of H (only the two relevant rows).
            Hwork[j][j]     =  cs[j] * f + sn[j] * hh;
            Hwork[j + 1][j] = 0.0;

            // Apply to remaining columns.
            for (std::size_t l = j + 1; l < sz; ++l) {
                const double t0 = Hwork[j][l];
                const double t1 = Hwork[j + 1][l];
                Hwork[j][l]     =  cs[j] * t0 + sn[j] * t1;
                Hwork[j + 1][l] = -sn[j] * t0 + cs[j] * t1;
            }

            // Apply to g.
            const double g0 = g[j];
            const double g1 = g[j + 1];
            g[j]     =  cs[j] * g0 + sn[j] * g1;
            g[j + 1] = -sn[j] * g0 + cs[j] * g1;
        }

        // Backward substitution: solve the sz×sz upper triangular system.
        std::vector<double> y(sz, 0.0);
        for (int i = static_cast<int>(sz) - 1; i >= 0; --i) {
            double sum = g[static_cast<std::size_t>(i)];
            for (std::size_t j = static_cast<std::size_t>(i) + 1; j < sz; ++j)
                sum -= Hwork[static_cast<std::size_t>(i)][j] * y[j];
            if (std::abs(Hwork[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)]) > 0.0)
                y[static_cast<std::size_t>(i)] =
                    sum / Hwork[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
        }

        // Update x = x + Q[:, 0:steps] * y.
        for (std::size_t j = 0; j < sz; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                x[i] += y[j] * ar.Q(i, j);
            }
        }

        ++total_iters;

        // Check convergence.
        const double final_res = std::abs(g[sz]);
        if (final_res < opts.tolerance || ar.breakdown) {
            return GMRESResult{std::move(x), total_iters, final_res};
        }
    }

    const double final_res = norm2(A * x - b);
    std::ostringstream oss;
    oss << "solve_gmres: did not converge in " << opts.max_iterations
        << " restarts (final residual = " << final_res << ")";
    throw NonConvergenceError(oss.str());
}

BiCGSTABResult solve_bicgstab(const Matrix& A, const Vector& b, BiCGSTABOptions opts) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "solve_bicgstab requires a square matrix, got "
            << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }
    const std::size_t n = A.rows();
    if (b.size() != n) {
        std::ostringstream oss;
        oss << "solve_bicgstab: rhs size " << b.size() << " does not match dimension " << n;
        throw DimensionMismatchError(oss.str());
    }

    Vector x(n, 0.0);
    Vector r = b;  // r = b - A*x0, x0 = 0
    Vector r_hat = r;  // shadow residual, fixed throughout

    double rho_old = 1.0, alpha = 1.0, omega = 1.0;
    Vector v(n, 0.0), p(n, 0.0);

    for (int iter = 1; iter <= opts.max_iterations; ++iter) {
        const double rho_new = dot(r_hat, r);

        if (std::abs(rho_new) < 1e-300) {
            std::ostringstream oss;
            oss << "solve_bicgstab: breakdown (rho near zero) at iteration " << iter;
            throw NonConvergenceError(oss.str());
        }

        const double beta = (rho_new / rho_old) * (alpha / omega);

        for (std::size_t i = 0; i < n; ++i)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);

        v = A * p;

        const double denom_alpha = dot(r_hat, v);
        if (std::abs(denom_alpha) < 1e-300) {
            std::ostringstream oss;
            oss << "solve_bicgstab: breakdown (r_hat·v near zero) at iteration " << iter;
            throw NonConvergenceError(oss.str());
        }
        alpha = rho_new / denom_alpha;

        Vector s(n);
        for (std::size_t i = 0; i < n; ++i) s[i] = r[i] - alpha * v[i];

        const double s_norm = norm2(s);
        if (s_norm < opts.tolerance) {
            for (std::size_t i = 0; i < n; ++i) x[i] += alpha * p[i];
            return BiCGSTABResult{std::move(x), iter, s_norm};
        }

        const Vector t = A * s;
        const double tt = dot(t, t);

        if (std::abs(tt) < 1e-300) {
            std::ostringstream oss;
            oss << "solve_bicgstab: breakdown (t·t near zero) at iteration " << iter;
            throw NonConvergenceError(oss.str());
        }

        omega = dot(t, s) / tt;

        if (std::abs(omega) < 1e-300) {
            std::ostringstream oss;
            oss << "solve_bicgstab: breakdown (omega near zero) at iteration " << iter;
            throw NonConvergenceError(oss.str());
        }

        for (std::size_t i = 0; i < n; ++i) x[i] += alpha * p[i] + omega * s[i];
        for (std::size_t i = 0; i < n; ++i) r[i] = s[i] - omega * t[i];

        const double r_norm = norm2(r);
        if (r_norm < opts.tolerance) {
            return BiCGSTABResult{std::move(x), iter, r_norm};
        }

        rho_old = rho_new;
    }

    const double final_res = norm2(A * x - b);
    std::ostringstream oss;
    oss << "solve_bicgstab: did not converge in " << opts.max_iterations
        << " iterations (final residual = " << final_res << ")";
    throw NonConvergenceError(oss.str());
}

}  // namespace linalgebra
