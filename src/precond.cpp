export module linalgebra:precond;
import std;
import :error;
import :vector;
import :matrix;
import :norms;
import :triangular_solve;
import :lu;
import :qr;

// References used throughout this file:
//   T&B  — Trefethen & Bau, "Numerical Linear Algebra"
//   GVL  — Golub & Van Loan, "Matrix Computations" 4th ed.
//   Hager — Hager, W.W. (1984), SIAM J. Sci. Stat. Comput. 5(2):311-316

export namespace linalgebra {

// Condition number estimation (1-norm, Hager/LINPACK power iteration)
// Reference: Hager (1984); GVL §2.3.3

[[nodiscard]] double condition_number_1norm(const Matrix& A,
                                            double singular_tolerance = 1e-12);

// Jacobi (diagonal) preconditioner

struct JacobiPrecond {
    Vector inv_diag;
};

[[nodiscard]] JacobiPrecond precond_jacobi(const Matrix& A,
                                           double zero_tolerance = 1e-14);

[[nodiscard]] Vector apply(const JacobiPrecond& P, const Vector& x);

// ILU(0) preconditioner (for dense matrices = LU without pivoting)
// Reference: Saad, "Iterative Methods for Sparse Linear Systems" §10.3

struct ILU0Precond {
    Matrix LU;  // combined: strict lower = L multipliers, upper = U
};

[[nodiscard]] ILU0Precond precond_ilu0(const Matrix& A,
                                       double zero_tolerance = 1e-14);

[[nodiscard]] Vector apply(const ILU0Precond& P, const Vector& b);

// Least-squares solver via column-pivoting QR
// Reference: T&B Lecture 11; GVL §5.5

struct LstsqResult {
    Vector x;
    std::size_t rank;
    double residual_norm;
};

struct LstsqOptions {
    double rank_tolerance = 1e-12;
};

[[nodiscard]] LstsqResult lstsq(const Matrix& A, const Vector& b,
                                 LstsqOptions opts = {});

}  // namespace linalgebra

namespace {

// Solve A^T z = rhs given a pre-computed LU factorization of A.
// PA = LU  =>  A^T = U^T L^T P
// Steps: (1) solve U^T q = rhs, (2) solve L^T w = q, (3) z[perm[i]] = w[i]
linalgebra::Vector solve_transpose(const linalgebra::LUResult& lu,
                                   const linalgebra::Vector& rhs) {
    const std::size_t n = lu.L.rows();

    // Solve U^T q = rhs  (U^T is lower triangular)
    const linalgebra::Matrix Ut = linalgebra::transpose(lu.U);
    const linalgebra::Vector q = linalgebra::forward_substitution(Ut, rhs);

    // Solve L^T w = q  (L^T is upper triangular, unit diagonal)
    const linalgebra::Matrix Lt = linalgebra::transpose(lu.L);
    const linalgebra::Vector w = linalgebra::backward_substitution(Lt, q, 1e-14, true);

    // Apply inverse permutation: z[perm[i]] = w[i]
    linalgebra::Vector z(n);
    for (std::size_t i = 0; i < n; ++i) z[lu.perm[i]] = w[i];
    return z;
}

}  // namespace

namespace linalgebra {

double condition_number_1norm(const Matrix& A, double singular_tolerance) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "condition_number_1norm requires a square matrix, got "
            << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }
    const std::size_t n = A.rows();

    // Exact 1-norm of A: max column sum of absolute values.
    double norm_A = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        double col_sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) col_sum += std::abs(A(i, j));
        norm_A = std::max(norm_A, col_sum);
    }

    const LUResult lu = lu_factor(A, singular_tolerance);

    // Estimate ||A^{-1}||_1 via the power-iteration method (Hager 1984).
    // Start with x = [1/n, ..., 1/n].
    Vector x(n, 1.0 / static_cast<double>(n));
    double est = 0.0;

    for (int iter = 0; iter < 5; ++iter) {
        const Vector y = lu_solve(lu, x);  // y = A^{-1} x

        // 1-norm of y.
        double y1 = 0.0;
        for (std::size_t i = 0; i < n; ++i) y1 += std::abs(y[i]);

        if (y1 <= est) break;
        est = y1;

        Vector xi(n);
        for (std::size_t i = 0; i < n; ++i) xi[i] = (y[i] >= 0.0) ? 1.0 : -1.0;

        // z = A^{-T} xi
        const Vector z = solve_transpose(lu, xi);

        // Find the index maximizing |z[j]|.
        std::size_t j_max = 0;
        double max_z = std::abs(z[0]);
        for (std::size_t i = 1; i < n; ++i) {
            if (std::abs(z[i]) > max_z) {
                max_z = std::abs(z[i]);
                j_max = i;
            }
        }

        // Convergence check.
        double xz = 0.0;
        for (std::size_t i = 0; i < n; ++i) xz += std::abs(z[i]) / static_cast<double>(n);
        if (max_z <= xz) break;

        // New starting vector: e_{j_max}.
        x.fill(0.0);
        x[j_max] = 1.0;
    }

    return norm_A * est;
}

JacobiPrecond precond_jacobi(const Matrix& A, double zero_tolerance) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "precond_jacobi requires a square matrix, got "
            << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }
    const std::size_t n = A.rows();
    Vector inv_diag(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(A(i, i)) <= zero_tolerance) {
            std::ostringstream oss;
            oss << "precond_jacobi: zero diagonal entry at index " << i;
            throw SingularMatrixError(oss.str());
        }
        inv_diag[i] = 1.0 / A(i, i);
    }
    return JacobiPrecond{std::move(inv_diag)};
}

Vector apply(const JacobiPrecond& P, const Vector& x) {
    const std::size_t n = x.size();
    if (n != P.inv_diag.size()) {
        throw DimensionMismatchError("apply(JacobiPrecond): size mismatch");
    }
    Vector result(n);
    for (std::size_t i = 0; i < n; ++i) result[i] = P.inv_diag[i] * x[i];
    return result;
}

ILU0Precond precond_ilu0(const Matrix& A, double zero_tolerance) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "precond_ilu0 requires a square matrix, got "
            << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }
    const std::size_t n = A.rows();
    Matrix LU = A;

    for (std::size_t k = 0; k < n; ++k) {
        if (std::abs(LU(k, k)) <= zero_tolerance) {
            std::ostringstream oss;
            oss << "precond_ilu0: near-zero pivot at step " << k;
            throw SingularMatrixError(oss.str());
        }
        for (std::size_t i = k + 1; i < n; ++i) {
            LU(i, k) /= LU(k, k);
            for (std::size_t j = k + 1; j < n; ++j) {
                LU(i, j) -= LU(i, k) * LU(k, j);
            }
        }
    }
    return ILU0Precond{std::move(LU)};
}

Vector apply(const ILU0Precond& P, const Vector& b) {
    const std::size_t n = P.LU.rows();
    if (b.size() != n) {
        throw DimensionMismatchError("apply(ILU0Precond): size mismatch");
    }

    // Extract L (unit lower) and U (upper) from combined storage.
    Matrix L = Matrix::zeros(n, n);
    Matrix U = Matrix::zeros(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        L(i, i) = 1.0;
        for (std::size_t j = 0; j < i; ++j) L(i, j) = P.LU(i, j);
        for (std::size_t j = i; j < n; ++j) U(i, j) = P.LU(i, j);
    }

    const Vector y = forward_substitution(L, b, 1e-14, true);
    return backward_substitution(U, y);
}

LstsqResult lstsq(const Matrix& A, const Vector& b, LstsqOptions opts) {
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    if (m < n) {
        std::ostringstream oss;
        oss << "lstsq requires rows >= cols, got " << m << "x" << n;
        throw DimensionMismatchError(oss.str());
    }
    if (b.size() != m) {
        std::ostringstream oss;
        oss << "lstsq: rhs size " << b.size() << " does not match rows " << m;
        throw DimensionMismatchError(oss.str());
    }

    const QRColPivResult qr = qr_colpiv(A, opts.rank_tolerance);
    const std::size_t r = qr.rank;

    // c = Q^T b  (Q is m×n with orthonormal columns)
    Vector c(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        double d = 0.0;
        for (std::size_t i = 0; i < m; ++i) d += qr.Q(i, j) * b[i];
        c[j] = d;
    }

    Vector x(n, 0.0);

    if (r > 0) {
        Matrix Rr(r, r);
        for (std::size_t i = 0; i < r; ++i)
            for (std::size_t j = 0; j < r; ++j)
                Rr(i, j) = qr.R(i, j);

        Vector cr(r);
        for (std::size_t i = 0; i < r; ++i) cr[i] = c[i];

        const Vector y = backward_substitution(Rr, cr);

        // Permuted solution: x_perm[0:r] = y, x_perm[r:n] = 0, then un-permute.
        for (std::size_t j = 0; j < r; ++j) x[qr.perm[j]] = y[j];
    }

    const Vector res = A * x - b;
    const double residual_norm = norm2(res);

    return LstsqResult{std::move(x), r, residual_norm};
}

}  // namespace linalgebra
