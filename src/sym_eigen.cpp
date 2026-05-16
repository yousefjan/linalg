export module linalgebra:sym_eigen;
import std;
import :error;
import :vector;
import :matrix;
import :norms;
import :lu;

// References used throughout this file:
//   T&B  — Trefethen & Bau, "Numerical Linear Algebra"
//   GVL  — Golub & Van Loan, "Matrix Computations" 4th ed.

export namespace linalgebra {

// Symmetric tridiagonalization via Householder reflections
// Reference: GVL §8.3.1; T&B Lecture 26
//
// For symmetric A, computes Q and T such that A = Q T Q^T,
// where T is symmetric tridiagonal and Q is orthogonal.

struct TridiagonalizeResult {
    Matrix T;  // symmetric tridiagonal (full n×n)
    Matrix Q;  // orthogonal: A = Q T Q^T
};

[[nodiscard]] TridiagonalizeResult tridiagonalize(const Matrix& A,
                                                   double symmetry_tolerance = 1e-12);

// Eigenvectors via inverse iteration
// Reference: GVL §7.6.1; T&B Lecture 27
//
// Given A and a vector of approximate eigenvalues, returns a matrix whose
// columns are the corresponding (approximate) eigenvectors.

struct InverseIterationOptions {
    double tolerance     = 1e-10;
    int    max_iterations = 100;
};

struct InverseIterationResult {
    Matrix eigenvectors;  // n × k, column j is eigenvector for eigenvalues[j]
    Vector residuals;     // ||A*v_j - lambda_j*v_j||_2 for each j
};

[[nodiscard]] InverseIterationResult eigenvectors_inverse_iteration(
    const Matrix& A,
    const Vector& eigenvalues,
    InverseIterationOptions opts = {});

}  // namespace linalgebra

namespace linalgebra {

TridiagonalizeResult tridiagonalize(const Matrix& A, double symmetry_tolerance) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "tridiagonalize requires a square matrix, got "
            << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }

    const std::size_t n = A.rows();

    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            if (std::abs(A(i, j) - A(j, i)) > symmetry_tolerance)
                throw LinAlgError("tridiagonalize requires a symmetric matrix");

    Matrix T = A;
    Matrix Q = Matrix::identity(n);

    for (std::size_t k = 0; k + 2 <= n; ++k) {
        const std::size_t p = n - k - 1;  // length of sub-column

        // Form Householder vector u from T[k+1:, k].
        std::vector<double> u(p);
        for (std::size_t i = 0; i < p; ++i) u[i] = T(k + 1 + i, k);

        double x_norm = 0.0;
        for (double v : u) x_norm += v * v;
        x_norm = std::sqrt(x_norm);

        if (x_norm == 0.0) continue;

        const double sigma = (u[0] >= 0.0 ? 1.0 : -1.0) * x_norm;
        u[0] += sigma;

        double utu = 0.0;
        for (double v : u) utu += v * v;
        const double tau = 2.0 / utu;

        // Apply H = I - tau*u*u^T symmetrically: T <- H T H
        // Left: T[k+1:, :] -= tau * u * (u^T T[k+1:, :])
        for (std::size_t j = 0; j < n; ++j) {
            double d = 0.0;
            for (std::size_t i = 0; i < p; ++i) d += u[i] * T(k + 1 + i, j);
            const double coeff = tau * d;
            for (std::size_t i = 0; i < p; ++i) T(k + 1 + i, j) -= coeff * u[i];
        }

        // Right: T[:, k+1:] -= tau * (T[:, k+1:] u) * u^T
        for (std::size_t j = 0; j < n; ++j) {
            double d = 0.0;
            for (std::size_t i = 0; i < p; ++i) d += T(j, k + 1 + i) * u[i];
            const double coeff = tau * d;
            for (std::size_t i = 0; i < p; ++i) T(j, k + 1 + i) -= coeff * u[i];
        }

        // Explicitly zero sub-subdiagonal entries for numerical cleanliness.
        for (std::size_t i = 1; i < p; ++i) {
            T(k + 1 + i, k) = 0.0;
            T(k, k + 1 + i) = 0.0;
        }

        // Accumulate Q: Q[:, k+1:] -= tau * (Q[:, k+1:] u) * u^T
        for (std::size_t j = 0; j < n; ++j) {
            double d = 0.0;
            for (std::size_t i = 0; i < p; ++i) d += Q(j, k + 1 + i) * u[i];
            const double coeff = tau * d;
            for (std::size_t i = 0; i < p; ++i) Q(j, k + 1 + i) -= coeff * u[i];
        }
    }

    return TridiagonalizeResult{std::move(T), std::move(Q)};
}

InverseIterationResult eigenvectors_inverse_iteration(const Matrix& A,
                                                       const Vector& eigenvalues,
                                                       InverseIterationOptions opts) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "eigenvectors_inverse_iteration requires a square matrix, got "
            << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }

    const std::size_t n = A.rows();
    const std::size_t k = eigenvalues.size();

    Matrix evecs(n, k);
    Vector residuals(k, 0.0);

    // Starting vector: uniform unit vector.
    Vector v0(n, 1.0 / std::sqrt(static_cast<double>(n)));

    for (std::size_t col = 0; col < k; ++col) {
        double lambda = eigenvalues[col];

        // Build shifted matrix B = A - lambda*I.
        Matrix B = A;
        for (std::size_t i = 0; i < n; ++i) B(i, i) -= lambda;

        // Try to factor; if near-singular (either at factorization or solve time), perturb the shift.
        auto make_lu = [&]() -> LUResult {
            try {
                LUResult result = lu_factor(B, 1e-14);
                lu_solve(result, v0);  // probe: backward_substitution validates U diagonal
                return result;
            } catch (const SingularMatrixError&) {
                lambda += 1e-7;
                for (std::size_t i = 0; i < n; ++i) B(i, i) += 1e-7;
                return lu_factor(B, 1e-14);
            }
        };
        LUResult lu_b = make_lu();

        Vector v = v0;

        bool converged = false;
        for (int iter = 0; iter < opts.max_iterations; ++iter) {
            Vector w = lu_solve(lu_b, v);

            // Normalize.
            double w_norm = norm2(w);
            if (w_norm == 0.0) break;
            v = w / w_norm;

            // Residual: ||A v - lambda v||.
            const Vector Av = A * v;
            double res = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = Av[i] - eigenvalues[col] * v[i];
                res += d * d;
            }
            res = std::sqrt(res);

            if (res < opts.tolerance) {
                converged = true;
                residuals[col] = res;
                break;
            }
        }

        if (!converged) {
            std::ostringstream oss;
            oss << "eigenvectors_inverse_iteration: did not converge for eigenvalue "
                << col << " (lambda = " << eigenvalues[col] << ") in "
                << opts.max_iterations << " iterations";
            throw NonConvergenceError(oss.str());
        }

        for (std::size_t i = 0; i < n; ++i) evecs(i, col) = v[i];
    }

    return InverseIterationResult{std::move(evecs), std::move(residuals)};
}

}  // namespace linalgebra
