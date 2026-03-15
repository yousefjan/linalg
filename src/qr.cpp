#include "qr.hpp"

#include <cmath>
#include <sstream>

#include "linalg_error.hpp"

namespace linalg {

namespace {

void require_tall(const Matrix& A, const char* name) {
    if (A.rows() < A.cols()) {
        std::ostringstream oss;
        oss << name << " requires rows >= cols, got " << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }
}

// ||column j of M||_2
double col_norm(const Matrix& M, std::size_t j) {
    double s = 0.0;
    for (std::size_t i = 0; i < M.rows(); ++i) {
        s += M(i, j) * M(i, j);
    }
    return std::sqrt(s);
}

// dot product of column j of M with column k of N (same number of rows)
double col_dot(const Matrix& M, std::size_t j, const Matrix& N, std::size_t k) {
    double s = 0.0;
    for (std::size_t i = 0; i < M.rows(); ++i) {
        s += M(i, j) * N(i, k);
    }
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Classical Gram-Schmidt
// ---------------------------------------------------------------------------
//
// For column j:
//   R[i][j] = <a_j, q_i>   for i < j   
//   v       = a_j - sum_i R[i][j] * q_i
//   R[j][j] = ||v||
//   q_j     = v / R[j][j]
//

QRResult qr_classical_gs(const Matrix& A, double zero_tolerance) {
    require_tall(A, "qr_classical_gs");
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    Matrix Q = Matrix::zeros(m, n);
    Matrix R = Matrix::zeros(n, n);

    for (std::size_t j = 0; j < n; ++j) {
        // Start with column j of A.
        for (std::size_t i = 0; i < m; ++i) Q(i, j) = A(i, j);

        // Project out existing basis vectors using the *original* A column.
        for (std::size_t k = 0; k < j; ++k) {
            R(k, j) = col_dot(A, j, Q, k);       // <a_j, q_k>
            for (std::size_t i = 0; i < m; ++i) {
                Q(i, j) -= R(k, j) * Q(i, k);
            }
        }

        const double norm = col_norm(Q, j);
        if (norm <= zero_tolerance) {
            std::ostringstream oss;
            oss << "qr_classical_gs: column " << j
                << " is (nearly) linearly dependent (norm = " << norm << ")";
            throw SingularMatrixError(oss.str());
        }
        R(j, j) = norm;
        for (std::size_t i = 0; i < m; ++i) Q(i, j) /= norm;
    }

    return QRResult{std::move(Q), std::move(R)};
}

// ---------------------------------------------------------------------------
// Modified Gram-Schmidt
// ---------------------------------------------------------------------------
//
// For column j:
//   v = a_j
//   For k = 0 .. j-1:
//     R[k][j] = <v, q_k> 
//     v       = v - R[k][j] * q_k
//   R[j][j] = ||v||
//   q_j     = v / R[j][j]
//
// Each subtraction uses the already-updated v, so round-off is re-corrected
// at every sub-step rather than compounding into one subtraction.

QRResult qr_modified_gs(const Matrix& A, double zero_tolerance) {
    require_tall(A, "qr_modified_gs");
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    Matrix Q = Matrix::zeros(m, n);
    Matrix R = Matrix::zeros(n, n);

    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) Q(i, j) = A(i, j);

        for (std::size_t k = 0; k < j; ++k) {
            R(k, j) = col_dot(Q, j, Q, k);       // <v_running, q_k>
            for (std::size_t i = 0; i < m; ++i) {
                Q(i, j) -= R(k, j) * Q(i, k);
            }
        }

        const double norm = col_norm(Q, j);
        if (norm <= zero_tolerance) {
            std::ostringstream oss;
            oss << "qr_modified_gs: column " << j
                << " is (nearly) linearly dependent (norm = " << norm << ")";
            throw SingularMatrixError(oss.str());
        }
        R(j, j) = norm;
        for (std::size_t i = 0; i < m; ++i) Q(i, j) /= norm;
    }

    return QRResult{std::move(Q), std::move(R)};
}

// ---------------------------------------------------------------------------
// Householder QR
// ---------------------------------------------------------------------------
//
// At step k, build a Householder reflector H_k that maps R[k:, k] to
// -sign(R[k,k]) * ||R[k:,k]|| * e_1.
//
// H = I - (2 / (u^T u)) * u * u^T
//   where u = x + sign(x_0) * ||x|| * e_1   (sign chosen to avoid cancellation)
//
// H is never formed explicitly.  It is applied via the rank-1 update:
//   M[k:, :] -= u * (2/(u^T u) * (u^T M[k:, :]))
//
// After n reflections, the working copy of A has become R (upper triangular).
// Q is accumulated by applying each H_k to an identity matrix from the left.
// The thin Q (m x n) is the first n columns of the full m x m orthogonal Q.

QRResult qr_householder(const Matrix& A) {
    require_tall(A, "qr_householder");
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    // Will become R.
    Matrix work = A;

    // Q accumulated as full m x m orthogonal matrix; trim to m x n.
    Matrix Q_full = Matrix::identity(m);

    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t p = m - k;  // length of the subvector

        // Build Householder vector u from the subcolumn work[k:, k].
        std::vector<double> u(p);
        for (std::size_t i = 0; i < p; ++i) u[i] = work(k + i, k);

        const double x_norm = [&] {
            double s = 0.0;
            for (double v : u) s += v * v;
            return std::sqrt(s);
        }();

        if (x_norm == 0.0) continue;

        // sigma = sign(u[0]) * ||x||
        const double sigma = (u[0] >= 0.0 ? 1.0 : -1.0) * x_norm;
        u[0] += sigma;

        const double utu = [&] {
            double s = 0.0;
            for (double v : u) s += v * v;
            return s;
        }();
        const double tau = 2.0 / utu;

        // Apply H_k to work[k:, k:n] 
        for (std::size_t j = k; j < n; ++j) {
            double dot = 0.0;
            for (std::size_t i = 0; i < p; ++i) dot += u[i] * work(k + i, j);
            const double coeff = tau * dot;
            for (std::size_t i = 0; i < p; ++i) work(k + i, j) -= coeff * u[i];
        }

        // Apply H_k to Q_full[k:, 0:m] 
        for (std::size_t j = 0; j < m; ++j) {
            double dot = 0.0;
            for (std::size_t i = 0; i < p; ++i) dot += u[i] * Q_full(k + i, j);
            const double coeff = tau * dot;
            for (std::size_t i = 0; i < p; ++i) Q_full(k + i, j) -= coeff * u[i];
        }
    }

    // Thin Q: first n columns of Q_full^T
    Matrix Q(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Q(i, j) = Q_full(j, i);

    // Thin R: first n rows of work
    Matrix R(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            R(i, j) = work(i, j);

    return QRResult{std::move(Q), std::move(R)};
}

}  // namespace linalg
