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

double col_norm(const Matrix& M, std::size_t j) {
    double s = 0.0;
    for (std::size_t i = 0; i < M.rows(); ++i) {
        s += M(i, j) * M(i, j);
    }
    return std::sqrt(s);
}

double col_dot(const Matrix& M, std::size_t j, const Matrix& N, std::size_t k) {
    double s = 0.0;
    for (std::size_t i = 0; i < M.rows(); ++i) {
        s += M(i, j) * N(i, k);
    }
    return s;
}

}  // namespace

// --- Gram-Schmidt ---

QRResult qr_classical_gs(const Matrix& A, double zero_tolerance) {
    require_tall(A, "qr_classical_gs");
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    Matrix Q = Matrix::zeros(m, n);
    Matrix R = Matrix::zeros(n, n);

    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) Q(i, j) = A(i, j);

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

// --- Modified Gram-Schmidt ---

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

// --- Householder QR ---

QRResult qr_householder(const Matrix& A) {
    require_tall(A, "qr_householder");
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    // Will become R.
    Matrix work = A;

    Matrix Q_full = Matrix::identity(m);

    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t p = m - k;  // length of the subvector

        std::vector<double> u(p);
        for (std::size_t i = 0; i < p; ++i) u[i] = work(k + i, k);

        const double x_norm = [&] {
            double s = 0.0;
            for (double v : u) s += v * v;
            return std::sqrt(s);
        }();

        if (x_norm == 0.0) continue;

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

    Matrix Q(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Q(i, j) = Q_full(j, i);

    Matrix R(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            R(i, j) = work(i, j);

    return QRResult{std::move(Q), std::move(R)};
}

}  // namespace linalg
