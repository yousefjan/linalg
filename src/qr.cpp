export module linalgebra:qr;
import std;
import :error;
import :vector;
import :matrix;

export namespace linalgebra {

struct QRResult {
    Matrix Q;
    Matrix R;
};

// Classical Gram-Schmidt.
// Mathematically natural but numerically fragile: orthogonality of Q
// degrades rapidly on ill-conditioned inputs.
// Provided for comparison — prefer modified_gs or householder in practice.
//
// Throws DimensionMismatchError if rows < cols.
// Throws SingularMatrixError   if a column is (nearly) linearly dependent.
QRResult qr_classical_gs(const Matrix& A, double zero_tolerance = 1e-14);

// Modified Gram-Schmidt.
// Subtracts each projection immediately on the running vector rather than
// on the original column.  Algebraically equivalent to classical GS but
// numerically much better — round-off stays local instead of accumulating.
//
// Same exceptions as classical GS.
QRResult qr_modified_gs(const Matrix& A, double zero_tolerance = 1e-14);

// Householder QR.
// Applies a sequence of orthogonal reflections to zero out below-diagonal
// entries column by column.  Backward-stable and the standard choice for
// dense QR.  Works correctly on rank-deficient matrices (zero pivots
// produce zero diagonal entries in R without throwing).
//
// Throws DimensionMismatchError if rows < cols.
QRResult qr_householder(const Matrix& A);

}  // namespace linalgebra

namespace {

void require_tall(const linalgebra::Matrix& A, const char* name) {
    if (A.rows() < A.cols()) {
        std::ostringstream oss;
        oss << name << " requires rows >= cols, got " << A.rows() << "x" << A.cols();
        throw linalgebra::DimensionMismatchError(oss.str());
    }
}

double col_norm(const linalgebra::Matrix& M, std::size_t j) {
    double s = 0.0;
    for (std::size_t i = 0; i < M.rows(); ++i) {
        s += M(i, j) * M(i, j);
    }
    return std::sqrt(s);
}

double col_dot(const linalgebra::Matrix& M, std::size_t j,
               const linalgebra::Matrix& N, std::size_t k) {
    double s = 0.0;
    for (std::size_t i = 0; i < M.rows(); ++i) {
        s += M(i, j) * N(i, k);
    }
    return s;
}

}  // namespace

namespace linalgebra {

QRResult qr_classical_gs(const Matrix& A, double zero_tolerance) {
    require_tall(A, "qr_classical_gs");
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    Matrix Q = Matrix::zeros(m, n);
    Matrix R = Matrix::zeros(n, n);

    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) Q(i, j) = A(i, j);

        for (std::size_t k = 0; k < j; ++k) {
            R(k, j) = col_dot(A, j, Q, k);
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

QRResult qr_modified_gs(const Matrix& A, double zero_tolerance) {
    require_tall(A, "qr_modified_gs");
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    Matrix Q = Matrix::zeros(m, n);
    Matrix R = Matrix::zeros(n, n);

    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) Q(i, j) = A(i, j);

        for (std::size_t k = 0; k < j; ++k) {
            R(k, j) = col_dot(Q, j, Q, k);
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

QRResult qr_householder(const Matrix& A) {
    require_tall(A, "qr_householder");
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    Matrix work = A;
    Matrix Q_full = Matrix::identity(m);

    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t p = m - k;

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

        for (std::size_t j = k; j < n; ++j) {
            double d = 0.0;
            for (std::size_t i = 0; i < p; ++i) d += u[i] * work(k + i, j);
            const double coeff = tau * d;
            for (std::size_t i = 0; i < p; ++i) work(k + i, j) -= coeff * u[i];
        }

        for (std::size_t j = 0; j < m; ++j) {
            double d = 0.0;
            for (std::size_t i = 0; i < p; ++i) d += u[i] * Q_full(k + i, j);
            const double coeff = tau * d;
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

}  // namespace linalgebra
