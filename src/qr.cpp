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

struct QRColPivResult {
    Matrix Q;
    Matrix R;
    std::vector<std::size_t> perm;
    std::size_t rank;
};

// Householder QR with column pivoting (rank-revealing).
// At each step, the column with largest remaining norm is selected as pivot.
// The numerical rank is determined by comparing diagonal entries of R to
// rank_tolerance * |R(0,0)|.
//
// Returns Q (m x n), R (n x n upper triangular), perm (column permutation),
// and rank (numerical rank estimate).
//
// Throws DimensionMismatchError if rows < cols.
QRColPivResult qr_colpiv(const Matrix& A, double rank_tolerance = 1e-12);

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

QRColPivResult qr_colpiv(const Matrix& A, double rank_tolerance) {
    require_tall(A, "qr_colpiv");
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();

    Matrix work = A;
    Matrix Q_full = Matrix::identity(m);

    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), std::size_t{0});

    // Precompute column norms squared.
    std::vector<double> col_norms_sq(n);
    for (std::size_t j = 0; j < n; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < m; ++i) s += work(i, j) * work(i, j);
        col_norms_sq[j] = s;
    }

    std::size_t rank = n;

    for (std::size_t k = 0; k < n; ++k) {
        // Find pivot: column with largest remaining norm.
        std::size_t pivot = k;
        double max_norm = col_norms_sq[k];
        for (std::size_t j = k + 1; j < n; ++j) {
            if (col_norms_sq[j] > max_norm) {
                max_norm = col_norms_sq[j];
                pivot = j;
            }
        }

        // Swap columns k and pivot.
        if (pivot != k) {
            for (std::size_t i = 0; i < m; ++i) {
                std::swap(work(i, k), work(i, pivot));
            }
            std::swap(col_norms_sq[k], col_norms_sq[pivot]);
            std::swap(perm[k], perm[pivot]);
        }

        // Householder reflector for column k.
        const std::size_t p = m - k;

        std::vector<double> u(p);
        for (std::size_t i = 0; i < p; ++i) u[i] = work(k + i, k);

        const double x_norm = [&] {
            double s = 0.0;
            for (double v : u) s += v * v;
            return std::sqrt(s);
        }();

        if (x_norm == 0.0) {
            // Remaining columns are zero — rank determined.
            rank = k;
            break;
        }

        // Check rank: if this pivot norm is small relative to R(0,0).
        if (k > 0) {
            const double r00 = std::abs(work(0, 0));
            if (x_norm <= rank_tolerance * r00) {
                rank = k;
                break;
            }
        }

        const double sigma = (u[0] >= 0.0 ? 1.0 : -1.0) * x_norm;
        u[0] += sigma;

        const double utu = [&] {
            double s = 0.0;
            for (double v : u) s += v * v;
            return s;
        }();
        const double tau = 2.0 / utu;

        // Apply reflector to work columns k..n-1.
        for (std::size_t j = k; j < n; ++j) {
            double d = 0.0;
            for (std::size_t i = 0; i < p; ++i) d += u[i] * work(k + i, j);
            const double coeff = tau * d;
            for (std::size_t i = 0; i < p; ++i) work(k + i, j) -= coeff * u[i];
        }

        // Apply reflector to Q_full.
        for (std::size_t j = 0; j < m; ++j) {
            double d = 0.0;
            for (std::size_t i = 0; i < p; ++i) d += u[i] * Q_full(k + i, j);
            const double coeff = tau * d;
            for (std::size_t i = 0; i < p; ++i) Q_full(k + i, j) -= coeff * u[i];
        }

        // Update column norms (downdate).
        for (std::size_t j = k + 1; j < n; ++j) {
            const double val = work(k, j);
            col_norms_sq[j] -= val * val;
            if (col_norms_sq[j] < 0.0) col_norms_sq[j] = 0.0;
        }
    }

    // Extract Q (m x n) and R (n x n).
    Matrix Q(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Q(i, j) = Q_full(j, i);

    Matrix R(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            R(i, j) = work(i, j);

    return QRColPivResult{std::move(Q), std::move(R), std::move(perm), rank};
}

}  // namespace linalgebra
