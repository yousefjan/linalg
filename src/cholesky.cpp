export module linalgebra:cholesky;
import std;
import :error;
import :vector;
import :matrix;
import :triangular_solve;

export namespace linalgebra {

struct CholeskyResult {
    Matrix L;
};

CholeskyResult cholesky_factor(const Matrix& A, double tolerance = 1e-12);

Vector cholesky_solve(const CholeskyResult& chol, const Vector& b);

}  // namespace linalgebra

namespace linalgebra {

CholeskyResult cholesky_factor(const Matrix& A, double tolerance) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "cholesky_factor requires a square matrix, got " << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }

    const std::size_t n = A.rows();

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::abs(A(i, j) - A(j, i)) > tolerance) {
                throw LinAlgError("cholesky_factor requires a symmetric matrix");
            }
        }
    }

    Matrix L = Matrix::zeros(n, n);

    for (std::size_t j = 0; j < n; ++j) {
        double sum = A(j, j);
        for (std::size_t k = 0; k < j; ++k) {
            sum -= L(j, k) * L(j, k);
        }

        if (sum <= tolerance) {
            std::ostringstream oss;
            oss << "cholesky_factor: matrix is not positive definite (diagonal became "
                << sum << " at step " << j << ")";
            throw LinAlgError(oss.str());
        }

        L(j, j) = std::sqrt(sum);

        for (std::size_t i = j + 1; i < n; ++i) {
            double s = A(i, j);
            for (std::size_t k = 0; k < j; ++k) {
                s -= L(i, k) * L(j, k);
            }
            L(i, j) = s / L(j, j);
        }
    }

    return CholeskyResult{std::move(L)};
}

Vector cholesky_solve(const CholeskyResult& chol, const Vector& b) {
    const std::size_t n = chol.L.rows();

    if (b.size() != n) {
        std::ostringstream oss;
        oss << "cholesky_solve: rhs size " << b.size()
            << " does not match factorization size " << n;
        throw DimensionMismatchError(oss.str());
    }

    const Vector y = forward_substitution(chol.L, b);
    return backward_substitution(transpose(chol.L), y);
}

}  // namespace linalgebra
