#include "lu.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

#include "linalg_error.hpp"
#include "triangular_solve.hpp"

namespace linalg {

LUResult lu_factor(const Matrix& A, double singular_tolerance) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "lu_factor requires a square matrix, got " << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }

    const std::size_t n = A.rows();

    Matrix work = A;

    Matrix L = Matrix::zeros(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        L(i, i) = 1.0;
    }

    Matrix U = Matrix::zeros(n, n);

    std::vector<std::size_t> perm(n);
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    int sign = 1;

    for (std::size_t k = 0; k < n; ++k) {
        // ---- Partial pivoting: find row with largest magnitude in column k ----
        std::size_t pivot_row = k;
        double max_val = std::abs(work(k, k));
        for (std::size_t i = k + 1; i < n; ++i) {
            const double val = std::abs(work(i, k));
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }

        if (pivot_row != k) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(work(k, j), work(pivot_row, j));
            }
            for (std::size_t j = 0; j < k; ++j) {
                std::swap(L(k, j), L(pivot_row, j));
            }
            std::swap(perm[k], perm[pivot_row]);
            sign = -sign;
        }

        // ---- Singularity check ----
        if (std::abs(work(k, k)) <= singular_tolerance) {
            std::ostringstream oss;
            oss << "lu_factor: near-zero pivot " << work(k, k) << " at step " << k
                << " (tolerance " << singular_tolerance << ")";
            throw SingularMatrixError(oss.str());
        }

        // ---- Record U row k ----
        for (std::size_t j = k; j < n; ++j) {
            U(k, j) = work(k, j);
        }

        // ---- Compute multipliers and eliminate below pivot ----
        for (std::size_t i = k + 1; i < n; ++i) {
            L(i, k) = work(i, k) / work(k, k);
            for (std::size_t j = k + 1; j < n; ++j) {
                work(i, j) -= L(i, k) * work(k, j);
            }
            work(i, k) = 0.0;
        }
    }

    return LUResult{std::move(L), std::move(U), std::move(perm), sign};
}

Vector lu_solve(const LUResult& lu, const Vector& b) {
    const std::size_t n = lu.L.rows();

    if (b.size() != n) {
        std::ostringstream oss;
        oss << "lu_solve: rhs size " << b.size() << " does not match factorization size " << n;
        throw DimensionMismatchError(oss.str());
    }

    Vector pb(n);
    for (std::size_t i = 0; i < n; ++i) {
        pb[i] = b[lu.perm[i]];
    }

    const Vector y = forward_substitution(lu.L, pb, /*singular_tolerance=*/1e-14,
                                          /*unit_diagonal=*/true);

    return backward_substitution(lu.U, y);
}

}  // namespace linalg
