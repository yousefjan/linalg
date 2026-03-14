#pragma once

#include <cstddef>
#include <vector>

#include "matrix.hpp"
#include "vector.hpp"

namespace linalg {

// Result of LU factorization with partial pivoting.
//
// The factorization satisfies PA = LU, where:
//   P  is the permutation matrix encoded by `perm`
//   L  is unit lower triangular  (L[i][i] == 1)
//   U  is upper triangular
//
// `perm[i]` = index of the original row that ended up at position i.
// Applying P to a vector b means: (Pb)[i] = b[perm[i]].
//
// `sign` is the sign of the permutation: +1 if an even number of row
// swaps were made, -1 if odd.  Useful for computing det(A) = sign * prod(diag(U)).
struct LUResult {
    Matrix L;
    Matrix U;
    std::vector<std::size_t> perm;
    int sign;
};

// Compute the LU factorization of A with partial pivoting.
//
// Throws DimensionMismatchError if A is not square.
// Throws SingularMatrixError   if A is (numerically) singular, i.e. any
//   pivot is smaller in magnitude than `singular_tolerance`.
LUResult lu_factor(const Matrix& A, double singular_tolerance = 1e-12);

// Solve Ax = b given a precomputed LU factorization.
//
// Applies the stored permutation, then forward / backward substitution.
// Throws DimensionMismatchError if b.size() != lu.L.rows().
Vector lu_solve(const LUResult& lu, const Vector& b);

}  // namespace linalg
