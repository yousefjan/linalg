#pragma once

#include "matrix.hpp"
#include "vector.hpp"

namespace linalg {

// Thin QR factorization result.
//
// For an m x n matrix A with m >= n:
//   Q  is m x n with orthonormal columns  (Q^T Q = I_n)
//   R  is n x n upper triangular
//   A  = Q * R
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

}  // namespace linalg
