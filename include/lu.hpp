#pragma once

#include <cstddef>
#include <vector>

#include "matrix.hpp"
#include "vector.hpp"

namespace linalg {

struct LUResult {
    Matrix L;
    Matrix U;
    std::vector<std::size_t> perm;
    int sign;
};

LUResult lu_factor(const Matrix& A, double singular_tolerance = 1e-12);

Vector lu_solve(const LUResult& lu, const Vector& b);

}  // namespace linalg
