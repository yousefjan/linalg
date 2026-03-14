#pragma once

#include <cstddef>

#include "matrix.hpp"
#include "vector.hpp"

namespace linalg {

Vector forward_substitution(
    const Matrix& lower,
    const Vector& rhs,
    double singular_tolerance = 1e-12,
    bool unit_diagonal = false);

Vector backward_substitution(
    const Matrix& upper,
    const Vector& rhs,
    double singular_tolerance = 1e-12,
    bool unit_diagonal = false);

}  // namespace linalg
