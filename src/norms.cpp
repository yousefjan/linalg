#include "norms.hpp"

#include <cmath>

namespace linalg {

double norm2(const Vector& vector) { return std::sqrt(dot(vector, vector)); }

}  // namespace linalg
