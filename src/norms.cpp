export module linalgebra:norms;
import std;
import :vector;

export namespace linalgebra {

double norm2(const Vector& vector);

}  // namespace linalgebra

namespace linalgebra {

double norm2(const Vector& vector) { return std::sqrt(dot(vector, vector)); }

}  // namespace linalgebra
