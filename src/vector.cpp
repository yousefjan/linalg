#include "vector.hpp"
#include "linalg_error.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace linalg {

namespace {

void check_same_size(const Vector& lhs, const Vector& rhs, const char* operation) {
    if (lhs.size() != rhs.size()) {
        std::ostringstream oss;
        oss << operation << " requires equal vector sizes, got " << lhs.size() << " and "
            << rhs.size();
        throw DimensionMismatchError(oss.str());
    }
}

}  // namespace

Vector::Vector(std::size_t n) : data_(n) {}

Vector::Vector(std::size_t n, double value) : data_(n, value) {}

Vector::Vector(std::initializer_list<double> values) : data_(values) {}

std::size_t Vector::size() const noexcept { return data_.size(); }

bool Vector::empty() const noexcept { return data_.empty(); }

double& Vector::operator[](std::size_t i) {
    check_index(i);
    return data_[i];
}

const double& Vector::operator[](std::size_t i) const {
    check_index(i);
    return data_[i];
}

void Vector::fill(double value) { std::fill(data_.begin(), data_.end(), value); }

double* Vector::data() noexcept { return data_.data(); }

const double* Vector::data() const noexcept { return data_.data(); }

void Vector::check_index(std::size_t i) const {
    if (i >= data_.size()) {
        throw std::out_of_range("Vector index out of range");
    }
}

Vector operator+(const Vector& lhs, const Vector& rhs) {
    check_same_size(lhs, rhs, "Vector addition");

    Vector result(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

Vector operator-(const Vector& lhs, const Vector& rhs) {
    check_same_size(lhs, rhs, "Vector subtraction");

    Vector result(lhs.size());
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] - rhs[i];
    }
    return result;
}

Vector operator*(const Vector& v, double scalar) {
    Vector result(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}

Vector operator*(double scalar, const Vector& v) { return v * scalar; }

Vector operator/(const Vector& v, double scalar) {
    if (scalar == 0.0) {
        throw std::invalid_argument("Vector scalar division requires a nonzero scalar");
    }

    Vector result(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] / scalar;
    }
    return result;
}

double dot(const Vector& lhs, const Vector& rhs) {
    check_same_size(lhs, rhs, "Dot product");

    return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), 0.0);
}

}  // namespace linalg
