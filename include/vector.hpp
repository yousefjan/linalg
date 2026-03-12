#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace linalg {

class Vector {
public:
    Vector() = default;
    explicit Vector(std::size_t n);
    Vector(std::size_t n, double value);
    Vector(std::initializer_list<double> values);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    double& operator[](std::size_t i);
    const double& operator[](std::size_t i) const;

    void fill(double value);

    double* data() noexcept;
    const double* data() const noexcept;

    auto begin() noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end() const noexcept { return data_.end(); }
    auto cbegin() const noexcept { return data_.cbegin(); }
    auto cend() const noexcept { return data_.cend(); }

private:
    void check_index(std::size_t i) const;

    std::vector<double> data_;
};

Vector operator+(const Vector& lhs, const Vector& rhs);
Vector operator-(const Vector& lhs, const Vector& rhs);
Vector operator*(const Vector& v, double scalar);
Vector operator*(double scalar, const Vector& v);
Vector operator/(const Vector& v, double scalar);
double dot(const Vector& lhs, const Vector& rhs);

}  // namespace linalg
