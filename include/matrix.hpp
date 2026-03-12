#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>
#include "vector.hpp"

namespace linalg {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols);
    Matrix(std::size_t rows, std::size_t cols, double value);
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t cols() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

    double& operator()(std::size_t i, std::size_t j);
    const double& operator()(std::size_t i, std::size_t j) const;

    void fill(double value);

    double* data() noexcept;
    const double* data() const noexcept;

    static Matrix identity(std::size_t n);
    static Matrix zeros(std::size_t rows, std::size_t cols);

private:
    [[nodiscard]] std::size_t index(std::size_t i, std::size_t j) const;
    void check_bounds(std::size_t i, std::size_t j) const;

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

Matrix transpose(const Matrix& matrix);
Matrix operator+(const Matrix& lhs, const Matrix& rhs);
Matrix operator-(const Matrix& lhs, const Matrix& rhs);
Vector operator*(const Matrix& matrix, const Vector& vector);
Matrix operator*(const Matrix& lhs, const Matrix& rhs);

}  // namespace linalg

