#include "matrix.hpp"

#include <algorithm>
#include <stdexcept>

namespace linalg {

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values) : rows_(values.size()) {
    if (rows_ == 0) {
        cols_ = 0;
        return;
    }

    cols_ = values.begin()->size();
    data_.reserve(rows_ * cols_);

    for (const auto& row : values) {
        if (row.size() != cols_) {
            throw std::invalid_argument("Matrix initializer rows must have equal length");
        }
        data_.insert(data_.end(), row.begin(), row.end());
    }
}


std::size_t Matrix::rows() const noexcept { return rows_; }

std::size_t Matrix::cols() const noexcept { return cols_; }

bool Matrix::empty() const noexcept { return data_.empty(); }

double& Matrix::operator()(std::size_t i, std::size_t j) {
    check_bounds(i, j);
    return data_[index(i, j)];
}

const double& Matrix::operator()(std::size_t i, std::size_t j) const {
    check_bounds(i, j);
    return data_[index(i, j)];
}

void Matrix::fill(double value) { std::fill(data_.begin(), data_.end(), value); }

double* Matrix::data() noexcept { return data_.data(); }

const double* Matrix::data() const noexcept { return data_.data(); }

Matrix Matrix::identity(std::size_t n) {
    Matrix result(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) { return Matrix(rows, cols, 0.0); }

std::size_t Matrix::index(std::size_t i, std::size_t j) const { return i * cols_ + j; }

void Matrix::check_bounds(std::size_t i, std::size_t j) const {
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix index out of bounds");
    }
}

}  // namespace linalg
