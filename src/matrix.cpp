#include "matrix.hpp"
#include "linalg_error.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace linalg {

namespace {

void check_same_shape(const Matrix& lhs, const Matrix& rhs, const char* operation) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        std::ostringstream oss;
        oss << operation << " requires equal matrix shapes, got " << lhs.rows() << "x" << lhs.cols() << " and " << rhs.rows() << "x" << rhs.cols();
        throw DimensionMismatchError(oss.str());
    }
}

}  // namespace

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

Matrix transpose(const Matrix& matrix) {
    Matrix result(matrix.cols(), matrix.rows());
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        for (std::size_t j = 0; j < matrix.cols(); ++j) {
            result(j, i) = matrix(i, j);
        }
    }
    return result;
}

Matrix operator+(const Matrix& lhs, const Matrix& rhs) {
    check_same_shape(lhs, rhs, "Matrix addition");

    Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        for (std::size_t j = 0; j < lhs.cols(); ++j) {
            result(i, j) = lhs(i, j) + rhs(i, j);
        }
    }
    return result;
}

Matrix operator-(const Matrix& lhs, const Matrix& rhs) {
    check_same_shape(lhs, rhs, "Matrix subtraction");

    Matrix result(lhs.rows(), lhs.cols());
    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        for (std::size_t j = 0; j < lhs.cols(); ++j) {
            result(i, j) = lhs(i, j) - rhs(i, j);
        }
    }
    return result;
}

Vector operator*(const Matrix& matrix, const Vector& vector) {
    if (matrix.cols() != vector.size()) {
        std::ostringstream oss;
        oss << "Matrix-vector multiplication requires matrix columns to match vector size, got "
            << matrix.cols() << " and " << vector.size();
        throw DimensionMismatchError(oss.str());
    }

    Vector result(matrix.rows());
    for (std::size_t i = 0; i < matrix.rows(); ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < matrix.cols(); ++j) {
            sum += matrix(i, j) * vector[j];
        }
        result[i] = sum;
    }
    return result;
}

Matrix operator*(const Matrix& lhs, const Matrix& rhs) {
    if (lhs.cols() != rhs.rows()) {
        std::ostringstream oss;
        oss << "Matrix multiplication requires lhs.cols() == rhs.rows(), got " << lhs.cols()
            << " and " << rhs.rows();
        throw DimensionMismatchError(oss.str());
    }

    Matrix result(lhs.rows(), rhs.cols());
    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        for (std::size_t k = 0; k < lhs.cols(); ++k) {
            const double lhs_ik = lhs(i, k);
            for (std::size_t j = 0; j < rhs.cols(); ++j) {
                result(i, j) += lhs_ik * rhs(k, j);
            }
        }
    }
    return result;
}

}  // namespace linalg
