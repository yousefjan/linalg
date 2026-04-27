module;

#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

export module linalgebra:matrix;
import std;
import :error;
import :vector;

export namespace linalgebra {

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

}  // namespace linalgebra

namespace {

void check_same_shape(const linalgebra::Matrix& lhs, const linalgebra::Matrix& rhs,
                      const char* operation) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        std::ostringstream oss;
        oss << operation << " requires equal matrix shapes, got " << lhs.rows() << "x"
            << lhs.cols() << " and " << rhs.rows() << "x" << rhs.cols();
        throw linalgebra::DimensionMismatchError(oss.str());
    }
}

double dot_product_scalar(const double* lhs, const double* rhs, std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__ARM_NEON)
double horizontal_sum(float64x2_t values) {
    return vgetq_lane_f64(values, 0) + vgetq_lane_f64(values, 1);
}

double dot_product_neon(const double* lhs, const double* rhs, std::size_t count) {
    std::size_t i = 0;
    float64x2_t acc0 = vdupq_n_f64(0.0);
    float64x2_t acc1 = vdupq_n_f64(0.0);

    for (; i + 3 < count; i += 4) {
        const float64x2_t lhs0 = vld1q_f64(lhs + i);
        const float64x2_t rhs0 = vld1q_f64(rhs + i);
        const float64x2_t lhs1 = vld1q_f64(lhs + i + 2);
        const float64x2_t rhs1 = vld1q_f64(rhs + i + 2);

        acc0 = vaddq_f64(acc0, vmulq_f64(lhs0, rhs0));
        acc1 = vaddq_f64(acc1, vmulq_f64(lhs1, rhs1));
    }

    return horizontal_sum(acc0) + horizontal_sum(acc1) +
           dot_product_scalar(lhs + i, rhs + i, count - i);
}
#endif

double dot_product_simd(const double* lhs, const double* rhs, std::size_t count) {
#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__ARM_NEON)
    return dot_product_neon(lhs, rhs, count);
#else
    return dot_product_scalar(lhs, rhs, count);
#endif
}

}  // namespace

namespace linalgebra {

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values)
    : rows_(values.size()) {
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

    const Matrix rhs_transposed = transpose(rhs);
    Matrix result(lhs.rows(), rhs.cols());

    const std::size_t inner_dim = lhs.cols();
    const double* lhs_data = lhs.data();
    const double* rhs_t_data = rhs_transposed.data();
    double* result_data = result.data();

    for (std::size_t i = 0; i < lhs.rows(); ++i) {
        const double* lhs_row = lhs_data + i * inner_dim;
        for (std::size_t j = 0; j < rhs.cols(); ++j) {
            const double* rhs_column = rhs_t_data + j * inner_dim;
            result_data[i * rhs.cols() + j] = dot_product_simd(lhs_row, rhs_column, inner_dim);
        }
    }
    return result;
}

}  // namespace linalgebra
