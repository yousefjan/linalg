#include "matrix.hpp"
#include "linalg_error.hpp"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>

#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && \
    (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#include <immintrin.h>
#endif

#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_FP64_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif

namespace linalg {

namespace {

void check_same_shape(const Matrix& lhs, const Matrix& rhs, const char* operation) {
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
        std::ostringstream oss;
        oss << operation << " requires equal matrix shapes, got " << lhs.rows() << "x" << lhs.cols() << " and " << rhs.rows() << "x" << rhs.cols();
        throw DimensionMismatchError(oss.str());
    }
}

double dot_product_scalar(const double* lhs, const double* rhs, std::size_t count) {
    double sum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__AVX512F__)
double horizontal_sum(__m512d values) {
    alignas(64) double lanes[8];
    _mm512_store_pd(lanes, values);
    double sum = 0.0;
    for (double lane : lanes) {
        sum += lane;
    }
    return sum;
}

double dot_product_avx512(const double* lhs, const double* rhs, std::size_t count) {
    std::size_t i = 0;
    __m512d acc0 = _mm512_setzero_pd();
    __m512d acc1 = _mm512_setzero_pd();

    for (; i + 15 < count; i += 16) {
        const __m512d lhs0 = _mm512_loadu_pd(lhs + i);
        const __m512d rhs0 = _mm512_loadu_pd(rhs + i);
        const __m512d lhs1 = _mm512_loadu_pd(lhs + i + 8);
        const __m512d rhs1 = _mm512_loadu_pd(rhs + i + 8);

        acc0 = _mm512_add_pd(acc0, _mm512_mul_pd(lhs0, rhs0));
        acc1 = _mm512_add_pd(acc1, _mm512_mul_pd(lhs1, rhs1));
    }

    return horizontal_sum(acc0) + horizontal_sum(acc1) +
           dot_product_scalar(lhs + i, rhs + i, count - i);
}
#endif

#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__AVX2__)
double horizontal_sum(__m256d values) {
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, values);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

double dot_product_avx2(const double* lhs, const double* rhs, std::size_t count) {
    std::size_t i = 0;
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    for (; i + 7 < count; i += 8) {
        const __m256d lhs0 = _mm256_loadu_pd(lhs + i);
        const __m256d rhs0 = _mm256_loadu_pd(rhs + i);
        const __m256d lhs1 = _mm256_loadu_pd(lhs + i + 4);
        const __m256d rhs1 = _mm256_loadu_pd(rhs + i + 4);

        acc0 = _mm256_add_pd(acc0, _mm256_mul_pd(lhs0, rhs0));
        acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(lhs1, rhs1));
    }

    return horizontal_sum(acc0) + horizontal_sum(acc1) +
           dot_product_scalar(lhs + i, rhs + i, count - i);
}
#endif

#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__AVX__) && !defined(__AVX2__)
double horizontal_sum(__m256d values) {
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, values);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

double dot_product_avx(const double* lhs, const double* rhs, std::size_t count) {
    std::size_t i = 0;
    __m256d acc = _mm256_setzero_pd();

    for (; i + 3 < count; i += 4) {
        const __m256d lhs_values = _mm256_loadu_pd(lhs + i);
        const __m256d rhs_values = _mm256_loadu_pd(rhs + i);
        acc = _mm256_add_pd(acc, _mm256_mul_pd(lhs_values, rhs_values));
    }

    return horizontal_sum(acc) + dot_product_scalar(lhs + i, rhs + i, count - i);
}
#endif

#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_FP64_VECTOR_ARITHMETIC)
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
#if !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__AVX512F__)
    return dot_product_avx512(lhs, rhs, count);
#elif !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__AVX2__)
    return dot_product_avx2(lhs, rhs, count);
#elif !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__AVX__) && !defined(__AVX2__)
    return dot_product_avx(lhs, rhs, count);
#elif !defined(LINEAR_ALGEBRA_FORCE_SCALAR_MATMUL) && defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_FP64_VECTOR_ARITHMETIC)
    return dot_product_neon(lhs, rhs, count);
#else
    return dot_product_scalar(lhs, rhs, count);
#endif
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

}  // namespace linalg
