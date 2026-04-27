export module linalgebra:triangular_solve;
import std;
import :error;
import :vector;
import :matrix;

export namespace linalgebra {

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

}  // namespace linalgebra

namespace {

void validate_square_system(const linalgebra::Matrix& matrix, const linalgebra::Vector& rhs,
                             const char* operation) {
    if (matrix.rows() != matrix.cols()) {
        std::ostringstream oss;
        oss << operation << " requires a square matrix, got " << matrix.rows() << "x"
            << matrix.cols();
        throw linalgebra::DimensionMismatchError(oss.str());
    }

    if (matrix.rows() != rhs.size()) {
        std::ostringstream oss;
        oss << operation << " requires matrix dimension to match rhs size, got "
            << matrix.rows() << " and " << rhs.size();
        throw linalgebra::DimensionMismatchError(oss.str());
    }
}

void validate_tolerance(double singular_tolerance) {
    if (singular_tolerance < 0.0) {
        throw std::invalid_argument("Singular tolerance must be nonnegative");
    }
}

void validate_lower_triangular(const linalgebra::Matrix& lower, double singular_tolerance,
                                bool unit_diagonal) {
    for (std::size_t i = 0; i < lower.rows(); ++i) {
        for (std::size_t j = i + 1; j < lower.cols(); ++j) {
            if (std::abs(lower(i, j)) > singular_tolerance) {
                throw std::invalid_argument(
                    "Forward substitution requires a lower-triangular matrix");
            }
        }

        if (!unit_diagonal && std::abs(lower(i, i)) <= singular_tolerance) {
            throw linalgebra::SingularMatrixError(
                "Forward substitution encountered a zero or tiny diagonal entry");
        }
    }
}

void validate_upper_triangular(const linalgebra::Matrix& upper, double singular_tolerance,
                                bool unit_diagonal) {
    for (std::size_t i = 0; i < upper.rows(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (std::abs(upper(i, j)) > singular_tolerance) {
                throw std::invalid_argument(
                    "Backward substitution requires an upper-triangular matrix");
            }
        }

        if (!unit_diagonal && std::abs(upper(i, i)) <= singular_tolerance) {
            throw linalgebra::SingularMatrixError(
                "Backward substitution encountered a negligible diagonal entry");
        }
    }
}

}  // namespace

namespace linalgebra {

Vector forward_substitution(const Matrix& lower, const Vector& rhs,
                             double singular_tolerance, bool unit_diagonal) {
    validate_tolerance(singular_tolerance);
    validate_square_system(lower, rhs, "Forward substitution");
    validate_lower_triangular(lower, singular_tolerance, unit_diagonal);

    Vector solution(lower.rows());
    for (std::size_t i = 0; i < lower.rows(); ++i) {
        double sum = rhs[i];
        for (std::size_t j = 0; j < i; ++j) {
            sum -= lower(i, j) * solution[j];
        }

        if (unit_diagonal) {
            solution[i] = sum;
        } else {
            solution[i] = sum / lower(i, i);
        }
    }

    return solution;
}

Vector backward_substitution(const Matrix& upper, const Vector& rhs,
                              double singular_tolerance, bool unit_diagonal) {
    validate_tolerance(singular_tolerance);
    validate_square_system(upper, rhs, "Backward substitution");
    validate_upper_triangular(upper, singular_tolerance, unit_diagonal);

    Vector solution(upper.rows());
    for (std::size_t offset = 0; offset < upper.rows(); ++offset) {
        const std::size_t i = upper.rows() - 1 - offset;
        double sum = rhs[i];
        for (std::size_t j = i + 1; j < upper.cols(); ++j) {
            sum -= upper(i, j) * solution[j];
        }

        if (unit_diagonal) {
            solution[i] = sum;
        } else {
            solution[i] = sum / upper(i, i);
        }
    }

    return solution;
}

}  // namespace linalgebra
