#include "linalg_error.hpp"
#include "matrix.hpp"
#include "norms.hpp"
#include "triangular_solve.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using linalg::DimensionMismatchError;
using linalg::Matrix;
using linalg::SingularMatrixError;
using linalg::Vector;

namespace {

double residual_norm(const Matrix& a, const Vector& x, const Vector& b) {
    return linalg::norm2((a * x) - b);
}

}  // namespace

TEST_CASE("Forward substitution solves lower-triangular systems", "[triangular]") {
    const Matrix lower{
        {2.0, 0.0, 0.0},
        {-1.0, 3.0, 0.0},
        {4.0, 2.0, 1.0}
    };
    const Vector expected{1.0, 2.0, -1.0};
    const Vector rhs = lower * expected;

    const Vector x = linalg::forward_substitution(lower, rhs);
    REQUIRE(x.size() == expected.size());
    CHECK(x[0] == Catch::Approx(expected[0]));
    CHECK(x[1] == Catch::Approx(expected[1]));
    CHECK(x[2] == Catch::Approx(expected[2]));
    CHECK(residual_norm(lower, x, rhs) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("Backward substitution solves upper-triangular systems", "[triangular]") {
    const Matrix upper{
        {4.0, -2.0, 1.0},
        {0.0, 3.0, 5.0},
        {0.0, 0.0, -2.0}
    };
    const Vector expected{2.0, -1.0, 3.0};
    const Vector rhs = upper * expected;

    const Vector x = linalg::backward_substitution(upper, rhs);
    REQUIRE(x.size() == expected.size());
    CHECK(x[0] == Catch::Approx(expected[0]));
    CHECK(x[1] == Catch::Approx(expected[1]));
    CHECK(x[2] == Catch::Approx(expected[2]));
    CHECK(residual_norm(upper, x, rhs) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("Triangular solves support unit-diagonal systems", "[triangular]") {
    const Matrix lower{
        {1.0, 0.0, 0.0},
        {-2.0, 1.0, 0.0},
        {3.0, -1.0, 1.0}
    };
    const Vector rhs{1.0, 0.0, 4.0};

    const Vector x = linalg::forward_substitution(lower, rhs, 1e-12, true);
    CHECK(x[0] == Catch::Approx(1.0));
    CHECK(x[1] == Catch::Approx(2.0));
    CHECK(x[2] == Catch::Approx(3.0));
}

TEST_CASE("Triangular solves reject shape and structure mismatches", "[triangular]") {
    const Matrix nonsquare(2, 3);
    const Vector rhs2{1.0, 2.0};
    CHECK_THROWS_AS(linalg::forward_substitution(nonsquare, rhs2), DimensionMismatchError);

    const Matrix lower{
        {1.0, 1.0},
        {2.0, 3.0}
    };
    CHECK_THROWS_AS(linalg::forward_substitution(lower, rhs2), std::invalid_argument);

    const Matrix upper{
        {1.0, 2.0},
        {1.0, 3.0}
    };
    CHECK_THROWS_AS(linalg::backward_substitution(upper, rhs2), std::invalid_argument);
}

TEST_CASE("Triangular solves detect negligible diagonal entries", "[triangular]") {
    const Matrix lower{
        {1e-14, 0.0},
        {2.0, 1.0}
    };
    const Vector rhs{1.0, 2.0};
    CHECK_THROWS_AS(linalg::forward_substitution(lower, rhs), SingularMatrixError);

    const Matrix upper{
        {1.0, 2.0},
        {0.0, 1e-14}
    };
    CHECK_THROWS_AS(linalg::backward_substitution(upper, rhs), SingularMatrixError);
}
