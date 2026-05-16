import linalgebra;

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <random>

using linalgebra::DimensionMismatchError;
using linalgebra::Matrix;
using linalgebra::SVDResult;

namespace {

double frobenius_diff(const Matrix& A, const Matrix& B) {
    double s = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) {
            const double d = A(i, j) - B(i, j);
            s += d * d;
        }
    return std::sqrt(s);
}

// Reconstruct A = U * diag(sigma) * Vt and measure error
double reconstruction_error(const Matrix& A, const SVDResult& svd) {
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    const std::size_t p = svd.sigma.size();

    // Build U_thin (m x p) — first p columns of U.
    Matrix U_thin(m, p);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < p; ++j)
            U_thin(i, j) = svd.U(i, j);

    // Build Sigma_diag (p x p).
    Matrix S(p, p, 0.0);
    for (std::size_t i = 0; i < p; ++i) S(i, i) = svd.sigma[i];

    // Build Vt_thin (p x n) — first p rows of Vt.
    Matrix Vt_thin(p, n);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Vt_thin(i, j) = svd.Vt(i, j);

    const Matrix recon = U_thin * S * Vt_thin;
    return frobenius_diff(A, recon);
}

// Check orthogonality of the first k columns of M
double col_ortho_error(const Matrix& M, std::size_t k) {
    const std::size_t m = M.rows();
    double s = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            double dot = 0.0;
            for (std::size_t r = 0; r < m; ++r) dot += M(r, i) * M(r, j);
            const double expected = (i == j) ? 1.0 : 0.0;
            const double d = dot - expected;
            s += d * d;
        }
    }
    return std::sqrt(s);
}

}  // namespace

TEST_CASE("svd: 2x2 diagonal", "[svd]") {
    Matrix A{{3.0, 0.0}, {0.0, -2.0}};
    auto res = linalgebra::svd(A);

    REQUIRE(res.sigma[0] == Catch::Approx(3.0).epsilon(1e-10));
    REQUIRE(res.sigma[1] == Catch::Approx(2.0).epsilon(1e-10));
    REQUIRE(reconstruction_error(A, res) < 1e-10);
}

TEST_CASE("svd: 3x2 tall matrix", "[svd]") {
    Matrix A{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    auto res = linalgebra::svd(A);

    REQUIRE(reconstruction_error(A, res) < 1e-10);

    REQUIRE(res.sigma[0] >= res.sigma[1]);
    REQUIRE(res.sigma[1] >= 0.0);

    REQUIRE(col_ortho_error(res.U, 2) < 1e-10);

    REQUIRE(col_ortho_error(linalgebra::transpose(res.Vt), 2) < 1e-10);
}

TEST_CASE("svd: rank-1 matrix", "[svd]") {
    // A = u * v^T for u = [1,2,3], v = [2,1].
    Matrix A{{2.0, 1.0}, {4.0, 2.0}, {6.0, 3.0}};
    auto res = linalgebra::svd(A);

    REQUIRE(reconstruction_error(A, res) < 1e-10);
    // Second singular value should be near zero.
    REQUIRE(res.sigma[0] > 1e-10);
    REQUIRE(res.sigma[1] < 1e-8);
}

TEST_CASE("svd: identity 4x4", "[svd]") {
    auto I = Matrix::identity(4);
    auto res = linalgebra::svd(I);

    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(res.sigma[i] == Catch::Approx(1.0).epsilon(1e-10));
    REQUIRE(reconstruction_error(I, res) < 1e-10);
}

TEST_CASE("svd: 4x4 random", "[svd]") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    Matrix A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = dist(rng);

    auto res = linalgebra::svd(A);
    REQUIRE(reconstruction_error(A, res) < 1e-9);
    REQUIRE(col_ortho_error(res.U, 4) < 1e-10);
    REQUIRE(col_ortho_error(linalgebra::transpose(res.Vt), 4) < 1e-10);
}

TEST_CASE("svd: Hilbert 4x4", "[svd]") {
    Matrix H(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            H(i, j) = 1.0 / static_cast<double>(i + j + 1);
    auto res = linalgebra::svd(H);
    REQUIRE(reconstruction_error(H, res) < 1e-10);
    for (std::size_t i = 0; i < 4; ++i) REQUIRE(res.sigma[i] >= 0.0);
}

TEST_CASE("svd: wide matrix throws", "[svd]") {
    Matrix A(2, 4);
    REQUIRE_THROWS_AS(linalgebra::svd(A), DimensionMismatchError);
}
