import linalgebra;

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <random>

using linalgebra::DimensionMismatchError;
using linalgebra::LinAlgError;
using linalgebra::Matrix;
using linalgebra::Vector;

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

// Check orthogonality ||Q^T Q - I||_F < tol.
double ortho_error(const Matrix& Q) {
    const std::size_t n = Q.rows();
    const std::size_t m = Q.cols();
    const Matrix Qt = linalgebra::transpose(Q);
    const Matrix QtQ = Qt * Q;
    const Matrix I = Matrix::identity(m);
    return frobenius_diff(QtQ, I);
}

// Build a symmetric matrix from B^T B + shift * I.
Matrix make_sym(std::size_t n, double shift, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix B(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            B(i, j) = dist(rng);
    Matrix A = linalgebra::transpose(B) * B;
    for (std::size_t i = 0; i < n; ++i) A(i, i) += shift;
    return A;
}

}  // namespace

// tridiagonalize

TEST_CASE("tridiagonalize: 2x2 symmetric", "[sym_eigen][tridiagonalize]") {
    Matrix A{{4.0, 2.0}, {2.0, 3.0}};
    auto res = linalgebra::tridiagonalize(A);

    // Reconstruction: Q T Q^T == A.
    const Matrix QTQT = res.Q * res.T * linalgebra::transpose(res.Q);
    REQUIRE(frobenius_diff(A, QTQT) < 1e-12);

    // Q must be orthogonal.
    REQUIRE(ortho_error(res.Q) < 1e-12);

    // T must be tridiagonal: T(i,j) == 0 for |i-j| > 1.
    const std::size_t n = res.T.rows();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (i > j + 1 || j > i + 1)
                REQUIRE(std::abs(res.T(i, j)) < 1e-12);
}

TEST_CASE("tridiagonalize: 4x4 symmetric", "[sym_eigen][tridiagonalize]") {
    Matrix A{{6.0, 2.0, 1.0, 0.0},
             {2.0, 5.0, 3.0, 1.0},
             {1.0, 3.0, 4.0, 2.0},
             {0.0, 1.0, 2.0, 3.0}};
    auto res = linalgebra::tridiagonalize(A);

    const Matrix QTQT = res.Q * res.T * linalgebra::transpose(res.Q);
    REQUIRE(frobenius_diff(A, QTQT) < 1e-10);
    REQUIRE(ortho_error(res.Q) < 1e-12);

    const std::size_t n = res.T.rows();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (i > j + 1 || j > i + 1)
                REQUIRE(std::abs(res.T(i, j)) < 1e-10);
}

TEST_CASE("tridiagonalize: identity", "[sym_eigen][tridiagonalize]") {
    auto I = Matrix::identity(5);
    auto res = linalgebra::tridiagonalize(I);
    REQUIRE(frobenius_diff(I, res.Q * res.T * linalgebra::transpose(res.Q)) < 1e-12);
}

TEST_CASE("tridiagonalize: random symmetric", "[sym_eigen][tridiagonalize]") {
    std::mt19937 rng(77);
    Matrix A = make_sym(8, 5.0, rng);
    auto res = linalgebra::tridiagonalize(A);
    REQUIRE(frobenius_diff(A, res.Q * res.T * linalgebra::transpose(res.Q)) < 1e-9);
    REQUIRE(ortho_error(res.Q) < 1e-11);
}

TEST_CASE("tridiagonalize: non-symmetric throws", "[sym_eigen][tridiagonalize]") {
    Matrix A{{1.0, 2.0}, {3.0, 4.0}};
    REQUIRE_THROWS_AS(linalgebra::tridiagonalize(A), LinAlgError);
}

TEST_CASE("tridiagonalize: non-square throws", "[sym_eigen][tridiagonalize]") {
    Matrix A(2, 3);
    REQUIRE_THROWS_AS(linalgebra::tridiagonalize(A), DimensionMismatchError);
}

// eigenvectors_inverse_iteration

TEST_CASE("eigenvectors_inverse_iteration: 2x2 diagonal", "[sym_eigen][inverse_iter]") {
    Matrix A(2, 2, 0.0);
    A(0, 0) = 3.0; A(1, 1) = 7.0;
    Vector lambdas{3.0, 7.0};

    auto res = linalgebra::eigenvectors_inverse_iteration(A, lambdas);

    // Each column should satisfy A*v ≈ lambda*v.
    for (std::size_t col = 0; col < 2; ++col) {
        Vector v(2);
        v[0] = res.eigenvectors(0, col);
        v[1] = res.eigenvectors(1, col);
        const Vector Av = A * v;
        const double lam = lambdas[col];
        double resid = 0.0;
        for (std::size_t i = 0; i < 2; ++i) {
            const double d = Av[i] - lam * v[i];
            resid += d * d;
        }
        REQUIRE(std::sqrt(resid) < 1e-8);
    }
}

TEST_CASE("eigenvectors_inverse_iteration: 3x3 symmetric", "[sym_eigen][inverse_iter]") {
    // Known symmetric matrix: compute eigenvalues with francis, then eigenvectors.
    Matrix A{{6.0, 2.0, 1.0},
             {2.0, 3.0, 1.0},
             {1.0, 1.0, 1.0}};

    auto eig = linalgebra::eigenvalues_francis(A);
    // Use real eigenvalues only.
    auto res = linalgebra::eigenvectors_inverse_iteration(A, eig.eigenvalues_real);

    for (std::size_t col = 0; col < 3; ++col) {
        Vector v(3);
        for (std::size_t i = 0; i < 3; ++i) v[i] = res.eigenvectors(i, col);
        const Vector Av = A * v;
        const double lam = eig.eigenvalues_real[col];
        double resid = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            const double d = Av[i] - lam * v[i];
            resid += d * d;
        }
        REQUIRE(std::sqrt(resid) < 1e-6);
    }
}

TEST_CASE("eigenvectors_inverse_iteration: non-square throws", "[sym_eigen][inverse_iter]") {
    Matrix A(2, 3);
    Vector lambdas{1.0};
    REQUIRE_THROWS_AS(linalgebra::eigenvectors_inverse_iteration(A, lambdas),
                      DimensionMismatchError);
}
