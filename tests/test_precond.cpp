import linalgebra;

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>

using linalgebra::DimensionMismatchError;
using linalgebra::Matrix;
using linalgebra::SingularMatrixError;
using linalgebra::Vector;

// condition_number_1norm

TEST_CASE("condition_number_1norm: identity", "[precond][condition]") {
    for (std::size_t n : {1, 2, 5}) {
        auto I = Matrix::identity(n);
        REQUIRE(linalgebra::condition_number_1norm(I) == Catch::Approx(1.0).epsilon(1e-10));
    }
}

TEST_CASE("condition_number_1norm: diagonal matrix", "[precond][condition]") {
    // diag(1, 10, 100): ||A||_1 = 100, ||A^{-1}||_1 = 1, so cond = 100.
    Matrix A(3, 3, 0.0);
    A(0, 0) = 1.0; A(1, 1) = 10.0; A(2, 2) = 100.0;
    const double c = linalgebra::condition_number_1norm(A);
    // Power-iteration estimator gives a lower bound; for simple diagonal it should be exact.
    REQUIRE(c == Catch::Approx(100.0).epsilon(1e-8));
}

TEST_CASE("condition_number_1norm: ill-conditioned Hilbert 4x4", "[precond][condition]") {
    // Hilbert matrix H(i,j) = 1/(i+j+1).
    Matrix H(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            H(i, j) = 1.0 / static_cast<double>(i + j + 1);
    const double c = linalgebra::condition_number_1norm(H);
    REQUIRE(c > 1e3);  // Hilbert matrices are notoriously ill-conditioned
}

TEST_CASE("condition_number_1norm: non-square throws", "[precond][condition]") {
    Matrix A(2, 3);
    REQUIRE_THROWS_AS(linalgebra::condition_number_1norm(A), DimensionMismatchError);
}

// precond_jacobi

TEST_CASE("precond_jacobi: diagonal matrix", "[precond][jacobi]") {
    Matrix A(3, 3, 0.0);
    A(0, 0) = 2.0; A(1, 1) = 4.0; A(2, 2) = 8.0;
    auto P = linalgebra::precond_jacobi(A);

    REQUIRE(P.inv_diag[0] == Catch::Approx(0.5));
    REQUIRE(P.inv_diag[1] == Catch::Approx(0.25));
    REQUIRE(P.inv_diag[2] == Catch::Approx(0.125));

    Vector x{1.0, 1.0, 1.0};
    auto y = linalgebra::apply(P, x);
    REQUIRE(y[0] == Catch::Approx(0.5));
    REQUIRE(y[1] == Catch::Approx(0.25));
    REQUIRE(y[2] == Catch::Approx(0.125));
}

TEST_CASE("precond_jacobi: zero diagonal throws", "[precond][jacobi]") {
    Matrix A{{1.0, 0.0}, {0.0, 0.0}};
    REQUIRE_THROWS_AS(linalgebra::precond_jacobi(A), SingularMatrixError);
}

TEST_CASE("precond_jacobi: non-square throws", "[precond][jacobi]") {
    Matrix A(2, 3);
    REQUIRE_THROWS_AS(linalgebra::precond_jacobi(A), DimensionMismatchError);
}

// precond_ilu0

TEST_CASE("precond_ilu0: identity", "[precond][ilu0]") {
    auto I = Matrix::identity(3);
    auto P = linalgebra::precond_ilu0(I);
    Vector b{3.0, 1.0, 4.0};
    auto y = linalgebra::apply(P, b);
    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE(y[i] == Catch::Approx(b[i]).margin(1e-12));
}

TEST_CASE("precond_ilu0: 3x3 SPD", "[precond][ilu0]") {
    // Dense ILU0 = exact LU without pivoting, so for any nonsingular A,
    // apply(P, A*x) should recover x.
    Matrix A{{4.0, 2.0, 0.0},
             {2.0, 3.0, 1.0},
             {0.0, 1.0, 2.0}};
    auto P = linalgebra::precond_ilu0(A);

    Vector x_true{1.0, 2.0, 3.0};
    Vector rhs = A * x_true;
    auto x_rec = linalgebra::apply(P, rhs);

    for (std::size_t i = 0; i < 3; ++i)
        REQUIRE(x_rec[i] == Catch::Approx(x_true[i]).margin(1e-10));
}

TEST_CASE("precond_ilu0: near-singular pivot throws", "[precond][ilu0]") {
    Matrix A{{0.0, 1.0}, {1.0, 1.0}};
    REQUIRE_THROWS_AS(linalgebra::precond_ilu0(A), SingularMatrixError);
}

// lstsq

TEST_CASE("lstsq: square full-rank system", "[precond][lstsq]") {
    Matrix A{{2.0, 1.0}, {1.0, 3.0}};
    Vector b{5.0, 7.0};
    auto res = linalgebra::lstsq(A, b);
    REQUIRE(res.rank == 2);
    REQUIRE(linalgebra::norm2(A * res.x - b) < 1e-10);
}

TEST_CASE("lstsq: overdetermined full-rank", "[precond][lstsq]") {
    // A is 3x2, consistent overdetermined system.
    Matrix A{{1.0, 1.0}, {2.0, 1.0}, {3.0, 1.0}};
    Vector b{6.0, 5.0, 7.0};
    auto res = linalgebra::lstsq(A, b);
    REQUIRE(res.rank == 2);
    // Residual should be the minimum achievable (verify normal equations: A^T A x = A^T b).
    Vector AtAx = linalgebra::transpose(A) * (A * res.x);
    Vector Atb  = linalgebra::transpose(A) * b;
    for (std::size_t i = 0; i < 2; ++i)
        REQUIRE(AtAx[i] == Catch::Approx(Atb[i]).margin(1e-8));
}

TEST_CASE("lstsq: rank-deficient", "[precond][lstsq]") {
    // col2 = 2*col1 → rank 1.
    Matrix A{{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}};
    Vector b{1.0, 2.0, 3.0};
    auto res = linalgebra::lstsq(A, b);
    REQUIRE(res.rank < 2);
    // The residual should be the minimum achievable.
    REQUIRE(res.residual_norm < 1e-8);
}

TEST_CASE("lstsq: non-tall matrix throws", "[precond][lstsq]") {
    Matrix A(2, 3);
    Vector b(2, 0.0);
    REQUIRE_THROWS_AS(linalgebra::lstsq(A, b), DimensionMismatchError);
}
