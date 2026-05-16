export module linalgebra:expm;
import std;
import :error;
import :vector;
import :matrix;
import :lu;

// References used throughout this file:
//   Higham — "Functions of Matrices: Theory and Computation" (SIAM 2008)
//   Al-Mohy & Higham (2009), SIAM J. Matrix Anal. Appl. 31(3)

export namespace linalgebra {

// Matrix exponential via Padé [13/13] approximant + scaling and squaring.
[[nodiscard]] Matrix expm(const Matrix& A);

}  // namespace linalgebra

namespace {

// Padé [13/13] coefficients, Higham (2008) Table 10.4.
constexpr double pade_b[14] = {
    64764752532480000.0,
    32382376266240000.0,
     7771770303897600.0,
     1187353796428800.0,
      129060195264000.0,
       10559470521600.0,
         670442572800.0,
          33522128640.0,
           1323241920.0,
             40840800.0,
               960960.0,
                16380.0,
                  182.0,
                    1.0,
};

// theta_13: ||A||_1 threshold below which no scaling is needed.
constexpr double theta_13 = 5.371920351148152;

// 1-norm of a matrix.
double one_norm(const linalgebra::Matrix& M) {
    const std::size_t n = M.cols();
    const std::size_t m = M.rows();
    double result = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        double col_sum = 0.0;
        for (std::size_t i = 0; i < m; ++i) col_sum += std::abs(M(i, j));
        result = std::max(result, col_sum);
    }
    return result;
}

// Evaluate the Padé [13/13] numerator U and denominator V for matrix B.
// Uses the factored evaluation from Higham Algorithm 10.20.
//
//   W1 = b[13]*A6 + b[11]*A4 + b[9]*A2
//   W2 = b[7]*A6  + b[5]*A4  + b[3]*A2 + b[1]*I
//   Z1 = b[12]*A6 + b[10]*A4 + b[8]*A2
//   Z2 = b[6]*A6  + b[4]*A4  + b[2]*A2 + b[0]*I
//   W  = A6*W1 + W2
//   U  = B * W
//   V  = A6*Z1 + Z2
//
// expm(B) ≈ (V - U)^{-1} * (V + U)
std::pair<linalgebra::Matrix, linalgebra::Matrix>
pade13(const linalgebra::Matrix& B) {
    const std::size_t n = B.rows();
    const linalgebra::Matrix I = linalgebra::Matrix::identity(n);

    const linalgebra::Matrix A2 = B * B;
    const linalgebra::Matrix A4 = A2 * A2;
    const linalgebra::Matrix A6 = A2 * A4;

    // W1 = b[13]*A6 + b[11]*A4 + b[9]*A2
    linalgebra::Matrix W1(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            W1(i, j) = pade_b[13] * A6(i, j) + pade_b[11] * A4(i, j) + pade_b[9] * A2(i, j);

    // W2 = b[7]*A6 + b[5]*A4 + b[3]*A2 + b[1]*I
    linalgebra::Matrix W2(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            W2(i, j) = pade_b[7] * A6(i, j) + pade_b[5] * A4(i, j)
                     + pade_b[3] * A2(i, j) + pade_b[1] * I(i, j);

    // Z1 = b[12]*A6 + b[10]*A4 + b[8]*A2
    linalgebra::Matrix Z1(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Z1(i, j) = pade_b[12] * A6(i, j) + pade_b[10] * A4(i, j) + pade_b[8] * A2(i, j);

    // Z2 = b[6]*A6 + b[4]*A4 + b[2]*A2 + b[0]*I
    linalgebra::Matrix Z2(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Z2(i, j) = pade_b[6] * A6(i, j) + pade_b[4] * A4(i, j)
                     + pade_b[2] * A2(i, j) + pade_b[0] * I(i, j);

    // W = A6*W1 + W2
    const linalgebra::Matrix W = A6 * W1 + W2;

    // U = B * W  (numerator)
    const linalgebra::Matrix U = B * W;

    // V = A6*Z1 + Z2  (denominator)
    const linalgebra::Matrix V = A6 * Z1 + Z2;

    return {U, V};
}

}  // namespace

namespace linalgebra {

Matrix expm(const Matrix& A) {
    if (A.rows() != A.cols()) {
        std::ostringstream oss;
        oss << "expm requires a square matrix, got " << A.rows() << "x" << A.cols();
        throw DimensionMismatchError(oss.str());
    }

    const std::size_t n = A.rows();
    if (n == 0) return Matrix::identity(0);

    // Determine scaling factor s such that ||A / 2^s||_1 <= theta_13.
    const double norm_A = one_norm(A);
    int s = 0;
    if (norm_A > theta_13) {
        s = static_cast<int>(std::ceil(std::log2(norm_A / theta_13)));
        if (s < 0) s = 0;
    }

    // Scale B = A / 2^s.
    const double scale = 1.0 / std::ldexp(1.0, s);
    Matrix B(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            B(i, j) = A(i, j) * scale;

    // Compute Padé [13/13] approximant: R = (V - U)^{-1} * (V + U).
    auto [U, V] = pade13(B);

    // Numerator = V + U,  Denominator = V - U.
    Matrix Numerator(n, n, 0.0);
    Matrix Denominator(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Numerator(i, j)   = V(i, j) + U(i, j);
            Denominator(i, j) = V(i, j) - U(i, j);
        }
    }

    const LUResult lu_denom = lu_factor(Denominator);

    Matrix R(n, n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        Vector col(n);
        for (std::size_t i = 0; i < n; ++i) col[i] = Numerator(i, j);
        const Vector sol = lu_solve(lu_denom, col);
        for (std::size_t i = 0; i < n; ++i) R(i, j) = sol[i];
    }

    for (int i = 0; i < s; ++i) R = R * R;

    return R;
}

}  // namespace linalgebra
