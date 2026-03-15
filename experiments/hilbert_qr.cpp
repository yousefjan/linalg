// Experiment: QR methods on Hilbert matrices
//
// The Hilbert matrix H[i][j] = 1/(i+j+1) is the canonical ill-conditioned
// dense matrix.  Its condition number grows roughly as (3.5 * e)^n / sqrt(n),
// reaching ~10^13 at n=10 and ~10^18 at n=14.
//
// We compare classical GS, modified GS, and Householder QR on:
//   - reconstruction error   ||A - QR||_F
//   - orthogonality error    ||Q^T Q - I||_F
//   - wall-clock time (minimum over several trials)

#include "matrix.hpp"
#include "qr.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

using linalg::Matrix;
using linalg::QRResult;

// ---------------------------------------------------------------------------
// Matrix construction
// ---------------------------------------------------------------------------

Matrix hilbert(std::size_t n) {
    Matrix H(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            H(i, j) = 1.0 / static_cast<double>(i + j + 1);
    return H;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

double reconstruction_error(const Matrix& A, const QRResult& qr) {
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    const Matrix QR = qr.Q * qr.R;
    double err = 0.0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            const double d = A(i, j) - QR(i, j);
            err += d * d;
        }
    return std::sqrt(err);
}

double orthogonality_error(const QRResult& qr) {
    const Matrix& Q = qr.Q;
    const std::size_t n = Q.cols();
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < Q.rows(); ++k) s += Q(k, i) * Q(k, j);
            const double d = s - (i == j ? 1.0 : 0.0);
            err += d * d;
        }
    return std::sqrt(err);
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

using Clock = std::chrono::high_resolution_clock;
using Seconds = std::chrono::duration<double>;

// Run fn() `trials` times, return minimum elapsed seconds.
template<typename Fn>
double min_time(Fn fn, int trials = 5) {
    double best = 1e18;
    for (int t = 0; t < trials; ++t) {
        const auto t0 = Clock::now();
        fn();
        const auto t1 = Clock::now();
        best = std::min(best, Seconds(t1 - t0).count());
    }
    return best;
}

// ---------------------------------------------------------------------------
// Run one method on one size, return {recon, ortho, time} or nullopt on failure
// ---------------------------------------------------------------------------

using QRFn = std::function<QRResult(const Matrix&)>;

struct Result { double recon, ortho, time_s; };

std::optional<Result> measure(const Matrix& A, QRFn fn) {
    try {
        // Run once to get metrics.
        const QRResult qr = fn(A);
        const double re = reconstruction_error(A, qr);
        const double oe = orthogonality_error(qr);
        // Time over multiple trials.
        const double t = min_time([&] { fn(A); });
        return Result{re, oe, t};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// ---------------------------------------------------------------------------
// Pretty printing
// ---------------------------------------------------------------------------

void print_row(const std::string& method, std::optional<Result> r) {
    std::cout << std::left << std::setw(16) << method;
    if (!r) {
        std::cout << "  FAILED (linearly dependent columns)\n";
        return;
    }
    std::cout << std::scientific << std::setprecision(2)
              << std::setw(14) << r->recon
              << std::setw(14) << r->ortho
              << std::fixed << std::setprecision(3)
              << std::setw(10) << r->time_s * 1e6 << " µs\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    std::cout << std::string(70, '*') << "\n";
    std::cout << "  Hilbert QR Experiment — comparing GS variants and Householder\n";
    std::cout << std::string(70, '*') << "\n\n";
    std::cout <<
        "H[i][j] = 1/(i+j+1).  Condition number grows ~exponentially with n.\n"
        "Orthogonality loss in classical GS tracks condition number directly.\n"
        "Modified GS recovers ~half the lost digits.  Householder is unaffected.\n\n";

    const std::size_t sizes[] = {2, 3, 4, 5, 6, 7, 8, 10, 12};

    for (std::size_t n : sizes) {
        const Matrix H = hilbert(n);

        std::cout << std::string(70, '-') << "\n";
        std::cout << "  n = " << n << "\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << std::left
                  << std::setw(16) << "Method"
                  << std::setw(14) << "||A-QR||_F"
                  << std::setw(14) << "||QtQ-I||_F"
                  << std::setw(10) << "Time\n";
        std::cout << std::string(70, ' ') << "\n";

        print_row("classical_gs",
            measure(H, [](const Matrix& A) { return linalg::qr_classical_gs(A); }));
        print_row("modified_gs",
            measure(H, [](const Matrix& A) { return linalg::qr_modified_gs(A); }));
        print_row("householder",
            measure(H, [](const Matrix& A) { return linalg::qr_householder(A); }));
    }

    return 0;
}
