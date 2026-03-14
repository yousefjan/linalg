#include "matrix.hpp"
#include "norms.hpp"
#include "triangular_solve.hpp"
#include "vector.hpp"

#include <iomanip>
#include <iostream>

int main() {
    const linalg::Matrix basis = linalg::Matrix::identity(3);
    const linalg::Matrix upper{
        {4.0, -2.0, 1.0},
        {0.0, 3.0, 5.0},
        {0.0, 0.0, -2.0}
    };
    const linalg::Vector expected{2.0, -1.0, 3.0};
    const linalg::Vector rhs = upper * expected;
    const linalg::Vector x = linalg::backward_substitution(upper, rhs);

    std::cout << "Week 3 triangular solve demo\n";
    std::cout << "Identity matrix diagonal: ";
    for (std::size_t i = 0; i < basis.rows(); ++i) {
        std::cout << basis(i, i) << (i + 1 == basis.rows() ? '\n' : ' ');
    }

    std::cout << "Recovered solution x: ";
    for (std::size_t i = 0; i < x.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << x[i]
                  << (i + 1 == x.size() ? '\n' : ' ');
    }

    const linalg::Vector residual = (upper * x) - rhs;
    std::cout << "Residual 2-norm = " << linalg::norm2(residual) << '\n';

    return 0;
}
