#include "matrix.hpp"
#include "vector.hpp"

#include <iomanip>
#include <iostream>

int main() {
    const linalg::Matrix basis = linalg::Matrix::identity(3);
    const linalg::Vector x{1.0, -2.0, 0.5};

    std::cout << "Identity matrix diagonal: ";
    for (std::size_t i = 0; i < basis.rows(); ++i) {
        std::cout << basis(i, i) << (i + 1 == basis.rows() ? '\n' : ' ');
    }

    std::cout << "Vector x: ";
    for (std::size_t i = 0; i < x.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << x[i]
                  << (i + 1 == x.size() ? '\n' : ' ');
    }

    std::cout << "x dot x = " << linalg::dot(x, x) << '\n';

    return 0;
}
