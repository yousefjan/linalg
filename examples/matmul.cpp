#include "matrix.hpp"

#include <iomanip>
#include <iostream>

int main() {
    const linalg::Matrix a{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    const linalg::Matrix b{
        {7.0, 8.0},
        {9.0, 10.0},
        {11.0, 12.0}
    };

    const linalg::Matrix c = a * b;

    std::cout << "Matrix multiplication\n";
    std::cout << "A is " << a.rows() << " x " << a.cols() << '\n';
    std::cout << "B is " << b.rows() << " x " << b.cols() << '\n';
    std::cout << "C = A * B:\n";

    for (std::size_t i = 0; i < c.rows(); ++i) {
        for (std::size_t j = 0; j < c.cols(); ++j) {
            std::cout << std::fixed << std::setprecision(2) << c(i, j)
                      << (j + 1 == c.cols() ? '\n' : ' ');
        }
    }

    return 0;
}
