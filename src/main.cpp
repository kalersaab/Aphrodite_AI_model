#include "kaler/math/Matrix.hpp"
#include "kaler/nn/Activations.hpp"

#include <iomanip>
#include <iostream>

int main() {
    using kaler::math::Matrix;
    std::cout << "APHRODITE — A GPT Language Model Implementation\n";
    std::cout << "Phase 1: Neural network foundation\n\n";

    const Matrix a(2, 3, {1, 2, 3, 4, 5, 6});
    const Matrix b(3, 2, {7, 8, 9, 10, 11, 12});
    const Matrix c = kaler::math::matmul(a, b);

    std::cout << "A x B:\n" << std::fixed << std::setprecision(2);
    for (std::size_t r = 0; r < c.rows(); ++r) {
        for (std::size_t col = 0; col < c.cols(); ++col)
            std::cout << std::setw(8) << c(r, col);
        std::cout << '\n';
    }

    const Matrix logits(2, 3, {1000.0, 1001.0, 1002.0, 1.0, 2.0, 3.0});
    const Matrix probabilities = logits.softmaxRows();
    std::cout << "\nSoftmax row sums (each should be 1): ";
    for (std::size_t r = 0; r < probabilities.rows(); ++r) {
        double sum = 0.0;
        for (std::size_t col = 0; col < probabilities.cols(); ++col)
            sum += probabilities(r, col);
        std::cout << sum << (r + 1 == probabilities.rows() ? '\n' : ' ');
    }

    std::cout << "\nPhase 1 complete. This is math infrastructure, not a trained model yet.\n";
    return 0;
}
