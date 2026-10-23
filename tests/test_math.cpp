#include "kaler/math/Matrix.hpp"
#include "kaler/nn/Activations.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
int failures = 0;

void check(bool condition, const std::string& name) {
    if (condition) std::cout << "[PASS] " << name << '\n';
    else { std::cerr << "[FAIL] " << name << '\n'; ++failures; }
}

bool near(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}
}

int main() {
    using kaler::math::Matrix;

    const Matrix a(2, 3, {1, 2, 3, 4, 5, 6});
    check(a.rows() == 2 && a.cols() == 3, "matrix shape");
    check(near(a(1, 2), 6.0), "row-major indexing");

    const Matrix t = a.transpose();
    check(t.rows() == 3 && t.cols() == 2 && near(t(2, 1), 6.0), "transpose");

    const Matrix b(3, 2, {7, 8, 9, 10, 11, 12});
    const Matrix product = kaler::math::matmul(a, b);
    check(product.rows() == 2 && product.cols() == 2, "matmul output shape");
    check(near(product(0, 0), 58.0) && near(product(0, 1), 64.0)
          && near(product(1, 0), 139.0) && near(product(1, 1), 154.0),
          "matmul known answer");

    const Matrix sum = a + a;
    check(near(sum(1, 2), 12.0), "matrix addition");
    const Matrix diff = sum - a;
    check(near(diff(0, 1), 2.0), "matrix subtraction");
    check(near((a * 2.0)(1, 1), 10.0), "scalar multiplication");

    const Matrix identity = Matrix::identity(3);
    const Matrix identityProduct = kaler::math::matmul(a, identity);
    check(near(identityProduct(1, 2), a(1, 2)), "identity multiplication");

    const Matrix logits(2, 3, {1000, 1001, 1002, -1000, -1001, -1002});
    const Matrix probs = logits.softmaxRows();
    bool normalized = true;
    for (std::size_t r = 0; r < probs.rows(); ++r) {
        double s = 0.0;
        for (std::size_t c = 0; c < probs.cols(); ++c) {
            normalized = normalized && std::isfinite(probs(r, c))
                         && probs(r, c) >= 0.0 && probs(r, c) <= 1.0;
            s += probs(r, c);
        }
        normalized = normalized && near(s, 1.0);
    }
    check(normalized, "stable row-wise softmax");

    const Matrix activationInput(1, 4, {-2, -0.5, 0, 2});
    const Matrix reluOut = kaler::nn::relu(activationInput);
    check(near(reluOut(0, 0), 0) && near(reluOut(0, 3), 2), "ReLU");
    const Matrix tanhOut = kaler::nn::tanhActivation(activationInput);
    check(near(tanhOut(0, 0), std::tanh(-2.0)), "tanh activation");

    bool boundsThrew = false;
    try { (void)a(99, 0); } catch (const std::out_of_range&) { boundsThrew = true; }
    check(boundsThrew, "out-of-bounds access throws");

    bool shapeThrew = false;
    try {
        const Matrix bad(1, 1, {1});
        (void)(a + bad);
    } catch (const std::invalid_argument&) { shapeThrew = true; }
    check(shapeThrew, "shape mismatch throws");

    bool matmulThrew = false;
    try {
        const Matrix bad(4, 1, 0.0);
        (void)kaler::math::matmul(a, bad);
    } catch (const std::invalid_argument&) { matmulThrew = true; }
    check(matmulThrew, "invalid matmul dimensions throw");

    std::cout << "\n";
    if (failures == 0) {
        std::cout << "All tests passed.\n";
        return 0;
    }
    std::cerr << failures << " test(s) failed.\n";
    return 1;
}
