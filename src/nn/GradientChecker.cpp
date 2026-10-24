#include "GradientChecker.h"

namespace kaler {
namespace nn {

bool GradientChecker::check_gradients(Parameter& param,
                                     LossFunction loss_fn,
                                     double epsilon,
                                     double tolerance) {
    auto& data = param.data();
    const auto& analytical_grad = param.grad();
    
    math::Matrix numerical_grad(data.rows(), data.cols(), 0.0);
    
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            double original = data(i, j);
            
            const_cast<double&>(data(i, j)) = original + epsilon;
            double loss_plus = loss_fn();
            
            const_cast<double&>(data(i, j)) = original - epsilon;
            double loss_minus = loss_fn();
            
            const_cast<double&>(numerical_grad(i, j)) = (loss_plus - loss_minus) / (2.0 * epsilon);
            
            const_cast<double&>(data(i, j)) = original;
        }
    }
    
    double max_diff = 0.0;
    double max_rel_diff = 0.0;
    
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            double diff = std::abs(analytical_grad(i, j) - numerical_grad(i, j));
            max_diff = std::max(max_diff, diff);
            
            double denom = std::max(std::abs(analytical_grad(i, j)) + std::abs(numerical_grad(i, j)), 1e-8);
            double rel_diff = diff / denom;
            max_rel_diff = std::max(max_rel_diff, rel_diff);
        }
    }
    
    std::cout << "Gradient check for " << param.name() << ":\n";
    std::cout << "  Max absolute difference: " << max_diff << "\n";
    std::cout << "  Max relative difference: " << max_rel_diff << "\n";
    
    bool passed = max_rel_diff < tolerance;
    std::cout << "  Status: " << (passed ? "PASSED ✓" : "FAILED ✗") << "\n";
    
    return passed;
}

void GradientChecker::print_gradient_comparison(const math::Matrix& analytical,
                                               const math::Matrix& numerical) {
    std::cout << "\nGradient Comparison (first 5x5):\n";
    std::cout << "Analytical | Numerical | Difference\n";
    std::cout << "----------------------------------------\n";
    
    size_t max_rows = std::min(size_t(5), analytical.rows());
    size_t max_cols = std::min(size_t(5), analytical.cols());
    
    for (size_t i = 0; i < max_rows; ++i) {
        for (size_t j = 0; j < max_cols; ++j) {
            double diff = std::abs(analytical(i, j) - numerical(i, j));
            printf("%9.6f | %9.6f | %9.6f\n",
                   analytical(i, j), numerical(i, j), diff);
        }
    }
}

} // namespace nn
} // namespace kaler
