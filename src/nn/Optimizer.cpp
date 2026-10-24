#include "Optimizer.h"
#include <cmath>

namespace kaler {
namespace nn {

void SGD::step() {
    for (auto* param : parameters_) {
        auto& data = param->data();
        const auto& grad = param->grad();
        
        for (size_t i = 0; i < data.rows(); ++i) {
            for (size_t j = 0; j < data.cols(); ++j) {
                const_cast<double&>(data(i, j)) -= learning_rate_ * grad(i, j);
            }
        }
    }
}

void SGD::zero_grad() {
    for (auto* param : parameters_) {
        param->zero_grad();
    }
}

void Adam::step() {
    if (m_.empty()) {
        for (auto* param : parameters_) {
            m_.emplace_back(param->data().rows(), param->data().cols(), 0.0);
            v_.emplace_back(param->data().rows(), param->data().cols(), 0.0);
        }
    }
    
    t_++;
    
    // Compute bias correction terms with numerical safety
    const double bias_correction1 = std::max(1.0 - std::pow(beta1_, t_), 1e-8);
    const double bias_correction2 = std::max(1.0 - std::pow(beta2_, t_), 1e-8);
    
    for (size_t idx = 0; idx < parameters_.size(); ++idx) {
        auto* param = parameters_[idx];
        auto& data = param->data();
        const auto& grad = param->grad();
        auto& m = m_[idx];
        auto& v = v_[idx];
        
        for (size_t i = 0; i < data.rows(); ++i) {
            for (size_t j = 0; j < data.cols(); ++j) {
                // Update biased first moment estimate
                double m_val = beta1_ * m(i, j) + (1.0 - beta1_) * grad(i, j);
                // Update biased second raw moment estimate
                double v_val = beta2_ * v(i, j) + (1.0 - beta2_) * grad(i, j) * grad(i, j);
                
                const_cast<double&>(m(i, j)) = m_val;
                const_cast<double&>(v(i, j)) = v_val;
                
                // Compute bias-corrected moment estimates
                double m_hat = m_val / bias_correction1;
                double v_hat = v_val / bias_correction2;
                
                // Compute update with numerical safety
                double denominator = std::sqrt(v_hat) + epsilon_;
                double update = learning_rate_ * m_hat / denominator;
                
                // Guard against non-finite updates
                if (!std::isfinite(update)) {
                    continue; // Skip this parameter update
                }
                
                const_cast<double&>(data(i, j)) -= update;
            }
        }
    }
}

void Adam::zero_grad() {
    for (auto* param : parameters_) {
        param->zero_grad();
    }
}

} // namespace nn
} // namespace kaler
