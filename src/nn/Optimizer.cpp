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
    
    for (size_t idx = 0; idx < parameters_.size(); ++idx) {
        auto* param = parameters_[idx];
        auto& data = param->data();
        const auto& grad = param->grad();
        auto& m = m_[idx];
        auto& v = v_[idx];
        
        for (size_t i = 0; i < data.rows(); ++i) {
            for (size_t j = 0; j < data.cols(); ++j) {
                double m_val = beta1_ * m(i, j) + (1.0 - beta1_) * grad(i, j);
                double v_val = beta2_ * v(i, j) + (1.0 - beta2_) * grad(i, j) * grad(i, j);
                
                const_cast<double&>(m(i, j)) = m_val;
                const_cast<double&>(v(i, j)) = v_val;
                
                double m_hat = m_val / (1.0 - std::pow(beta1_, t_));
                double v_hat = v_val / (1.0 - std::pow(beta2_, t_));
                
                const_cast<double&>(data(i, j)) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
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
