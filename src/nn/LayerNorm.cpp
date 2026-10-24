#include "LayerNorm.h"
#include <cmath>

namespace kaler {
namespace nn {

LayerNorm::LayerNorm(size_t normalized_shape, double eps)
    : gamma_(normalized_shape, 1, "gamma")
    , beta_(normalized_shape, 1, "beta")
    , eps_(eps)
    , cached_input_(1, 1, 0.0)
    , cached_mean_(1, 1, 0.0)
    , cached_std_(1, 1, 0.0)
    , cached_normalized_(1, 1, 0.0) {
    
    for (size_t i = 0; i < normalized_shape; ++i) {
        const_cast<double&>(gamma_.data()(i, 0)) = 1.0;
        const_cast<double&>(beta_.data()(i, 0)) = 0.0;
    }
}

math::Matrix LayerNorm::forward(const math::Matrix& input) {
    cached_input_ = input;
    
    size_t d_model = input.rows();
    size_t seq_len = input.cols();
    
    cached_mean_ = math::Matrix(1, seq_len, 0.0);
    cached_std_ = math::Matrix(1, seq_len, 0.0);
    cached_normalized_ = math::Matrix(d_model, seq_len, 0.0);
    
    for (size_t i = 0; i < seq_len; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < d_model; ++j) {
            sum += input(j, i);
        }
        double mean = sum / d_model;
        const_cast<double&>(cached_mean_(0, i)) = mean;
        
        double var = 0.0;
        for (size_t j = 0; j < d_model; ++j) {
            double diff = input(j, i) - mean;
            var += diff * diff;
        }
        var /= d_model;
        double std = std::sqrt(var + eps_);
        const_cast<double&>(cached_std_(0, i)) = std;
        
        for (size_t j = 0; j < d_model; ++j) {
            double normalized = (input(j, i) - mean) / std;
            const_cast<double&>(cached_normalized_(j, i)) = normalized;
        }
    }
    
    math::Matrix output(d_model, seq_len, 0.0);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            const_cast<double&>(output(j, i)) = 
                gamma_.data()(j, 0) * cached_normalized_(j, i) + beta_.data()(j, 0);
        }
    }
    
    return output;
}

math::Matrix LayerNorm::backward(const math::Matrix& grad_output) {
    size_t d_model = grad_output.rows();
    size_t seq_len = grad_output.cols();
    
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_model; ++j) {
            const_cast<double&>(gamma_.grad()(j, 0)) += 
                grad_output(j, i) * cached_normalized_(j, i);
            const_cast<double&>(beta_.grad()(j, 0)) += grad_output(j, i);
        }
    }
    
    math::Matrix grad_input(d_model, seq_len, 0.0);
    
    for (size_t i = 0; i < seq_len; ++i) {
        double mean = cached_mean_(0, i);
        double std = cached_std_(0, i);
        
        double sum_grad_normalized = 0.0;
        double sum_grad_normalized_times_normalized = 0.0;
        
        for (size_t j = 0; j < d_model; ++j) {
            double grad_normalized = grad_output(j, i) * gamma_.data()(j, 0);
            sum_grad_normalized += grad_normalized;
            sum_grad_normalized_times_normalized += grad_normalized * cached_normalized_(j, i);
        }
        
        for (size_t j = 0; j < d_model; ++j) {
            double grad_normalized = grad_output(j, i) * gamma_.data()(j, 0);
            const_cast<double&>(grad_input(j, i)) = 
                (grad_normalized - sum_grad_normalized / d_model - 
                 cached_normalized_(j, i) * sum_grad_normalized_times_normalized / d_model) / std;
        }
    }
    
    return grad_input;
}

} // namespace nn
} // namespace kaler
