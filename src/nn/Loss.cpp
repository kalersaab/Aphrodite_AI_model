#include "Loss.h"
#include <limits>
#include <cmath>
#include <algorithm>

namespace kaler {
namespace nn {

math::Matrix Loss::softmax(const math::Matrix& logits) {
    math::Matrix output(logits.rows(), logits.cols(), 0.0);
    
    for (size_t j = 0; j < logits.cols(); ++j) {
        double max_val = logits(0, j);
        for (size_t i = 1; i < logits.rows(); ++i) {
            max_val = std::max(max_val, logits(i, j));
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < logits.rows(); ++i) {
            double val = std::exp(logits(i, j) - max_val);
            const_cast<double&>(output(i, j)) = val;
            sum += val;
        }
        
        // Numerical safety: handle underflow case
        if (sum < 1e-10 || !std::isfinite(sum)) {
            // Uniform distribution fallback
            for (size_t i = 0; i < logits.rows(); ++i) {
                const_cast<double&>(output(i, j)) = 1.0 / logits.rows();
            }
        } else {
            for (size_t i = 0; i < logits.rows(); ++i) {
                const_cast<double&>(output(i, j)) /= sum;
            }
        }
    }
    
    return output;
}

double Loss::cross_entropy(const math::Matrix& logits, const math::Matrix& targets) {
    math::Matrix probs = softmax(logits);
    
    double loss = 0.0;
    size_t batch_size = logits.cols();
    
    for (size_t j = 0; j < batch_size; ++j) {
        for (size_t i = 0; i < logits.rows(); ++i) {
            if (targets(i, j) > 0.0) {
                loss -= targets(i, j) * std::log(std::max(probs(i, j), 1e-10));
            }
        }
    }
    
    return loss / static_cast<double>(batch_size);
}

math::Matrix Loss::cross_entropy_grad(const math::Matrix& logits, const math::Matrix& targets) {
    math::Matrix probs = softmax(logits);
    math::Matrix grad(logits.rows(), logits.cols(), 0.0);
    
    size_t batch_size = logits.cols();
    
    for (size_t i = 0; i < grad.rows(); ++i) {
        for (size_t j = 0; j < grad.cols(); ++j) {
            const_cast<double&>(grad(i, j)) = (probs(i, j) - targets(i, j)) / static_cast<double>(batch_size);
        }
    }
    
    return grad;
}

} // namespace nn
} // namespace kaler
