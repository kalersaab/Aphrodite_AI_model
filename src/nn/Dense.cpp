#include "Dense.h"
#include "kaler/math/Matrix.hpp"

namespace kaler {
namespace nn {

Dense::Dense(size_t input_size, size_t output_size, bool use_bias)
    : weight_(output_size, input_size, "weight")
    , bias_(output_size, 1, "bias")
    , use_bias_(use_bias)
    , cached_input_(1, 1, 0.0) {
    
    weight_.xavier_init();
    bias_.zero_init();
}

math::Matrix Dense::forward(const math::Matrix& input) {
    cached_input_ = input;
    
    math::Matrix output = matmul(weight_.data(), input);
    
    if (use_bias_) {
        for (size_t i = 0; i < output.rows(); ++i) {
            for (size_t j = 0; j < output.cols(); ++j) {
                const_cast<double&>(output(i, j)) += bias_.data()(i, 0);
            }
        }
    }
    
    return output;
}

math::Matrix Dense::backward(const math::Matrix& grad_output) {
    math::Matrix input_T = cached_input_.transpose();
    weight_.grad() = matmul(grad_output, input_T);
    
    if (use_bias_) {
        for (size_t i = 0; i < grad_output.rows(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < grad_output.cols(); ++j) {
                sum += grad_output(i, j);
            }
            const_cast<double&>(bias_.grad()(i, 0)) = sum;
        }
    }
    
    math::Matrix weight_T = weight_.data().transpose();
    math::Matrix grad_input = matmul(weight_T, grad_output);
    
    return grad_input;
}

} // namespace nn
} // namespace kaler
