#ifndef KALER_DENSE_H
#define KALER_DENSE_H

#include "Parameter.h"
#include "kaler/math/Matrix.hpp"
#include <memory>

namespace kaler {
namespace nn {

class Dense {
public:
    Dense(size_t input_size, size_t output_size, bool use_bias = true);
    
    math::Matrix forward(const math::Matrix& input);
    math::Matrix backward(const math::Matrix& grad_output);
    
    Parameter& weight() { return weight_; }
    Parameter& bias() { return bias_; }
    
    const math::Matrix& cached_input() const { return cached_input_; }

private:
    Parameter weight_;
    Parameter bias_;
    bool use_bias_;
    
    math::Matrix cached_input_;
};

} // namespace nn
} // namespace kaler

#endif // KALER_DENSE_H
