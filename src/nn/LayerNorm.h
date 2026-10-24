#ifndef KALER_LAYERNORM_H
#define KALER_LAYERNORM_H

#include "Parameter.h"
#include "kaler/math/Matrix.hpp"

namespace kaler {
namespace nn {

class LayerNorm {
public:
    explicit LayerNorm(size_t normalized_shape, double eps = 1e-5);
    
    math::Matrix forward(const math::Matrix& input);
    math::Matrix backward(const math::Matrix& grad_output);
    
    Parameter& gamma() { return gamma_; }
    Parameter& beta() { return beta_; }

private:
    Parameter gamma_;
    Parameter beta_;
    double eps_;
    
    math::Matrix cached_input_;
    math::Matrix cached_mean_;
    math::Matrix cached_std_;
    math::Matrix cached_normalized_;
};

} // namespace nn
} // namespace kaler

#endif // KALER_LAYERNORM_H
