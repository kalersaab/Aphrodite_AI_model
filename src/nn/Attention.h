#ifndef KALER_ATTENTION_H
#define KALER_ATTENTION_H

#include "Dense.h"
#include "kaler/math/Matrix.hpp"
#include <memory>
#include <vector>

namespace kaler {
namespace nn {

class MultiHeadAttention {
public:
    MultiHeadAttention(size_t d_model, size_t num_heads);
    
    math::Matrix forward(const math::Matrix& x, bool use_causal_mask = true);
    math::Matrix backward(const math::Matrix& grad_output);
    
    std::vector<Parameter*> parameters();

private:
    size_t d_model_;
    size_t num_heads_;
    size_t d_k_;
    
    std::unique_ptr<Dense> q_proj_;
    std::unique_ptr<Dense> k_proj_;
    std::unique_ptr<Dense> v_proj_;
    std::unique_ptr<Dense> out_proj_;
    
    math::Matrix cached_input_;
    math::Matrix cached_q_;
    math::Matrix cached_k_;
    math::Matrix cached_v_;
    math::Matrix cached_attn_weights_;
    math::Matrix cached_attn_output_;
    bool use_causal_mask_;
    
    math::Matrix apply_causal_mask(const math::Matrix& scores);
    math::Matrix softmax_rows(const math::Matrix& x);
    math::Matrix softmax_rows_backward(const math::Matrix& grad, const math::Matrix& probs);
};

} // namespace nn
} // namespace kaler

#endif // KALER_ATTENTION_H
