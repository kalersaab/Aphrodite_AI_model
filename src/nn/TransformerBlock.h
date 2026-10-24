#ifndef KALER_TRANSFORMER_BLOCK_H
#define KALER_TRANSFORMER_BLOCK_H

#include "Attention.h"
#include "LayerNorm.h"
#include "Dense.h"
#include "kaler/math/Matrix.hpp"
#include <memory>
#include <vector>

namespace kaler {
namespace nn {

class FeedForward {
public:
    FeedForward(size_t d_model, size_t d_ff);
    
    math::Matrix forward(const math::Matrix& x);
    math::Matrix backward(const math::Matrix& grad_output);
    
    std::vector<Parameter*> parameters();

private:
    std::unique_ptr<Dense> fc1_;
    std::unique_ptr<Dense> fc2_;
    math::Matrix cached_hidden_;
};

class TransformerBlock {
public:
    TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff);
    
    math::Matrix forward(const math::Matrix& x);
    math::Matrix backward(const math::Matrix& grad_output);
    
    std::vector<Parameter*> parameters();

private:
    std::unique_ptr<LayerNorm> ln1_;
    std::unique_ptr<MultiHeadAttention> attn_;
    std::unique_ptr<LayerNorm> ln2_;
    std::unique_ptr<FeedForward> ff_;
    
    math::Matrix cached_x_;
    math::Matrix cached_ln1_out_;
    math::Matrix cached_attn_out_;
    math::Matrix cached_x_after_attn_;
    math::Matrix cached_ln2_out_;
    math::Matrix cached_ff_out_;
};

} // namespace nn
} // namespace kaler

#endif // KALER_TRANSFORMER_BLOCK_H
