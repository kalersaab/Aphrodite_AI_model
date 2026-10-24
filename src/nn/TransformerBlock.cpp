#include "TransformerBlock.h"
#include <cmath>

namespace kaler {
namespace nn {

FeedForward::FeedForward(size_t d_model, size_t d_ff)
    : cached_hidden_(1, 1, 0.0) {
    fc1_ = std::make_unique<Dense>(d_model, d_ff);
    fc2_ = std::make_unique<Dense>(d_ff, d_model);
}

math::Matrix FeedForward::forward(const math::Matrix& x) {
    math::Matrix hidden = fc1_->forward(x);
    
    cached_hidden_ = math::Matrix(hidden.rows(), hidden.cols(), 0.0);
    for (size_t i = 0; i < hidden.rows(); ++i) {
        for (size_t j = 0; j < hidden.cols(); ++j) {
            double val = std::max(0.0, hidden(i, j));
            const_cast<double&>(cached_hidden_(i, j)) = val;
        }
    }
    
    math::Matrix output = fc2_->forward(cached_hidden_);
    return output;
}

math::Matrix FeedForward::backward(const math::Matrix& grad_output) {
    math::Matrix grad_hidden = fc2_->backward(grad_output);
    
    math::Matrix grad_hidden_relu(grad_hidden.rows(), grad_hidden.cols(), 0.0);
    for (size_t i = 0; i < grad_hidden.rows(); ++i) {
        for (size_t j = 0; j < grad_hidden.cols(); ++j) {
            if (cached_hidden_(i, j) > 0.0) {
                const_cast<double&>(grad_hidden_relu(i, j)) = grad_hidden(i, j);
            }
        }
    }
    
    math::Matrix grad_input = fc1_->backward(grad_hidden_relu);
    return grad_input;
}

std::vector<Parameter*> FeedForward::parameters() {
    return {
        &fc1_->weight(),
        &fc1_->bias(),
        &fc2_->weight(),
        &fc2_->bias()
    };
}

TransformerBlock::TransformerBlock(size_t d_model, size_t num_heads, size_t d_ff)
    : cached_x_(1, 1, 0.0)
    , cached_ln1_out_(1, 1, 0.0)
    , cached_attn_out_(1, 1, 0.0)
    , cached_x_after_attn_(1, 1, 0.0)
    , cached_ln2_out_(1, 1, 0.0)
    , cached_ff_out_(1, 1, 0.0) {
    
    ln1_ = std::make_unique<LayerNorm>(d_model);
    attn_ = std::make_unique<MultiHeadAttention>(d_model, num_heads);
    ln2_ = std::make_unique<LayerNorm>(d_model);
    ff_ = std::make_unique<FeedForward>(d_model, d_ff);
}

math::Matrix TransformerBlock::forward(const math::Matrix& x) {
    cached_x_ = x;
    
    cached_ln1_out_ = ln1_->forward(x);
    
    cached_attn_out_ = attn_->forward(cached_ln1_out_, true);
    
    cached_x_after_attn_ = math::Matrix(x.rows(), x.cols(), 0.0);
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            const_cast<double&>(cached_x_after_attn_(i, j)) = 
                x(i, j) + cached_attn_out_(i, j);
        }
    }
    
    cached_ln2_out_ = ln2_->forward(cached_x_after_attn_);
    
    cached_ff_out_ = ff_->forward(cached_ln2_out_);
    
    math::Matrix output(x.rows(), x.cols(), 0.0);
    for (size_t i = 0; i < x.rows(); ++i) {
        for (size_t j = 0; j < x.cols(); ++j) {
            const_cast<double&>(output(i, j)) = 
                cached_x_after_attn_(i, j) + cached_ff_out_(i, j);
        }
    }
    
    return output;
}

math::Matrix TransformerBlock::backward(const math::Matrix& grad_output) {
    math::Matrix grad_ff_out = grad_output;
    math::Matrix grad_x_after_attn_2 = grad_output;
    
    math::Matrix grad_ln2_out = ff_->backward(grad_ff_out);
    
    math::Matrix grad_x_after_attn_1 = ln2_->backward(grad_ln2_out);
    
    math::Matrix grad_x_after_attn(grad_output.rows(), grad_output.cols(), 0.0);
    for (size_t i = 0; i < grad_output.rows(); ++i) {
        for (size_t j = 0; j < grad_output.cols(); ++j) {
            const_cast<double&>(grad_x_after_attn(i, j)) = 
                grad_x_after_attn_1(i, j) + grad_x_after_attn_2(i, j);
        }
    }
    
    math::Matrix grad_attn_out = grad_x_after_attn;
    math::Matrix grad_x_1 = grad_x_after_attn;
    
    math::Matrix grad_ln1_out = attn_->backward(grad_attn_out);
    
    math::Matrix grad_x_2 = ln1_->backward(grad_ln1_out);
    
    math::Matrix grad_input(grad_output.rows(), grad_output.cols(), 0.0);
    for (size_t i = 0; i < grad_output.rows(); ++i) {
        for (size_t j = 0; j < grad_output.cols(); ++j) {
            const_cast<double&>(grad_input(i, j)) = grad_x_1(i, j) + grad_x_2(i, j);
        }
    }
    
    return grad_input;
}

std::vector<Parameter*> TransformerBlock::parameters() {
    std::vector<Parameter*> params;
    
    params.push_back(&ln1_->gamma());
    params.push_back(&ln1_->beta());
    
    auto attn_params = attn_->parameters();
    params.insert(params.end(), attn_params.begin(), attn_params.end());
    
    params.push_back(&ln2_->gamma());
    params.push_back(&ln2_->beta());
    
    auto ff_params = ff_->parameters();
    params.insert(params.end(), ff_params.begin(), ff_params.end());
    
    return params;
}

} // namespace nn
} // namespace kaler
