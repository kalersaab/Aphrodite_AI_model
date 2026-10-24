#include "Attention.h"
#include <cmath>
#include <limits>

namespace kaler {
namespace nn {

MultiHeadAttention::MultiHeadAttention(size_t d_model, size_t num_heads)
    : d_model_(d_model)
    , num_heads_(num_heads)
    , d_k_(d_model / num_heads)
    , cached_input_(1, 1, 0.0)
    , cached_q_(1, 1, 0.0)
    , cached_k_(1, 1, 0.0)
    , cached_v_(1, 1, 0.0)
    , cached_attn_weights_(1, 1, 0.0)
    , cached_attn_output_(1, 1, 0.0)
    , use_causal_mask_(false) {
    
    q_proj_ = std::make_unique<Dense>(d_model, d_model, false);
    k_proj_ = std::make_unique<Dense>(d_model, d_model, false);
    v_proj_ = std::make_unique<Dense>(d_model, d_model, false);
    out_proj_ = std::make_unique<Dense>(d_model, d_model, false);
}

math::Matrix MultiHeadAttention::softmax_rows(const math::Matrix& x) {
    math::Matrix result(x.rows(), x.cols(), 0.0);
    
    for (size_t i = 0; i < x.rows(); ++i) {
        double max_val = x(i, 0);
        for (size_t j = 1; j < x.cols(); ++j) {
            max_val = std::max(max_val, x(i, j));
        }
        
        double sum = 0.0;
        for (size_t j = 0; j < x.cols(); ++j) {
            double val = std::exp(x(i, j) - max_val);
            const_cast<double&>(result(i, j)) = val;
            sum += val;
        }
        
        for (size_t j = 0; j < x.cols(); ++j) {
            const_cast<double&>(result(i, j)) /= sum;
        }
    }
    
    return result;
}

math::Matrix MultiHeadAttention::softmax_rows_backward(const math::Matrix& grad, const math::Matrix& probs) {
    math::Matrix grad_input(grad.rows(), grad.cols(), 0.0);
    
    for (size_t i = 0; i < grad.rows(); ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < grad.cols(); ++j) {
            sum += grad(i, j) * probs(i, j);
        }
        
        for (size_t j = 0; j < grad.cols(); ++j) {
            const_cast<double&>(grad_input(i, j)) = probs(i, j) * (grad(i, j) - sum);
        }
    }
    
    return grad_input;
}

math::Matrix MultiHeadAttention::apply_causal_mask(const math::Matrix& scores) {
    math::Matrix masked(scores.rows(), scores.cols(), 0.0);
    
    for (size_t i = 0; i < scores.rows(); ++i) {
        for (size_t j = 0; j < scores.cols(); ++j) {
            if (j <= i) {
                const_cast<double&>(masked(i, j)) = scores(i, j);
            } else {
                const_cast<double&>(masked(i, j)) = -1e9;
            }
        }
    }
    
    return masked;
}

math::Matrix MultiHeadAttention::forward(const math::Matrix& x, bool use_causal_mask) {
    cached_input_ = x;
    use_causal_mask_ = use_causal_mask;
    
    cached_q_ = q_proj_->forward(x);
    cached_k_ = k_proj_->forward(x);
    cached_v_ = v_proj_->forward(x);
    
    double scale = 1.0 / std::sqrt(static_cast<double>(d_k_));
    
    math::Matrix q_t = cached_q_.transpose();
    math::Matrix scores = matmul(q_t, cached_k_);
    
    for (size_t i = 0; i < scores.rows(); ++i) {
        for (size_t j = 0; j < scores.cols(); ++j) {
            const_cast<double&>(scores(i, j)) *= scale;
        }
    }
    
    if (use_causal_mask) {
        scores = apply_causal_mask(scores);
    }
    
    cached_attn_weights_ = softmax_rows(scores);
    
    math::Matrix v_t = cached_v_.transpose();
    math::Matrix attn_output_T = matmul(cached_attn_weights_, v_t);
    cached_attn_output_ = attn_output_T.transpose();
    
    math::Matrix output = out_proj_->forward(cached_attn_output_);
    
    return output;
}

math::Matrix MultiHeadAttention::backward(const math::Matrix& grad_output) {
    math::Matrix grad_attn_output = out_proj_->backward(grad_output);
    
    math::Matrix grad_attn_output_T = grad_attn_output.transpose();
    
    math::Matrix v_t = cached_v_.transpose();
    math::Matrix grad_weights = matmul(grad_attn_output_T, v_t.transpose());
    
    math::Matrix grad_v_t = matmul(cached_attn_weights_.transpose(), grad_attn_output_T);
    math::Matrix grad_v = grad_v_t.transpose();
    
    math::Matrix grad_scores = softmax_rows_backward(grad_weights, cached_attn_weights_);
    
    double scale = 1.0 / std::sqrt(static_cast<double>(d_k_));
    for (size_t i = 0; i < grad_scores.rows(); ++i) {
        for (size_t j = 0; j < grad_scores.cols(); ++j) {
            const_cast<double&>(grad_scores(i, j)) *= scale;
        }
    }
    
    if (use_causal_mask_) {
        for (size_t i = 0; i < grad_scores.rows(); ++i) {
            for (size_t j = 0; j < grad_scores.cols(); ++j) {
                if (j > i) {
                    const_cast<double&>(grad_scores(i, j)) = 0.0;
                }
            }
        }
    }
    
    math::Matrix k_t = cached_k_.transpose();
    math::Matrix grad_q_t = matmul(grad_scores, k_t);
    math::Matrix grad_q = grad_q_t.transpose();
    
    math::Matrix q_t = cached_q_.transpose();
    math::Matrix grad_k_t = matmul(grad_scores.transpose(), q_t);
    math::Matrix grad_k = grad_k_t.transpose();
    
    math::Matrix grad_x_q = q_proj_->backward(grad_q);
    math::Matrix grad_x_k = k_proj_->backward(grad_k);
    math::Matrix grad_x_v = v_proj_->backward(grad_v);
    
    math::Matrix grad_input(grad_x_q.rows(), grad_x_q.cols(), 0.0);
    for (size_t i = 0; i < grad_input.rows(); ++i) {
        for (size_t j = 0; j < grad_input.cols(); ++j) {
            const_cast<double&>(grad_input(i, j)) = 
                grad_x_q(i, j) + grad_x_k(i, j) + grad_x_v(i, j);
        }
    }
    
    return grad_input;
}

std::vector<Parameter*> MultiHeadAttention::parameters() {
    return {
        &q_proj_->weight(),
        &k_proj_->weight(),
        &v_proj_->weight(),
        &out_proj_->weight()
    };
}

} // namespace nn
} // namespace kaler
