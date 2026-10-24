#include "Embedding.h"
#include <cmath>

namespace kaler {
namespace nn {

Embedding::Embedding(size_t vocab_size, size_t embed_dim)
    : weight_(vocab_size, embed_dim, "token_embedding")
    , vocab_size_(vocab_size)
    , embed_dim_(embed_dim) {
    weight_.random_init(-0.02, 0.02);
}

math::Matrix Embedding::forward(const std::vector<int>& token_ids) {
    size_t seq_len = token_ids.size();
    math::Matrix output(embed_dim_, seq_len, 0.0);
    
    for (size_t i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        for (size_t j = 0; j < embed_dim_; ++j) {
            const_cast<double&>(output(j, i)) = weight_.data()(token_id, j);
        }
    }
    
    return output;
}

void Embedding::backward(const math::Matrix& grad_output, const std::vector<int>& token_ids) {
    size_t seq_len = token_ids.size();
    
    for (size_t i = 0; i < seq_len; ++i) {
        int token_id = token_ids[i];
        for (size_t j = 0; j < embed_dim_; ++j) {
            const_cast<double&>(weight_.grad()(token_id, j)) += grad_output(j, i);
        }
    }
}

PositionalEmbedding::PositionalEmbedding(size_t max_seq_len, size_t embed_dim)
    : weight_(max_seq_len, embed_dim, "positional_embedding")
    , max_seq_len_(max_seq_len)
    , embed_dim_(embed_dim) {
    
    for (size_t pos = 0; pos < max_seq_len; ++pos) {
        for (size_t i = 0; i < embed_dim; ++i) {
            double angle = pos / std::pow(10000.0, (2.0 * i) / embed_dim);
            if (i % 2 == 0) {
                const_cast<double&>(weight_.data()(pos, i)) = std::sin(angle);
            } else {
                const_cast<double&>(weight_.data()(pos, i)) = std::cos(angle);
            }
        }
    }
}

math::Matrix PositionalEmbedding::forward(size_t seq_len) {
    math::Matrix output(embed_dim_, seq_len, 0.0);
    
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < embed_dim_; ++j) {
            const_cast<double&>(output(j, i)) = weight_.data()(i, j);
        }
    }
    
    return output;
}

void PositionalEmbedding::backward(const math::Matrix& grad_output) {
    for (size_t i = 0; i < grad_output.cols(); ++i) {
        for (size_t j = 0; j < grad_output.rows(); ++j) {
            const_cast<double&>(weight_.grad()(i, j)) += grad_output(j, i);
        }
    }
}

} // namespace nn
} // namespace kaler
