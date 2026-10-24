#ifndef KALER_EMBEDDING_H
#define KALER_EMBEDDING_H

#include "Parameter.h"
#include "kaler/math/Matrix.hpp"
#include <vector>

namespace kaler {
namespace nn {

class Embedding {
public:
    Embedding(size_t vocab_size, size_t embed_dim);
    
    math::Matrix forward(const std::vector<int>& token_ids);
    void backward(const math::Matrix& grad_output, const std::vector<int>& token_ids);
    
    Parameter& weight() { return weight_; }

private:
    Parameter weight_;
    size_t vocab_size_;
    size_t embed_dim_;
};

class PositionalEmbedding {
public:
    PositionalEmbedding(size_t max_seq_len, size_t embed_dim);
    
    math::Matrix forward(size_t seq_len);
    void backward(const math::Matrix& grad_output);
    
    Parameter& weight() { return weight_; }

private:
    Parameter weight_;
    size_t max_seq_len_;
    size_t embed_dim_;
};

} // namespace nn
} // namespace kaler

#endif // KALER_EMBEDDING_H
