#ifndef APHRODITE_GPT_H
#define APHRODITE_GPT_H

#include "Embedding.h"
#include "TransformerBlock.h"
#include "LayerNorm.h"
#include "Dense.h"
#include "kaler/math/Matrix.hpp"
#include <memory>
#include <vector>

namespace kaler {
namespace nn {

struct GPTConfig {
    size_t vocab_size;
    size_t max_seq_len;
    size_t d_model;
    size_t num_heads;
    size_t num_layers;
    size_t d_ff;
    
    GPTConfig(size_t vocab = 100, size_t seq = 128, size_t model = 64,
              size_t heads = 4, size_t layers = 4, size_t ff = 256)
        : vocab_size(vocab)
        , max_seq_len(seq)
        , d_model(model)
        , num_heads(heads)
        , num_layers(layers)
        , d_ff(ff) {}
};

class GPT {
public:
    explicit GPT(const GPTConfig& config);
    
    math::Matrix forward(const std::vector<int>& token_ids);
    
    void backward(const math::Matrix& grad_output, const std::vector<int>& token_ids);
    
    double compute_loss(const std::vector<int>& token_ids);
    
    std::vector<Parameter*> parameters();
    
    std::vector<int> generate(const std::vector<int>& prompt, size_t max_new_tokens);

private:
    GPTConfig config_;
    
    std::unique_ptr<Embedding> token_embedding_;
    std::unique_ptr<PositionalEmbedding> pos_embedding_;
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;
    std::unique_ptr<LayerNorm> ln_f_;
    std::unique_ptr<Dense> lm_head_;
    
    std::vector<int> cached_token_ids_;
    math::Matrix cached_embeddings_;
};

} // namespace nn
} // namespace kaler

#endif // APHRODITE_GPT_H
