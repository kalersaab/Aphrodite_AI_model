#include "GPT.h"
#include "Loss.h"
#include <random>
#include <algorithm>

namespace kaler {
namespace nn {

GPT::GPT(const GPTConfig& config)
    : config_(config)
    , cached_embeddings_(1, 1, 0.0) {
    
    token_embedding_ = std::make_unique<Embedding>(config.vocab_size, config.d_model);
    pos_embedding_ = std::make_unique<PositionalEmbedding>(config.max_seq_len, config.d_model);
    
    for (size_t i = 0; i < config.num_layers; ++i) {
        blocks_.push_back(std::make_unique<TransformerBlock>(
            config.d_model, config.num_heads, config.d_ff));
    }
    
    ln_f_ = std::make_unique<LayerNorm>(config.d_model);
    lm_head_ = std::make_unique<Dense>(config.d_model, config.vocab_size, false);
}

math::Matrix GPT::forward(const std::vector<int>& token_ids) {
    cached_token_ids_ = token_ids;
    size_t seq_len = token_ids.size();
    
    math::Matrix tok_emb = token_embedding_->forward(token_ids);
    math::Matrix pos_emb = pos_embedding_->forward(seq_len);
    
    cached_embeddings_ = math::Matrix(config_.d_model, seq_len, 0.0);
    for (size_t i = 0; i < config_.d_model; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            const_cast<double&>(cached_embeddings_(i, j)) = tok_emb(i, j) + pos_emb(i, j);
        }
    }
    
    math::Matrix x = cached_embeddings_;
    
    for (auto& block : blocks_) {
        x = block->forward(x);
    }
    
    x = ln_f_->forward(x);
    
    math::Matrix logits = lm_head_->forward(x);
    
    return logits;
}

void GPT::backward(const math::Matrix& grad_output, const std::vector<int>& token_ids) {
    math::Matrix grad = lm_head_->backward(grad_output);
    
    grad = ln_f_->backward(grad);
    
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
    
    size_t seq_len = token_ids.size();
    pos_embedding_->backward(grad);
    token_embedding_->backward(grad, token_ids);
}

double GPT::compute_loss(const std::vector<int>& token_ids) {
    if (token_ids.size() < 2) {
        return 0.0;
    }
    
    std::vector<int> input_ids(token_ids.begin(), token_ids.end() - 1);
    
    math::Matrix logits = forward(input_ids);
    
    size_t seq_len = input_ids.size();
    double total_loss = 0.0;
    
    for (size_t i = 0; i < seq_len; ++i) {
        math::Matrix logit_col(config_.vocab_size, 1, 0.0);
        for (size_t j = 0; j < config_.vocab_size; ++j) {
            const_cast<double&>(logit_col(j, 0)) = logits(j, i);
        }
        
        math::Matrix target(config_.vocab_size, 1, 0.0);
        int target_token = token_ids[i + 1];
        const_cast<double&>(target(target_token, 0)) = 1.0;
        
        double loss = Loss::cross_entropy(logit_col, target);
        total_loss += loss;
    }
    
    return total_loss / seq_len;
}

std::vector<Parameter*> GPT::parameters() {
    std::vector<Parameter*> params;
    
    params.push_back(&token_embedding_->weight());
    params.push_back(&pos_embedding_->weight());
    
    for (auto& block : blocks_) {
        auto block_params = block->parameters();
        params.insert(params.end(), block_params.begin(), block_params.end());
    }
    
    params.push_back(&ln_f_->gamma());
    params.push_back(&ln_f_->beta());
    params.push_back(&lm_head_->weight());
    
    return params;
}

std::vector<int> GPT::generate(const std::vector<int>& prompt, size_t max_new_tokens) {
    std::vector<int> tokens = prompt;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < max_new_tokens; ++i) {
        size_t context_len = std::min(tokens.size(), config_.max_seq_len);
        std::vector<int> context(tokens.end() - context_len, tokens.end());
        
        math::Matrix logits = forward(context);
        
        math::Matrix last_logits(config_.vocab_size, 1, 0.0);
        for (size_t j = 0; j < config_.vocab_size; ++j) {
            const_cast<double&>(last_logits(j, 0)) = logits(j, context_len - 1);
        }
        
        math::Matrix probs = Loss::softmax(last_logits);
        
        std::vector<double> prob_vec(config_.vocab_size);
        for (size_t j = 0; j < config_.vocab_size; ++j) {
            prob_vec[j] = probs(j, 0);
        }
        
        std::discrete_distribution<> dist(prob_vec.begin(), prob_vec.end());
        int next_token = dist(gen);
        
        tokens.push_back(next_token);
    }
    
    return tokens;
}

} // namespace nn
} // namespace kaler
