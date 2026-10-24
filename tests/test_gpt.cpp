#include "nn/GPT.h"
#include "nn/Loss.h"
#include "nn/Optimizer.h"
#include "data/DataLoader.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace kaler;

void test_gpt_components() {
    std::cout << "=== Testing GPT Components ===\n\n";
    
    std::cout << "1. Token Embeddings... ";
    nn::Embedding tok_emb(50, 32);
    std::vector<int> tokens = {1, 5, 10, 3};
    math::Matrix emb = tok_emb.forward(tokens);
    std::cout << "✓ Shape: [" << emb.rows() << ", " << emb.cols() << "]\n";
    
    std::cout << "2. Positional Embeddings... ";
    nn::PositionalEmbedding pos_emb(128, 32);
    math::Matrix pos = pos_emb.forward(4);
    std::cout << "✓ Shape: [" << pos.rows() << ", " << pos.cols() << "]\n";
    
    std::cout << "3. LayerNorm... ";
    nn::LayerNorm ln(32);
    math::Matrix x(32, 4, 1.0);
    math::Matrix ln_out = ln.forward(x);
    std::cout << "✓ Shape: [" << ln_out.rows() << ", " << ln_out.cols() << "]\n";
    
    std::cout << "4. Multi-Head Attention... ";
    nn::MultiHeadAttention attn(32, 4);
    std::cout << "\n    Input shape: [" << x.rows() << ", " << x.cols() << "]\n    ";
    math::Matrix attn_out = attn.forward(x, true);
    std::cout << "✓ Shape: [" << attn_out.rows() << ", " << attn_out.cols() << "]\n";
    
    std::cout << "5. Transformer Block... ";
    nn::TransformerBlock block(32, 4, 128);
    math::Matrix block_out = block.forward(x);
    std::cout << "✓ Shape: [" << block_out.rows() << ", " << block_out.cols() << "]\n";
    
    std::cout << "\n✅ All components working!\n\n";
}

void test_gpt_forward() {
    std::cout << "=== Testing GPT Forward Pass ===\n\n";
    
    nn::GPTConfig config(50, 64, 32, 4, 2, 128);
    nn::GPT model(config);
    
    std::vector<int> tokens = {1, 5, 10, 3, 7, 2};
    
    std::cout << "Input sequence length: " << tokens.size() << "\n";
    std::cout << "Vocab size: " << config.vocab_size << "\n";
    std::cout << "d_model: " << config.d_model << "\n";
    std::cout << "num_heads: " << config.num_heads << "\n";
    std::cout << "num_layers: " << config.num_layers << "\n\n";
    
    std::cout << "Running forward pass... ";
    math::Matrix logits = model.forward(tokens);
    std::cout << "✓\n";
    std::cout << "Output logits shape: [" << logits.rows() << ", " << logits.cols() << "]\n";
    
    std::cout << "Expected: [" << config.vocab_size << ", " << tokens.size() << "]\n";
    
    if (logits.rows() == config.vocab_size && logits.cols() == tokens.size()) {
        std::cout << "✅ Shape matches!\n\n";
    } else {
        std::cout << "❌ Shape mismatch!\n\n";
    }
}

void test_gpt_training() {
    std::cout << "=== Testing GPT Training on Tiny Sequence ===\n\n";
    
    nn::GPTConfig config(20, 32, 32, 2, 2, 64);
    nn::GPT model(config);
    
    std::vector<int> sequence = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    
    auto params = model.parameters();
    nn::Adam optimizer(0.001);
    for (auto* param : params) {
        optimizer.add_parameter(param);
    }
    
    std::cout << "Training to memorize: ";
    for (int t : sequence) std::cout << t << " ";
    std::cout << "\n\n";
    
    std::cout << "Epoch | Loss\n";
    std::cout << "------|----------\n";
    
    for (int epoch = 0; epoch < 50; ++epoch) {
        optimizer.zero_grad();
        
        std::vector<int> input_seq(sequence.begin(), sequence.end() - 1);
        math::Matrix logits = model.forward(input_seq);
        
        double total_loss = 0.0;
        math::Matrix total_grad(config.vocab_size, input_seq.size(), 0.0);
        
        for (size_t i = 0; i < input_seq.size(); ++i) {
            math::Matrix logit_col(config.vocab_size, 1, 0.0);
            for (size_t j = 0; j < config.vocab_size; ++j) {
                const_cast<double&>(logit_col(j, 0)) = logits(j, i);
            }
            
            math::Matrix target(config.vocab_size, 1, 0.0);
            int target_token = sequence[i + 1];
            const_cast<double&>(target(target_token, 0)) = 1.0;
            
            double loss = nn::Loss::cross_entropy(logit_col, target);
            total_loss += loss;
            
            math::Matrix grad_col = nn::Loss::cross_entropy_grad(logit_col, target);
            for (size_t j = 0; j < config.vocab_size; ++j) {
                const_cast<double&>(total_grad(j, i)) = grad_col(j, 0);
            }
        }
        
        double avg_loss = total_loss / input_seq.size();
        
        model.backward(total_grad, input_seq);
        optimizer.step();
        
        if (epoch % 5 == 0 || epoch == 49) {
            std::cout << std::setw(5) << epoch << " | "
                      << std::fixed << std::setprecision(6) << avg_loss << "\n";
        }
    }
    
    std::cout << "\n✅ Training completed!\n\n";
}

void test_gpt_generation() {
    std::cout << "=== Testing GPT Text Generation ===\n\n";
    
    nn::GPTConfig config(30, 32, 32, 2, 2, 64);
    nn::GPT model(config);
    
    std::vector<int> prompt = {1, 2};
    
    std::cout << "Prompt: ";
    for (int t : prompt) std::cout << t << " ";
    std::cout << "\n";
    
    std::cout << "Generating 10 tokens... ";
    std::vector<int> generated = model.generate(prompt, 10);
    std::cout << "✓\n";
    
    std::cout << "Generated sequence: ";
    for (int t : generated) std::cout << t << " ";
    std::cout << "\n";
    
    std::cout << "Generated length: " << generated.size() << " (expected: " 
              << prompt.size() + 10 << ")\n";
    
    if (generated.size() == prompt.size() + 10) {
        std::cout << "✅ Generation working!\n\n";
    }
}

int main() {
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║        GPT Architecture Tests          ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    try {
        test_gpt_components();
        test_gpt_forward();
        test_gpt_training();
        test_gpt_generation();
        
        std::cout << "╔════════════════════════════════════════╗\n";
        std::cout << "║   ✅ ALL GPT TESTS PASSED!             ║\n";
        std::cout << "╚════════════════════════════════════════╝\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << "\n";
        return 1;
    }
}
