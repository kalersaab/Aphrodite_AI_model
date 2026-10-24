#include "../src/nn/GPT.h"
#include "../src/nn/Optimizer.h"
#include "../src/nn/Loss.h"
#include "../src/data/DataLoader.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

using namespace kaler;
using namespace kaler::nn;

// Test gradient flow through the full model
void test_full_model_gradient_flow() {
    std::cout << "\n=== Test: Full Model Gradient Flow ===\n";
    
    // Small model for testing
    GPTConfig config;
    config.vocab_size = 20;
    config.max_seq_len = 8;
    config.d_model = 16;
    config.num_heads = 4;
    config.num_layers = 2;
    config.d_ff = 32;
    
    GPT model(config);
    Adam optimizer(0.001);
    
    // Register all parameters
    auto params = model.parameters();
    for (auto* p : params) {
        optimizer.add_parameter(p);
    }
    
    std::cout << "Model has " << params.size() << " parameters\n";
    
    // Create simple training data
    std::vector<int> input_ids = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> target_ids = {2, 3, 4, 5, 6, 7, 8, 9};
    
    // Create one-hot targets
    math::Matrix targets(config.vocab_size, input_ids.size(), 0.0);
    for (size_t t = 0; t < target_ids.size(); ++t) {
        if (target_ids[t] < static_cast<int>(config.vocab_size)) {
            targets(static_cast<size_t>(target_ids[t]), t) = 1.0;
        }
    }
    
    // Training loop
    double initial_loss = 0.0;
    double final_loss = 0.0;
    
    std::cout << "Running training steps...\n";
    for (int step = 0; step < 50; ++step) {
        optimizer.zero_grad();
        
        // Forward pass
        math::Matrix logits = model.forward(input_ids);
        double loss = Loss::cross_entropy(logits, targets);
        
        if (step == 0) initial_loss = loss;
        if (step == 49) final_loss = loss;
        
        // Backward pass
        math::Matrix grad_loss = Loss::cross_entropy_grad(logits, targets);
        model.backward(grad_loss, input_ids);
        
        // Check for non-finite gradients
        bool has_nonfinite = false;
        double max_grad = 0.0;
        double grad_norm = 0.0;
        
        for (const auto* p : params) {
            const auto& g = p->grad();
            for (size_t r = 0; r < g.rows(); ++r) {
                for (size_t c = 0; c < g.cols(); ++c) {
                    double val = g(r, c);
                    if (!std::isfinite(val)) {
                        has_nonfinite = true;
                    }
                    max_grad = std::max(max_grad, std::abs(val));
                    grad_norm += val * val;
                }
            }
        }
        grad_norm = std::sqrt(grad_norm);
        
        if (has_nonfinite) {
            std::cout << "✗ FAILED: Non-finite gradients at step " << step << "\n";
            return;
        }
        
        // Gradient clipping
        double clip_norm = 1.0;
        if (grad_norm > clip_norm) {
            double scale = clip_norm / grad_norm;
            for (auto* p : params) {
                auto& g = p->grad();
                for (size_t r = 0; r < g.rows(); ++r) {
                    for (size_t c = 0; c < g.cols(); ++c) {
                        g(r, c) *= scale;
                    }
                }
            }
        }
        
        // Update
        optimizer.step();
        
        if (step % 10 == 0) {
            std::cout << "  Step " << step << ": loss=" << loss 
                      << ", grad_norm=" << grad_norm << ", max_grad=" << max_grad << "\n";
        }
    }
    
    std::cout << "Initial loss: " << initial_loss << "\n";
    std::cout << "Final loss: " << final_loss << "\n";
    std::cout << "Loss reduction: " << (initial_loss - final_loss) << "\n";
    
    if (final_loss < initial_loss && std::isfinite(final_loss)) {
        std::cout << "✓ PASSED: Model trained successfully with decreasing loss\n";
    } else {
        std::cout << "✗ FAILED: Loss did not decrease or is non-finite\n";
    }
}

// Test parameter weight magnitudes after many updates
void test_parameter_stability_over_time() {
    std::cout << "\n=== Test: Parameter Stability Over Time ===\n";
    
    nn::Parameter param(10, 10, "test_param");
    param.xavier_init();
    
    Adam optimizer(0.01);
    optimizer.add_parameter(&param);
    
    // Simulate many updates with random gradients
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 0.1);
    
    for (int step = 0; step < 1000; ++step) {
        param.zero_grad();
        
        // Inject random gradients
        for (size_t i = 0; i < param.data().rows(); ++i) {
            for (size_t j = 0; j < param.data().cols(); ++j) {
                const_cast<double&>(param.grad()(i, j)) = dist(rng);
            }
        }
        
        optimizer.step();
    }
    
    // Check parameter statistics
    double min_val = param.data()(0, 0);
    double max_val = param.data()(0, 0);
    double sum = 0.0;
    int count = 0;
    bool all_finite = true;
    
    for (size_t i = 0; i < param.data().rows(); ++i) {
        for (size_t j = 0; j < param.data().cols(); ++j) {
            double val = param.data()(i, j);
            if (!std::isfinite(val)) {
                all_finite = false;
            }
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
            count++;
        }
    }
    
    double mean = sum / count;
    
    std::cout << "After 1000 updates:\n";
    std::cout << "  Min: " << min_val << "\n";
    std::cout << "  Max: " << max_val << "\n";
    std::cout << "  Mean: " << mean << "\n";
    std::cout << "  All finite: " << (all_finite ? "yes" : "no") << "\n";
    
    if (all_finite && std::abs(min_val) < 1000.0 && std::abs(max_val) < 1000.0) {
        std::cout << "✓ PASSED: Parameters remain stable and finite\n";
    } else {
        std::cout << "✗ FAILED: Parameters became unstable\n";
    }
}

// Test gradient accumulation across multiple backward passes
void test_multi_backward_accumulation() {
    std::cout << "\n=== Test: Multi-Backward Gradient Accumulation ===\n";
    
    Dense layer(5, 3, false);
    layer.weight().zero_init();
    
    // Set specific weight values
    const_cast<double&>(layer.weight().data()(0, 0)) = 1.0;
    const_cast<double&>(layer.weight().data()(1, 1)) = 1.0;
    const_cast<double&>(layer.weight().data()(2, 2)) = 1.0;
    
    math::Matrix input1(5, 1, 1.0);
    math::Matrix input2(5, 1, 2.0);
    
    math::Matrix grad_output(3, 1, 1.0);
    
    // First forward-backward
    layer.forward(input1);
    layer.backward(grad_output);
    
    double grad_after_first = layer.weight().grad()(0, 0);
    std::cout << "Gradient after first backward: " << grad_after_first << "\n";
    
    // Second forward-backward WITHOUT zeroing
    layer.forward(input2);
    layer.backward(grad_output);
    
    double grad_after_second = layer.weight().grad()(0, 0);
    std::cout << "Gradient after second backward: " << grad_after_second << "\n";
    
    // Expected: first backward contributes input1, second contributes input2
    // So gradient should be input1[0] + input2[0] = 1.0 + 2.0 = 3.0
    double expected = 3.0;
    double diff = std::abs(grad_after_second - expected);
    
    if (diff < 1e-6) {
        std::cout << "✓ PASSED: Gradients accumulate correctly across multiple backwards\n";
    } else {
        std::cout << "✗ FAILED: Expected " << expected << ", got " << grad_after_second << "\n";
    }
}

int main() {
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║     Aphrodite Gradient Flow Test Suite        ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    
    test_multi_backward_accumulation();
    test_parameter_stability_over_time();
    test_full_model_gradient_flow();
    
    std::cout << "\n╔════════════════════════════════════════════════╗\n";
    std::cout << "║         Test Suite Complete                    ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    
    return 0;
}
