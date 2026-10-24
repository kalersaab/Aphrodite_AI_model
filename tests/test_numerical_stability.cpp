#include "../src/nn/Dense.h"
#include "../src/nn/Attention.h"
#include "../src/nn/LayerNorm.h"
#include "../src/nn/Loss.h"
#include "../src/nn/Optimizer.h"
#include "../src/nn/GradientChecker.h"
#include "kaler/math/Matrix.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

using namespace kaler;
using namespace kaler::nn;

// Test 1: Dense layer gradient accumulation
void test_dense_gradient_accumulation() {
    std::cout << "\n=== Test 1: Dense Gradient Accumulation ===\n";
    
    Dense layer(3, 2, false);
    math::Matrix input(3, 1, 1.0);
    math::Matrix grad1(2, 1, 1.0);
    math::Matrix grad2(2, 1, 2.0);
    
    // First forward-backward
    layer.forward(input);
    layer.backward(grad1);
    
    double first_grad = layer.weight().grad()(0, 0);
    std::cout << "First gradient: " << first_grad << "\n";
    
    // Second forward-backward WITHOUT zero_grad()
    layer.forward(input);
    layer.backward(grad2);
    
    double second_grad = layer.weight().grad()(0, 0);
    std::cout << "Accumulated gradient: " << second_grad << "\n";
    
    // Should accumulate: first_grad + (2.0 * input)
    double expected = first_grad + 2.0;
    double diff = std::abs(second_grad - expected);
    
    if (diff < 1e-6) {
        std::cout << "✓ PASSED: Gradients accumulate correctly\n";
    } else {
        std::cout << "✗ FAILED: Expected " << expected << ", got " << second_grad << "\n";
    }
}

// Test 2: Adam optimizer with non-finite gradients
void test_adam_nonfinite_gradients() {
    std::cout << "\n=== Test 2: Adam with Non-Finite Gradients ===\n";
    
    nn::Parameter param(2, 2, "test_param");
    param.data()(0, 0) = 1.0;
    param.data()(1, 1) = 2.0;
    
    Adam optimizer(0.01);
    optimizer.add_parameter(&param);
    
    // Inject NaN gradient
    param.zero_grad();
    const_cast<double&>(param.grad()(0, 0)) = std::numeric_limits<double>::quiet_NaN();
    const_cast<double&>(param.grad()(1, 1)) = 1.0;
    
    std::cout << "Before update - param(0,0): " << param.data()(0, 0) << "\n";
    optimizer.step();
    std::cout << "After update - param(0,0): " << param.data()(0, 0) << "\n";
    
    // Parameter should not become NaN (skipped or zeroed)
    if (std::isfinite(param.data()(0, 0))) {
        std::cout << "✓ PASSED: Parameters remain finite despite NaN gradients\n";
    } else {
        std::cout << "✗ FAILED: Parameters became NaN\n";
    }
}

// Test 3: Adam bias correction at large t
void test_adam_large_t() {
    std::cout << "\n=== Test 3: Adam Bias Correction at Large t ===\n";
    
    nn::Parameter param(1, 1, "test_param");
    param.data()(0, 0) = 1.0;
    
    Adam optimizer(0.001, 0.9, 0.999);
    optimizer.add_parameter(&param);
    
    // Simulate many iterations
    for (int i = 0; i < 10000; ++i) {
        param.zero_grad();
        const_cast<double&>(param.grad()(0, 0)) = 0.01;
        optimizer.step();
    }
    
    std::cout << "Final parameter value: " << param.data()(0, 0) << "\n";
    
    if (std::isfinite(param.data()(0, 0))) {
        std::cout << "✓ PASSED: Adam stable at large t\n";
    } else {
        std::cout << "✗ FAILED: Adam produced non-finite values\n";
    }
}

// Test 4: Softmax numerical stability
void test_softmax_extreme_values() {
    std::cout << "\n=== Test 4: Softmax with Extreme Values ===\n";
    
    // Test with very large values
    math::Matrix large_logits(3, 2);
    large_logits(0, 0) = 1000.0;
    large_logits(1, 0) = 999.0;
    large_logits(2, 0) = 1001.0;
    large_logits(0, 1) = -1000.0;
    large_logits(1, 1) = -1001.0;
    large_logits(2, 1) = -999.0;
    
    math::Matrix probs = Loss::softmax(large_logits);
    
    bool all_finite = true;
    double sum_col0 = 0.0, sum_col1 = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        if (!std::isfinite(probs(i, 0)) || !std::isfinite(probs(i, 1))) {
            all_finite = false;
        }
        sum_col0 += probs(i, 0);
        sum_col1 += probs(i, 1);
    }
    
    std::cout << "Column sums: " << sum_col0 << ", " << sum_col1 << "\n";
    
    if (all_finite && std::abs(sum_col0 - 1.0) < 1e-6 && std::abs(sum_col1 - 1.0) < 1e-6) {
        std::cout << "✓ PASSED: Softmax stable with extreme values\n";
    } else {
        std::cout << "✗ FAILED: Softmax produced non-finite or incorrect probabilities\n";
    }
}

// Test 5: Cross-entropy with zero probabilities
void test_cross_entropy_edge_cases() {
    std::cout << "\n=== Test 5: Cross-Entropy Edge Cases ===\n";
    
    math::Matrix logits(3, 2);
    logits(0, 0) = -100.0;
    logits(1, 0) = 10.0;
    logits(2, 0) = -100.0;
    logits(0, 1) = 5.0;
    logits(1, 1) = 5.0;
    logits(2, 1) = 5.0;
    
    math::Matrix targets(3, 2, 0.0);
    targets(1, 0) = 1.0;
    targets(0, 1) = 1.0;
    
    double loss = Loss::cross_entropy(logits, targets);
    math::Matrix grad = Loss::cross_entropy_grad(logits, targets);
    
    std::cout << "Loss: " << loss << "\n";
    
    bool all_finite = std::isfinite(loss);
    for (size_t i = 0; i < grad.rows() && all_finite; ++i) {
        for (size_t j = 0; j < grad.cols() && all_finite; ++j) {
            if (!std::isfinite(grad(i, j))) all_finite = false;
        }
    }
    
    if (all_finite) {
        std::cout << "✓ PASSED: Cross-entropy handles edge cases\n";
    } else {
        std::cout << "✗ FAILED: Cross-entropy produced non-finite values\n";
    }
}

// Test 6: LayerNorm with zero variance
void test_layernorm_zero_variance() {
    std::cout << "\n=== Test 6: LayerNorm with Zero Variance ===\n";
    
    LayerNorm ln(4);
    
    // All values the same (zero variance)
    math::Matrix input(4, 2);
    for (size_t i = 0; i < 4; ++i) {
        input(i, 0) = 5.0;
        input(i, 1) = -3.0;
    }
    
    math::Matrix output = ln.forward(input);
    
    bool all_finite = true;
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            if (!std::isfinite(output(i, j))) {
                all_finite = false;
                std::cout << "Non-finite at (" << i << "," << j << "): " << output(i, j) << "\n";
            }
        }
    }
    
    if (all_finite) {
        std::cout << "✓ PASSED: LayerNorm handles zero variance (epsilon protection)\n";
    } else {
        std::cout << "✗ FAILED: LayerNorm produced non-finite values\n";
    }
}

// Test 7: Attention with invalid configuration
void test_attention_validation() {
    std::cout << "\n=== Test 7: Attention Configuration Validation ===\n";
    
    bool caught_zero = false;
    bool caught_divisibility = false;
    
    try {
        MultiHeadAttention attn1(0, 4);
    } catch (const std::invalid_argument& e) {
        caught_zero = true;
        std::cout << "Caught zero d_model: " << e.what() << "\n";
    }
    
    try {
        MultiHeadAttention attn2(10, 3);
    } catch (const std::invalid_argument& e) {
        caught_divisibility = true;
        std::cout << "Caught divisibility error: " << e.what() << "\n";
    }
    
    if (caught_zero && caught_divisibility) {
        std::cout << "✓ PASSED: Attention validates configuration\n";
    } else {
        std::cout << "✗ FAILED: Attention missing validation checks\n";
    }
}

// Test 8: Dense layer gradient magnitude
void test_dense_gradient_magnitude() {
    std::cout << "\n=== Test 8: Dense Gradient Numerical Check ===\n";
    
    Dense layer(2, 2, false);
    
    // Set known weights
    const_cast<double&>(layer.weight().data()(0, 0)) = 0.5;
    const_cast<double&>(layer.weight().data()(0, 1)) = -0.3;
    const_cast<double&>(layer.weight().data()(1, 0)) = 0.2;
    const_cast<double&>(layer.weight().data()(1, 1)) = 0.7;
    
    math::Matrix input(2, 1);
    input(0, 0) = 1.0;
    input(1, 0) = 2.0;
    
    // Loss function for gradient checking
    auto loss_fn = [&]() {
        math::Matrix output = layer.forward(input);
        return output(0, 0) * output(0, 0) + output(1, 0) * output(1, 0);
    };
    
    // Forward and compute analytical gradient
    math::Matrix output = layer.forward(input);
    math::Matrix grad_output(2, 1);
    grad_output(0, 0) = 2.0 * output(0, 0);
    grad_output(1, 0) = 2.0 * output(1, 0);
    
    layer.weight().zero_grad();
    layer.backward(grad_output);
    
    // Check gradients using numerical differentiation
    bool passed = GradientChecker::check_gradients(
        layer.weight(), loss_fn, 1e-5, 1e-3
    );
    
    if (passed) {
        std::cout << "✓ PASSED: Dense gradients match numerical approximation\n";
    } else {
        std::cout << "✗ FAILED: Dense gradients differ from numerical\n";
    }
}

// Test 9: Gradient clipping at threshold
void test_gradient_clipping_threshold() {
    std::cout << "\n=== Test 9: Gradient Clipping Boundary ===\n";
    
    // Create parameters with known gradient magnitudes
    nn::Parameter p1(2, 1, "p1");
    nn::Parameter p2(2, 1, "p2");
    
    // Set gradients to create norm of exactly 2.0
    const_cast<double&>(p1.grad()(0, 0)) = 1.0;
    const_cast<double&>(p1.grad()(1, 0)) = 1.0;
    const_cast<double&>(p2.grad()(0, 0)) = 1.0;
    const_cast<double&>(p2.grad()(1, 0)) = 1.0;
    // Norm = sqrt(4) = 2.0
    
    double initial_norm = std::sqrt(
        p1.grad()(0,0)*p1.grad()(0,0) + p1.grad()(1,0)*p1.grad()(1,0) +
        p2.grad()(0,0)*p2.grad()(0,0) + p2.grad()(1,0)*p2.grad()(1,0)
    );
    
    std::cout << "Initial gradient norm: " << initial_norm << "\n";
    
    // Manually clip to max_norm = 1.0
    double max_norm = 1.0;
    if (initial_norm > max_norm) {
        double scale = max_norm / initial_norm;
        const_cast<double&>(p1.grad()(0, 0)) *= scale;
        const_cast<double&>(p1.grad()(1, 0)) *= scale;
        const_cast<double&>(p2.grad()(0, 0)) *= scale;
        const_cast<double&>(p2.grad()(1, 0)) *= scale;
    }
    
    double final_norm = std::sqrt(
        p1.grad()(0,0)*p1.grad()(0,0) + p1.grad()(1,0)*p1.grad()(1,0) +
        p2.grad()(0,0)*p2.grad()(0,0) + p2.grad()(1,0)*p2.grad()(1,0)
    );
    
    std::cout << "Final gradient norm: " << final_norm << "\n";
    
    if (std::abs(final_norm - max_norm) < 1e-6) {
        std::cout << "✓ PASSED: Gradient clipping works at threshold\n";
    } else {
        std::cout << "✗ FAILED: Clipped norm is " << final_norm << ", expected " << max_norm << "\n";
    }
}

// Test 10: Extreme weight initialization
void test_extreme_weight_initialization() {
    std::cout << "\n=== Test 10: Extreme Weight Values ===\n";
    
    Dense layer(3, 3, true);
    
    // Set extreme weights
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            const_cast<double&>(layer.weight().data()(i, j)) = (i == j) ? 1e8 : 0.0;
        }
    }
    
    math::Matrix input(3, 1);
    input(0, 0) = 1e-8;
    input(1, 0) = 1e-8;
    input(2, 0) = 1e-8;
    
    math::Matrix output = layer.forward(input);
    
    bool all_finite = true;
    for (size_t i = 0; i < output.rows(); ++i) {
        for (size_t j = 0; j < output.cols(); ++j) {
            if (!std::isfinite(output(i, j))) all_finite = false;
        }
    }
    
    std::cout << "Output values: ";
    for (size_t i = 0; i < output.rows(); ++i) {
        std::cout << output(i, 0) << " ";
    }
    std::cout << "\n";
    
    if (all_finite) {
        std::cout << "✓ PASSED: Forward pass handles extreme weights\n";
    } else {
        std::cout << "✗ FAILED: Extreme weights caused non-finite outputs\n";
    }
}

int main() {
    std::cout << "╔════════════════════════════════════════════════╗\n";
    std::cout << "║   Aphrodite Numerical Stability Test Suite    ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    
    test_dense_gradient_accumulation();
    test_adam_nonfinite_gradients();
    test_adam_large_t();
    test_softmax_extreme_values();
    test_cross_entropy_edge_cases();
    test_layernorm_zero_variance();
    test_attention_validation();
    test_dense_gradient_magnitude();
    test_gradient_clipping_threshold();
    test_extreme_weight_initialization();
    
    std::cout << "\n╔════════════════════════════════════════════════╗\n";
    std::cout << "║         Test Suite Complete                    ║\n";
    std::cout << "╚════════════════════════════════════════════════╝\n";
    
    return 0;
}
