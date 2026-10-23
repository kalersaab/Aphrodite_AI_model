#include "nn/Dense.h"
#include "nn/Loss.h"
#include "nn/Optimizer.h"
#include "nn/GradientChecker.h"
#include <iostream>
#include <iomanip>

using namespace kaler;

void test_gradient_checking() {
    std::cout << "=== Gradient Checking Test ===\n\n";
    
    nn::Dense layer(3, 2);
    
    math::Matrix input(3, 1, 0.0);
    const_cast<double&>(input(0, 0)) = 1.0;
    const_cast<double&>(input(1, 0)) = 0.5;
    const_cast<double&>(input(2, 0)) = -0.5;
    
    math::Matrix target(2, 1, 0.0);
    const_cast<double&>(target(0, 0)) = 1.0;
    const_cast<double&>(target(1, 0)) = 0.0;
    
    auto loss_fn = [&]() -> double {
        math::Matrix output = layer.forward(input);
        return nn::Loss::cross_entropy(output, target);
    };
    
    layer.weight().zero_grad();
    layer.bias().zero_grad();
    
    math::Matrix output = layer.forward(input);
    double loss = nn::Loss::cross_entropy(output, target);
    math::Matrix grad_output = nn::Loss::cross_entropy_grad(output, target);
    layer.backward(grad_output);
    
    std::cout << "Forward pass loss: " << loss << "\n\n";
    
    bool weight_check = nn::GradientChecker::check_gradients(layer.weight(), loss_fn);
    std::cout << "\n";
    bool bias_check = nn::GradientChecker::check_gradients(layer.bias(), loss_fn);
    
    if (weight_check && bias_check) {
        std::cout << "\n✅ Gradient checking PASSED!\n\n";
    } else {
        std::cout << "\n❌ Gradient checking FAILED!\n\n";
    }
}

void test_synthetic_learning() {
    std::cout << "=== Synthetic Dataset Learning Test ===\n\n";
    
    std::cout << "Task: Learn XOR-like pattern\n";
    std::cout << "Input: 2D points, Output: 3 classes\n\n";
    
    std::vector<math::Matrix> inputs;
    std::vector<math::Matrix> targets;
    
    math::Matrix x1(2, 1, 0.0); const_cast<double&>(x1(0,0)) = 0.1; const_cast<double&>(x1(1,0)) = 0.1;
    math::Matrix t1(3, 1, 0.0); const_cast<double&>(t1(0,0)) = 1.0;
    inputs.push_back(x1); targets.push_back(t1);
    
    math::Matrix x2(2, 1, 0.0); const_cast<double&>(x2(0,0)) = 0.9; const_cast<double&>(x2(1,0)) = 0.1;
    math::Matrix t2(3, 1, 0.0); const_cast<double&>(t2(1,0)) = 1.0;
    inputs.push_back(x2); targets.push_back(t2);
    
    math::Matrix x3(2, 1, 0.0); const_cast<double&>(x3(0,0)) = 0.1; const_cast<double&>(x3(1,0)) = 0.9;
    math::Matrix t3(3, 1, 0.0); const_cast<double&>(t3(1,0)) = 1.0;
    inputs.push_back(x3); targets.push_back(t3);
    
    math::Matrix x4(2, 1, 0.0); const_cast<double&>(x4(0,0)) = 0.9; const_cast<double&>(x4(1,0)) = 0.9;
    math::Matrix t4(3, 1, 0.0); const_cast<double&>(t4(2,0)) = 1.0;
    inputs.push_back(x4); targets.push_back(t4);
    
    nn::Dense hidden(2, 8);
    nn::Dense output_layer(8, 3);
    
    nn::SGD optimizer(0.1);
    optimizer.add_parameter(&hidden.weight());
    optimizer.add_parameter(&hidden.bias());
    optimizer.add_parameter(&output_layer.weight());
    optimizer.add_parameter(&output_layer.bias());
    
    std::cout << "Training with SGD (lr=0.1)...\n\n";
    std::cout << "Epoch | Loss     | Accuracy\n";
    std::cout << "------|----------|----------\n";
    
    for (int epoch = 0; epoch < 200; ++epoch) {
        double total_loss = 0.0;
        int correct = 0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            optimizer.zero_grad();
            
            math::Matrix h = hidden.forward(inputs[i]);
            
            math::Matrix h_relu(h.rows(), h.cols(), 0.0);
            for (size_t r = 0; r < h.rows(); ++r) {
                for (size_t c = 0; c < h.cols(); ++c) {
                    const_cast<double&>(h_relu(r, c)) = std::max(0.0, h(r, c));
                }
            }
            
            math::Matrix output = output_layer.forward(h_relu);
            
            double loss = nn::Loss::cross_entropy(output, targets[i]);
            total_loss += loss;
            
            size_t pred_class = 0;
            double max_prob = output(0, 0);
            for (size_t j = 1; j < 3; ++j) {
                if (output(j, 0) > max_prob) {
                    max_prob = output(j, 0);
                    pred_class = j;
                }
            }
            
            size_t true_class = 0;
            for (size_t j = 0; j < 3; ++j) {
                if (targets[i](j, 0) > 0.5) {
                    true_class = j;
                    break;
                }
            }
            
            if (pred_class == true_class) correct++;
            
            math::Matrix grad_output = nn::Loss::cross_entropy_grad(output, targets[i]);
            math::Matrix grad_h = output_layer.backward(grad_output);
            
            math::Matrix grad_h_relu(grad_h.rows(), grad_h.cols(), 0.0);
            for (size_t r = 0; r < grad_h.rows(); ++r) {
                for (size_t c = 0; c < grad_h.cols(); ++c) {
                    if (h(r, c) > 0.0) {
                        const_cast<double&>(grad_h_relu(r, c)) = grad_h(r, c);
                    }
                }
            }
            
            hidden.backward(grad_h_relu);
            
            optimizer.step();
        }
        
        double avg_loss = total_loss / inputs.size();
        double accuracy = (double)correct / inputs.size();
        
        if (epoch % 20 == 0 || epoch == 199) {
            std::cout << std::setw(5) << epoch << " | "
                      << std::fixed << std::setprecision(6) << avg_loss << " | "
                      << std::fixed << std::setprecision(2) << (accuracy * 100.0) << "%\n";
        }
    }
    
    std::cout << "\n✅ Training completed!\n\n";
}

void test_adam_optimizer() {
    std::cout << "=== Adam Optimizer Test ===\n\n";
    
    std::vector<math::Matrix> inputs;
    std::vector<math::Matrix> targets;
    
    for (int i = 0; i < 8; ++i) {
        math::Matrix x(4, 1, 0.0);
        const_cast<double&>(x(0,0)) = (i & 1) ? 1.0 : 0.0;
        const_cast<double&>(x(1,0)) = (i & 2) ? 1.0 : 0.0;
        const_cast<double&>(x(2,0)) = (i & 4) ? 1.0 : 0.0;
        const_cast<double&>(x(3,0)) = 1.0;
        
        math::Matrix t(2, 1, 0.0);
        int label = ((i & 1) ^ (i & 2) >> 1) & 1;
        const_cast<double&>(t(0,0)) = label ? 1.0 : 0.0;
        const_cast<double&>(t(1,0)) = label ? 0.0 : 1.0;
        
        inputs.push_back(x);
        targets.push_back(t);
    }
    
    nn::Dense layer(4, 2);
    
    nn::Adam optimizer(0.01);
    optimizer.add_parameter(&layer.weight());
    optimizer.add_parameter(&layer.bias());
    
    std::cout << "Training with Adam (lr=0.01)...\n\n";
    std::cout << "Epoch | Loss\n";
    std::cout << "------|----------\n";
    
    for (int epoch = 0; epoch < 100; ++epoch) {
        double total_loss = 0.0;
        
        for (size_t i = 0; i < inputs.size(); ++i) {
            optimizer.zero_grad();
            
            math::Matrix output = layer.forward(inputs[i]);
            double loss = nn::Loss::cross_entropy(output, targets[i]);
            total_loss += loss;
            
            math::Matrix grad = nn::Loss::cross_entropy_grad(output, targets[i]);
            layer.backward(grad);
            
            optimizer.step();
        }
        
        double avg_loss = total_loss / inputs.size();
        
        if (epoch % 10 == 0 || epoch == 99) {
            std::cout << std::setw(5) << epoch << " | "
                      << std::fixed << std::setprecision(6) << avg_loss << "\n";
        }
    }
    
    std::cout << "\n✅ Adam optimization completed!\n\n";
}

int main() {
    std::cout << "╔════════════════════════════════════════╗\n";
    std::cout << "║   Neural Network Training Tests        ║\n";
    std::cout << "╚════════════════════════════════════════╝\n\n";
    
    try {
        test_gradient_checking();
        test_synthetic_learning();
        test_adam_optimizer();
        
        std::cout << "╔════════════════════════════════════════╗\n";
        std::cout << "║   ✅ ALL TESTS PASSED!                 ║\n";
        std::cout << "╚════════════════════════════════════════╝\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << "\n";
        return 1;
    }
}
