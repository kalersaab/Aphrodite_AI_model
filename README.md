# Aphrodite — Phase 1

A dependency-free C++20 numerical foundation for a GPT-style language model built from scratch.

## Requirements
- macOS with Xcode Command Line Tools (`xcode-select --install`)
- CMake 3.20+
- A C++20 compiler (Apple Clang is fine)

No pretrained weights, ML framework, or third-party math library is used in Phase 1.

## Build and run
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/aphrodite
```

## Phase 1 scope
- Row-major matrix storage
- Bounds-checked element access
- Matrix addition/subtraction/scaling
- Matrix multiplication
- Transpose
- Row-wise softmax with numerical stabilization
- ReLU and tanh activations
- Unit tests for known answers and basic numerical properties

This is the numerical foundation, not yet a language model. Phase 2 adds trainable parameters, gradients, loss, and an optimizer. Later phases add tokenization, causal attention, Transformer blocks, training, and generation.

## Suggested next milestone
Implement finite-difference gradient checking for a small linear layer and cross-entropy loss before building a full Transformer.
