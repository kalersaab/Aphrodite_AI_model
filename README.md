# Aphrodite — Phase 2 Training Pipeline

A small educational GPT-style character language model implemented in C++20 without pretrained weights or an ML framework.

## Requirements
- macOS + Xcode Command Line Tools (`xcode-select --install`)
- CMake 3.20+

## Build and test
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Train on Tiny Shakespeare
From the repository root:
```bash
./build/aphrodite_train data/raw/tiny_shakespeare.txt 1000
```
Arguments are dataset path and training steps. The trainer uses a contiguous 90/10 train/validation split, character-level next-token targets, Adam, warmup + cosine learning-rate decay, global gradient clipping, periodic validation, early stopping, checkpoint saving, and sample generation.

Metrics are printed as CSV-like lines. Best weights are saved to `checkpoints/aphrodite_best.bin`.

## Current limitations (intentional and important)
- Checkpoints currently store model weights + configuration + step + validation loss; Adam moment state is not yet persisted, so this is best-model saving, not exact optimizer-state resume.
- The current attention implementation uses a causal scaled dot-product attention calculation but does not yet split projections into independent attention heads despite the `num_heads` configuration. Correct multi-head splitting and gradient tests are a follow-up correctness milestone.
- Training uses one sequence per update; this is a small educational baseline, not a high-throughput trainer.
- Tiny Shakespeare is too small for broad language understanding. The target is to verify decreasing loss and recognizable character-level patterns.

## Checkpoint format
Versioned binary format with a magic header, architecture dimensions, step, validation loss, parameter shapes, and float64 parameter values. Load validates the architecture before loading.
