#include "nn/GPT.h"
#include "nn/Trainer.h"
#include <cmath>
#include <filesystem>
#include <iostream>

int main() {
    using namespace kaler::nn;
    const GPTConfig config(12, 16, 8, 2, 1, 16);
    GPT model(config);
    const auto path = std::filesystem::temp_directory_path() / "aphrodite_checkpoint_test.bin";
    auto params = model.parameters();
    const double original = params.front()->data()(0, 0);
    if (!Trainer::save_checkpoint(path.string(), model, config, 123, 1.25)) {
        std::cerr << "checkpoint save failed\n"; return 1;
    }
    params.front()->data()(0, 0) = original + 123.0;
    std::size_t step = 0;
    double val = 0.0;
    if (!Trainer::load_checkpoint(path.string(), model, config, &step, &val)) {
        std::cerr << "checkpoint load failed\n"; return 1;
    }
    std::filesystem::remove(path);
    if (std::abs(params.front()->data()(0, 0) - original) > 1e-12 || step != 123 || std::abs(val - 1.25) > 1e-12) {
        std::cerr << "checkpoint round trip mismatch\n"; return 1;
    }
    std::cout << "Checkpoint round-trip passed\n";
    return 0;
}
