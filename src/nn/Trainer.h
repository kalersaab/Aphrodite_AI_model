#ifndef APHRODITE_TRAINER_H
#define APHRODITE_TRAINER_H

#include "GPT.h"
#include "Optimizer.h"
#include "../data/DataLoader.h"
#include <cstddef>
#include <string>
#include <random>
#include <cstdint>
#include <vector>

namespace kaler::nn {

struct TrainConfig {
    std::size_t steps = 1000;
    std::size_t sequence_length = 32;
    std::size_t eval_interval = 100;
    std::size_t eval_batches = 20;
    std::size_t warmup_steps = 50;
    double max_learning_rate = 0.001;
    double min_learning_rate = 0.0001;
    double max_gradient_norm = 1.0;
    std::size_t early_stopping_patience = 8;
    std::uint32_t seed = 1337;
    std::string checkpoint_path = "checkpoints/aphrodite_best.bin";
};

class Trainer {
public:
    Trainer(GPT& model, Adam& optimizer, const data::CharTokenizer& tokenizer,
            const std::vector<int>& tokens, TrainConfig config = {});
    void train();

    static bool save_checkpoint(const std::string& path, GPT& model,
                                const GPTConfig& model_config, std::size_t step,
                                double validation_loss);
    static bool load_checkpoint(const std::string& path, GPT& model,
                                const GPTConfig& expected_config, std::size_t* step = nullptr,
                                double* validation_loss = nullptr);

private:
    GPT& model_;
    Adam& optimizer_;
    const data::CharTokenizer& tokenizer_;
    const std::vector<int>& tokens_;
    TrainConfig config_;
    std::vector<int> train_tokens_;
    std::vector<int> val_tokens_;
    std::mt19937 rng_;
    double last_gradient_norm_ = 0.0;

    double train_step();
    double evaluate(const std::vector<int>& split) const;
    double scheduled_learning_rate(std::size_t step) const;
    double clip_gradients();
    void print_sample(std::size_t step);
};

} // namespace kaler::nn
#endif
