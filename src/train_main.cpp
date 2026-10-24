#include "data/DataLoader.h"
#include "nn/GPT.h"
#include "nn/Optimizer.h"
#include "nn/Trainer.h"
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::string dataset = "data/raw/tiny_shakespeare.txt";
    std::size_t steps = 1000;
    if (argc > 1) dataset = argv[1];
    if (argc > 2) steps = static_cast<std::size_t>(std::stoul(argv[2]));

    try {
        kaler::data::TextDataLoader loader(dataset, 4, 32);
        if (!loader.load()) return 1;
        const auto& tokenizer = loader.tokenizer();
        const auto& tokens = loader.tokens();

        kaler::nn::GPTConfig model_config(tokenizer.vocab_size(), 64, 32, 2, 1, 64);
        kaler::nn::GPT model(model_config);
        kaler::nn::Adam optimizer(0.001);
        for (auto* parameter : model.parameters()) optimizer.add_parameter(parameter);

        kaler::nn::TrainConfig train_config;
        train_config.steps = steps;
        train_config.sequence_length = 32;
        train_config.eval_interval = 100;
        train_config.eval_batches = 10;
        train_config.checkpoint_path = "checkpoints/aphrodite_best.bin";
        kaler::nn::Trainer trainer(model, optimizer, tokenizer, tokens, train_config);
        trainer.train();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Training failed: " << e.what() << '\n';
        return 1;
    }
}
