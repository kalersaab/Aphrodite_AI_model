#include "Trainer.h"
#include "Loss.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace kaler::nn {
namespace {
constexpr char kMagic[8] = {'A','P','H','R','G','P','T','1'};
constexpr std::uint32_t kVersion = 1;

template <typename T>
void writeValue(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("checkpoint write failed");
}
template <typename T>
T readValue(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("checkpoint is truncated or unreadable");
    return value;
}
}

Trainer::Trainer(GPT& model, Adam& optimizer, const data::CharTokenizer& tokenizer,
                 const std::vector<int>& tokens, TrainConfig config)
    : model_(model), optimizer_(optimizer), tokenizer_(tokenizer), tokens_(tokens),
      config_(std::move(config)), rng_(config_.seed) {
    if (config_.sequence_length < 2) throw std::invalid_argument("sequence_length must be >= 2");
    if (tokens_.size() < config_.sequence_length + 10) throw std::invalid_argument("Not enough tokens to train/validate");
    const std::size_t split = static_cast<std::size_t>(tokens_.size() * 0.9);
    train_tokens_.assign(tokens_.begin(), tokens_.begin() + split);
    val_tokens_.assign(tokens_.begin() + split, tokens_.end());
    if (train_tokens_.size() <= config_.sequence_length + 1 || val_tokens_.size() <= config_.sequence_length + 1)
        throw std::invalid_argument("Train/validation split is too small for sequence_length");
}

double Trainer::scheduled_learning_rate(std::size_t step) const {
    if (config_.warmup_steps > 0 && step < config_.warmup_steps) {
        return config_.max_learning_rate * static_cast<double>(step + 1) / config_.warmup_steps;
    }
    const std::size_t decay_steps = std::max<std::size_t>(1, config_.steps > config_.warmup_steps ? config_.steps - config_.warmup_steps : 1);
    const double progress = std::min(1.0, static_cast<double>(step > config_.warmup_steps ? step - config_.warmup_steps : 0) / decay_steps);
    const double cosine = 0.5 * (1.0 + std::cos(3.14159265358979323846 * progress));
    return config_.min_learning_rate + cosine * (config_.max_learning_rate - config_.min_learning_rate);
}

double Trainer::clip_gradients() {
    double sum_sq = 0.0;
    auto params = model_.parameters();
    
    // First pass: compute norm and check for non-finite values
    bool has_nonfinite = false;
    for (const auto* p : params) {
        const auto& g = p->grad();
        for (std::size_t r = 0; r < g.rows(); ++r) {
            for (std::size_t c = 0; c < g.cols(); ++c) {
                double val = g(r, c);
                if (!std::isfinite(val)) {
                    has_nonfinite = true;
                    break;
                }
                sum_sq += val * val;
            }
            if (has_nonfinite) break;
        }
        if (has_nonfinite) break;
    }
    
    // Handle non-finite gradients by zeroing them
    if (has_nonfinite || !std::isfinite(sum_sq)) {
        std::cerr << "Warning: Non-finite gradients detected, zeroing all gradients\n";
        for (auto* p : params) {
            p->zero_grad();
        }
        return 0.0;
    }
    
    const double norm = std::sqrt(sum_sq);
    
    // Only clip if norm is substantial and exceeds threshold
    if (norm > config_.max_gradient_norm && norm > 1e-6) {
        const double scale = config_.max_gradient_norm / norm;
        for (auto* p : params) {
            auto& g = p->grad();
            for (std::size_t r = 0; r < g.rows(); ++r)
                for (std::size_t c = 0; c < g.cols(); ++c) g(r,c) *= scale;
        }
    }
    
    return norm;
}

double Trainer::train_step() {
    std::uniform_int_distribution<std::size_t> pick(0, train_tokens_.size() - config_.sequence_length - 2);
    const std::size_t start = pick(rng_);
    std::vector<int> input(train_tokens_.begin() + start,
                           train_tokens_.begin() + start + config_.sequence_length);
    const std::size_t vocab = tokenizer_.vocab_size();
    math::Matrix targets(vocab, config_.sequence_length, 0.0);
    for (std::size_t t = 0; t < config_.sequence_length; ++t)
        targets(static_cast<std::size_t>(train_tokens_[start + t + 1]), t) = 1.0;

    optimizer_.zero_grad();
    const math::Matrix logits = model_.forward(input);
    const double loss = Loss::cross_entropy(logits, targets);
    const math::Matrix grad = Loss::cross_entropy_grad(logits, targets);
    model_.backward(grad, input);
    last_gradient_norm_ = clip_gradients();
    optimizer_.step();
    return loss;
}

double Trainer::evaluate(const std::vector<int>& split) const {
    const std::size_t seq = config_.sequence_length;
    const std::size_t possible = split.size() - seq - 1;
    const std::size_t count = std::min(config_.eval_batches, possible);
    if (count == 0) return std::numeric_limits<double>::infinity();
    double total = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t start = count == 1 ? 0 : (i * possible) / count;
        std::vector<int> input(split.begin() + start, split.begin() + start + seq);
        math::Matrix targets(tokenizer_.vocab_size(), seq, 0.0);
        for (std::size_t t = 0; t < seq; ++t) targets(static_cast<std::size_t>(split[start + t + 1]), t) = 1.0;
        const math::Matrix logits = model_.forward(input);
        total += Loss::cross_entropy(logits, targets);
    }
    return total / count;
}

void Trainer::print_sample(std::size_t step) {
    const std::string prompt = "First Citizen:";
    const auto prompt_ids = tokenizer_.encode(prompt);
    const auto generated = model_.generate(prompt_ids, 120);
    std::cout << "\n--- sample @ step " << step << " ---\n";
    std::cout << tokenizer_.decode(generated) << "\n-----------------------------\n";
}

void Trainer::train() {
    std::cout << "Aphrodite training | tokens=" << tokens_.size()
              << " vocab=" << tokenizer_.vocab_size()
              << " train=" << train_tokens_.size() << " val=" << val_tokens_.size() << "\n";
    std::cout << "step,train_loss,val_loss,perplexity,lr,grad_norm,step_ms\n";
    double best_val = std::numeric_limits<double>::infinity();
    std::size_t best_step = 0, stale_evals = 0;
    for (std::size_t step = 0; step < config_.steps; ++step) {
        optimizer_.set_learning_rate(scheduled_learning_rate(step));
        const auto begin = std::chrono::steady_clock::now();
        const double loss = train_step();
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - begin).count();
        if ((step + 1) % config_.eval_interval == 0 || step == 0) {
            const double val = evaluate(val_tokens_);
            const double lr = optimizer_.learning_rate();
            std::cout << (step + 1) << ',' << std::setprecision(6) << loss << ',' << val << ','
                      << std::exp(std::min(val, 20.0)) << ',' << lr << ',' << last_gradient_norm_ << ',' << ms << '\n';
            if (val < best_val) {
                best_val = val;
                best_step = step + 1;
                stale_evals = 0;
                if (!save_checkpoint(config_.checkpoint_path, model_,
                    model_.config(), best_step, best_val)) {
                    std::cerr << "Warning: could not save checkpoint at " << config_.checkpoint_path << '\n';
                }
            } else {
                ++stale_evals;
            }
            print_sample(step + 1);
            if (config_.early_stopping_patience > 0 && stale_evals >= config_.early_stopping_patience) {
                std::cout << "Early stopping: validation loss did not improve for " << stale_evals << " evaluations.\n";
                break;
            }
        }
    }
    std::cout << "Training finished. Best validation loss=" << best_val << " at step " << best_step << '\n';
}

bool Trainer::save_checkpoint(const std::string& path, GPT& model,
                              const GPTConfig& cfg, std::size_t step, double val_loss) {
    try {
        const std::filesystem::path dest(path);
        if (dest.has_parent_path()) std::filesystem::create_directories(dest.parent_path());
        const std::string tmp = path + ".tmp";
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(kMagic, sizeof(kMagic)); writeValue(out, kVersion);
        const std::uint64_t dims[] = {cfg.vocab_size, cfg.max_seq_len, cfg.d_model, cfg.num_heads, cfg.num_layers, cfg.d_ff, step};
        for (auto d : dims) writeValue(out, d);
        writeValue(out, val_loss);
        const auto params = model.parameters();
        writeValue(out, static_cast<std::uint64_t>(params.size()));
        for (const auto* p : params) {
            writeValue(out, static_cast<std::uint64_t>(p->data().rows()));
            writeValue(out, static_cast<std::uint64_t>(p->data().cols()));
            for (std::size_t r = 0; r < p->data().rows(); ++r)
                for (std::size_t c = 0; c < p->data().cols(); ++c) writeValue(out, p->data()(r,c));
        }
        out.close();
        if (!out) return false;
        std::filesystem::rename(tmp, path);
        return true;
    } catch (...) { return false; }
}

bool Trainer::load_checkpoint(const std::string& path, GPT& model,
                              const GPTConfig& expected, std::size_t* step, double* val_loss) {
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[sizeof(kMagic)]{}; in.read(magic, sizeof(magic));
        if (!in || !std::equal(std::begin(magic), std::end(magic), std::begin(kMagic))) return false;
        if (readValue<std::uint32_t>(in) != kVersion) return false;
        const std::uint64_t actual[] = {readValue<std::uint64_t>(in), readValue<std::uint64_t>(in), readValue<std::uint64_t>(in), readValue<std::uint64_t>(in), readValue<std::uint64_t>(in), readValue<std::uint64_t>(in)};
        const std::uint64_t wanted[] = {expected.vocab_size, expected.max_seq_len, expected.d_model, expected.num_heads, expected.num_layers, expected.d_ff};
        for (int i=0;i<6;++i) if (actual[i] != wanted[i]) return false;
        const auto saved_step = readValue<std::uint64_t>(in);
        const double saved_val = readValue<double>(in);
        const auto params = model.parameters();
        if (readValue<std::uint64_t>(in) != params.size()) return false;
        for (auto* p : params) {
            if (readValue<std::uint64_t>(in) != p->data().rows() || readValue<std::uint64_t>(in) != p->data().cols()) return false;
            for (std::size_t r=0;r<p->data().rows();++r)
                for (std::size_t c=0;c<p->data().cols();++c) p->data()(r,c) = readValue<double>(in);
        }
        if (step) *step = saved_step;
        if (val_loss) *val_loss = saved_val;
        return true;
    } catch (...) { return false; }
}

} // namespace kaler::nn
