#include "DataLoader.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace kaler {
namespace data {

namespace {
std::vector<std::string> utf8Characters(const std::string& text) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < text.size();) {
        const unsigned char lead = static_cast<unsigned char>(text[i]);
        std::size_t len = 1;
        if ((lead & 0xE0) == 0xC0) len = 2;
        else if ((lead & 0xF0) == 0xE0) len = 3;
        else if ((lead & 0xF8) == 0xF0) len = 4;
        if (i + len > text.size()) len = 1;
        bool valid = len > 1;
        for (std::size_t j = 1; j < len && valid; ++j) {
            const unsigned char continuation = static_cast<unsigned char>(text[i + j]);
            if ((continuation & 0xC0) != 0x80) valid = false;
        }
        if (!valid) len = 1;
        out.emplace_back(text.substr(i, len));
        i += len;
    }
    return out;
}
}

void CharTokenizer::fit(const std::string& text) {
    char_to_idx_.clear();
    idx_to_char_.clear();
    // Reserve 0 for padding and 1 for unknown; UTF-8 characters start at 2.
    int next_idx = 2;
    for (const auto& token : utf8Characters(text)) {
        if (char_to_idx_.find(token) == char_to_idx_.end()) {
            char_to_idx_[token] = next_idx;
            idx_to_char_[next_idx] = token;
            ++next_idx;
        }
    }
    std::cout << "Vocabulary built: " << vocab_size() << " tokens (including PAD/UNK)\n";
}

std::vector<int> CharTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    const auto characters = utf8Characters(text);
    tokens.reserve(characters.size());
    for (const auto& token : characters) {
        auto it = char_to_idx_.find(token);
        tokens.push_back(it != char_to_idx_.end() ? it->second : unk_token());
    }
    return tokens;
}

std::string CharTokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    for (int token : tokens) {
        if (token == pad_token()) continue;
        auto it = idx_to_char_.find(token);
        if (it != idx_to_char_.end()) text += it->second;
        else text += '?';
    }
    return text;
}

void CharTokenizer::add_char(const std::string& token) {
    if (char_to_idx_.find(token) == char_to_idx_.end()) {
        const int next_idx = static_cast<int>(char_to_idx_.size()) + 2;
        char_to_idx_[token] = next_idx;
        idx_to_char_[next_idx] = token;
    }
}

TextDataLoader::TextDataLoader(const std::string& filepath,
                               size_t batch_size,
                               size_t seq_length)
    : filepath_(filepath)
    , batch_size_(batch_size)
    , seq_length_(seq_length)
    , current_pos_(0) {
}

std::string TextDataLoader::load_file() {
    std::ifstream file(filepath_);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filepath_);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool TextDataLoader::load() {
    try {
        std::string text = load_file();
        std::cout << "Loaded " << text.size() << " characters from " << filepath_ << "\n";
        
        if (batch_size_ == 0 || seq_length_ == 0) {
            throw std::runtime_error("batch_size and seq_length must be positive");
        }
        if (text.size() < seq_length_ + 1) {
            throw std::runtime_error("Dataset must contain at least seq_length + 1 characters");
        }
        tokenizer_.fit(text);
        tokens_ = tokenizer_.encode(text);
        
        std::cout << "Tokenized into " << tokens_.size() << " tokens\n";
        std::cout << "Number of batches: " << num_batches() << "\n";
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading data: " << e.what() << "\n";
        return false;
    }
}

TextDataLoader::Batch TextDataLoader::next_batch() {
    Batch batch;
    batch.input.reserve(batch_size_);
    batch.target.reserve(batch_size_);
    
    for (size_t i = 0; i < batch_size_; ++i) {
        if (current_pos_ + seq_length_ + 1 > tokens_.size()) {
            current_pos_ = 0;
        }
        
        std::vector<int> input_seq(tokens_.begin() + current_pos_,
                                   tokens_.begin() + current_pos_ + seq_length_);
        std::vector<int> target_seq(tokens_.begin() + current_pos_ + 1,
                                    tokens_.begin() + current_pos_ + seq_length_ + 1);
        
        batch.input.push_back(input_seq);
        batch.target.push_back(target_seq);
        
        current_pos_ += seq_length_;
    }
    
    return batch;
}

void TextDataLoader::reset() {
    current_pos_ = 0;
}

bool TextDataLoader::has_next() const {
    return current_pos_ + seq_length_ + 1 <= tokens_.size();
}

size_t TextDataLoader::num_batches() const {
    if (tokens_.empty()) return 0;
    return (tokens_.size() - 1) / (seq_length_ * batch_size_);
}

void DatasetSplitter::split_file(const std::string& input_path,
                                const std::string& output_dir,
                                float train_ratio,
                                float val_ratio) {
    std::ifstream file(input_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + input_path);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string text = buffer.str();
    
    size_t total_size = text.size();
    size_t train_size = static_cast<size_t>(total_size * train_ratio);
    size_t val_size = static_cast<size_t>(total_size * val_ratio);
    
    std::string train_data = text.substr(0, train_size);
    std::string val_data = text.substr(train_size, val_size);
    std::string test_data = text.substr(train_size + val_size);
    
    std::ofstream train_file(output_dir + "/train.txt");
    std::ofstream val_file(output_dir + "/val.txt");
    std::ofstream test_file(output_dir + "/test.txt");
    
    train_file << train_data;
    val_file << val_data;
    test_file << test_data;
    
    std::cout << "Split dataset:\n";
    std::cout << "  Train: " << train_data.size() << " chars\n";
    std::cout << "  Val:   " << val_data.size() << " chars\n";
    std::cout << "  Test:  " << test_data.size() << " chars\n";
}

} // namespace data
} // namespace kaler
