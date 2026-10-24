#ifndef KALER_DATALOADER_H
#define KALER_DATALOADER_H

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <memory>

namespace kaler {
namespace data {

/**
 * Simple character-level tokenizer for debugging
 * Converts text to integer sequences
 */
class CharTokenizer {
public:
    CharTokenizer() = default;
    
    // Build vocabulary from text
    void fit(const std::string& text);
    
    // Encode text to integers
    std::vector<int> encode(const std::string& text) const;
    
    // Decode integers back to text
    std::string decode(const std::vector<int>& tokens) const;
    
    // Get vocabulary size
    size_t vocab_size() const { return char_to_idx_.size() + 2; }
    
    // Get special tokens
    int pad_token() const { return 0; }
    int unk_token() const { return 1; }

private:
    std::unordered_map<std::string, int> char_to_idx_;
    std::unordered_map<int, std::string> idx_to_char_;
    
    void add_char(const std::string& token);
};

/**
 * Simple text data loader
 * Loads text files and provides batches for training
 */
class TextDataLoader {
public:
    TextDataLoader(const std::string& filepath, 
                   size_t batch_size = 32,
                   size_t seq_length = 64);
    
    // Load and tokenize the data
    bool load();
    
    // Get the next batch
    struct Batch {
        std::vector<std::vector<int>> input;   // [batch_size, seq_length]
        std::vector<std::vector<int>> target;  // [batch_size, seq_length]
    };
    
    Batch next_batch();
    
    // Reset to beginning
    void reset();
    
    // Check if we have more batches
    bool has_next() const;
    
    // Get dataset statistics
    size_t num_batches() const;
    size_t dataset_size() const { return tokens_.size(); }
    const std::vector<int>& tokens() const { return tokens_; }
    size_t vocab_size() const { return tokenizer_.vocab_size(); }
    
    // Get the tokenizer
    const CharTokenizer& tokenizer() const { return tokenizer_; }

private:
    std::string filepath_;
    size_t batch_size_;
    size_t seq_length_;
    size_t current_pos_;
    
    std::vector<int> tokens_;
    CharTokenizer tokenizer_;
    
    std::string load_file();
};

/**
 * Train/Val/Test split utility
 */
class DatasetSplitter {
public:
    static void split_file(const std::string& input_path,
                          const std::string& output_dir,
                          float train_ratio = 0.8f,
                          float val_ratio = 0.1f);
};

} // namespace data
} // namespace kaler

#endif // KALER_DATALOADER_H
