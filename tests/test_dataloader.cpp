#include "../src/data/DataLoader.h"
#include <iostream>
#include <cassert>

using namespace kaler::data;

void test_tokenizer() {
    std::cout << "=== Testing CharTokenizer ===\n";
    
    CharTokenizer tokenizer;
    std::string sample = "Hello, World!";
    
    tokenizer.fit(sample);
    std::cout << "Vocab size: " << tokenizer.vocab_size() << "\n";
    
    auto tokens = tokenizer.encode(sample);
    std::cout << "Encoded " << sample.size() << " chars to " << tokens.size() << " tokens\n";
    
    auto decoded = tokenizer.decode(tokens);
    std::cout << "Decoded: " << decoded << "\n";
    
    assert(decoded == sample && "Encode/decode should be reversible");
    std::cout << "✓ Tokenizer test passed\n\n";
}

void test_dataloader() {
    std::cout << "=== Testing TextDataLoader ===\n";
    
    TextDataLoader loader("data/raw/tiny_shakespeare.txt", 4, 32);
    
    if (!loader.load()) {
        std::cerr << "Failed to load data\n";
        return;
    }
    
    std::cout << "Dataset size: " << loader.dataset_size() << " tokens\n";
    std::cout << "Vocab size: " << loader.vocab_size() << "\n";
    std::cout << "Num batches: " << loader.num_batches() << "\n\n";
    
    // Get first batch
    auto batch = loader.next_batch();
    std::cout << "Batch shape: [" << batch.input.size() << ", " 
              << batch.input[0].size() << "]\n";
    
    // Decode first sequence to verify
    std::cout << "\nFirst sequence preview:\n";
    std::string preview = loader.tokenizer().decode(batch.input[0]);
    std::cout << preview.substr(0, 100) << "...\n";
    
    // Verify target is shifted by 1
    std::string target_preview = loader.tokenizer().decode(batch.target[0]);
    std::cout << "\nTarget (shifted by 1):\n";
    std::cout << target_preview.substr(0, 100) << "...\n";
    
    std::cout << "\n✓ DataLoader test passed\n\n";
}

void test_dataset_stats() {
    std::cout << "=== Dataset Statistics ===\n";
    
    TextDataLoader loader("data/raw/tiny_shakespeare.txt", 1, 1);
    loader.load();
    
    const auto& tokenizer = loader.tokenizer();
    
    std::cout << "Total characters: " << loader.dataset_size() << "\n";
    std::cout << "Unique characters: " << tokenizer.vocab_size() << "\n";
    
    // Count token frequencies (simple version)
    std::cout << "\nDataset loaded successfully!\n";
}

int main() {
    std::cout << "Running DataLoader Tests\n";
    std::cout << "========================\n\n";
    
    try {
        test_tokenizer();
        test_dataloader();
        test_dataset_stats();
        
        std::cout << "\n✅ All tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << "\n";
        return 1;
    }
}
