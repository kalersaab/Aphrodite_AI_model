# Training Data

## Stage 1: Debugging Dataset (10-100 KB)

### tiny_shakespeare.txt (~8 KB)
- **Purpose**: Verify training pipeline works
- **Source**: Shakespeare's Coriolanus (excerpt)
- **Size**: ~8 KB
- **Characters**: ~8,000
- **Use for**: 
  - Testing tokenization
  - Verifying gradient computation
  - Debugging training loop
  - Testing inference

### Next Steps
- **Stage 2**: Expand to 1-10 MB (TinyStories or WikiText-2)
- **Stage 3**: Scale to 50-200 MB (WikiText-103 or OpenWebText subset)

## Data Processing Pipeline

1. **Raw data** → `data/raw/`
2. **Tokenized data** → `data/processed/`
3. **Training splits** → train/val/test

## Usage

```bash
# From project root
./aphrodite --data data/raw/tiny_shakespeare.txt --train
```
