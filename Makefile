CXX = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -O2
INCLUDES = -Iinclude -Isrc

# Source files
MATH_SRCS = src/math/Matrix.cpp src/nn/Activations.cpp src/data/DataLoader.cpp \
            src/nn/Parameter.cpp src/nn/Dense.cpp src/nn/Loss.cpp \
            src/nn/Optimizer.cpp src/nn/GradientChecker.cpp \
            src/nn/Embedding.cpp src/nn/LayerNorm.cpp src/nn/Attention.cpp \
            src/nn/TransformerBlock.cpp src/nn/GPT.cpp
MATH_OBJS = $(MATH_SRCS:.cpp=.o)

# Targets
all: test_dataloader test_training test_gpt aphrodite aphrodite_tests

# Data loader test
test_dataloader: tests/test_dataloader.o $(MATH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Training test
test_training: tests/test_training.o $(MATH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# GPT test
test_gpt: tests/test_gpt.o $(MATH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Main executable
aphrodite: src/main.o $(MATH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Math tests
aphrodite_tests: tests/test_math.o $(MATH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(MATH_OBJS) src/main.o tests/test_math.o tests/test_dataloader.o tests/test_training.o tests/test_gpt.o
	rm -f aphrodite aphrodite_tests test_dataloader test_training test_gpt

# Run data loader test
test-data: test_dataloader
	./test_dataloader

# Run training test
test-train: test_training
	./test_training

# Run GPT test
test-gpt-run: test_gpt
	./test_gpt

.PHONY: all clean test-data test-train test-gpt-run
