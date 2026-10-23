CXX = clang++
CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -O2
INCLUDES = -Iinclude -Isrc

# Source files
MATH_SRCS = src/math/Matrix.cpp src/nn/Activations.cpp src/data/DataLoader.cpp \
            src/nn/Parameter.cpp src/nn/Dense.cpp src/nn/Loss.cpp \
            src/nn/Optimizer.cpp src/nn/GradientChecker.cpp
MATH_OBJS = $(MATH_SRCS:.cpp=.o)

# Targets
all: test_dataloader test_training kaler_gpt kaler_tests

# Data loader test
test_dataloader: tests/test_dataloader.o $(MATH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Training test
test_training: tests/test_training.o $(MATH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Main executable
kaler_gpt: src/main.o $(MATH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Math tests
kaler_tests: tests/test_math.o $(MATH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(MATH_OBJS) src/main.o tests/test_math.o tests/test_dataloader.o tests/test_training.o
	rm -f kaler_gpt kaler_tests test_dataloader test_training

# Run data loader test
test-data: test_dataloader
	./test_dataloader

# Run training test
test-train: test_training
	./test_training

.PHONY: all clean test-data test-train
