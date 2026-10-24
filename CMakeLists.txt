cmake_minimum_required(VERSION 3.20)
project(aphrodite VERSION 0.2.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(kaler_math
    src/math/Matrix.cpp
    src/nn/Activations.cpp
    src/data/DataLoader.cpp
    src/nn/Parameter.cpp
    src/nn/Dense.cpp
    src/nn/Loss.cpp
    src/nn/Optimizer.cpp
    src/nn/Embedding.cpp
    src/nn/LayerNorm.cpp
    src/nn/Attention.cpp
    src/nn/TransformerBlock.cpp
    src/nn/GPT.cpp
    src/nn/GradientChecker.cpp
    src/nn/Trainer.cpp
)
target_include_directories(kaler_math PUBLIC include src)
target_compile_options(kaler_math PRIVATE $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>)

add_executable(aphrodite src/main.cpp)
target_link_libraries(aphrodite PRIVATE kaler_math)
add_executable(aphrodite_train src/train_main.cpp)
target_link_libraries(aphrodite_train PRIVATE kaler_math)

enable_testing()
add_executable(kaler_tests tests/test_math.cpp)
target_link_libraries(kaler_tests PRIVATE kaler_math)
add_test(NAME kaler_math_tests COMMAND kaler_tests)
add_executable(test_dataloader tests/test_dataloader.cpp)
target_link_libraries(test_dataloader PRIVATE kaler_math)
add_test(NAME dataloader_tests COMMAND test_dataloader)
add_executable(test_gpt tests/test_gpt.cpp)
target_link_libraries(test_gpt PRIVATE kaler_math)
add_test(NAME gpt_tests COMMAND test_gpt)
add_executable(test_training tests/test_training.cpp)
target_link_libraries(test_training PRIVATE kaler_math)
add_test(NAME training_tests COMMAND test_training)

add_executable(test_checkpoint tests/test_checkpoint.cpp)
target_link_libraries(test_checkpoint PRIVATE kaler_math)
add_test(NAME checkpoint_tests COMMAND test_checkpoint)

add_executable(test_numerical_stability tests/test_numerical_stability.cpp)
target_link_libraries(test_numerical_stability PRIVATE kaler_math)
add_test(NAME numerical_stability_tests COMMAND test_numerical_stability)

add_executable(test_gradient_flow tests/test_gradient_flow.cpp)
target_link_libraries(test_gradient_flow PRIVATE kaler_math)
add_test(NAME gradient_flow_tests COMMAND test_gradient_flow)
