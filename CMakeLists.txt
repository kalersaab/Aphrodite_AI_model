cmake_minimum_required(VERSION 3.20)
project(aphrodite VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(kaler_math
    src/math/Matrix.cpp
    src/nn/Activations.cpp
    src/data/DataLoader.cpp
)
target_include_directories(kaler_math PUBLIC include src)
target_compile_options(kaler_math PRIVATE
    $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-Wall;-Wextra;-Wpedantic>
)

add_executable(aphrodite src/main.cpp)
target_link_libraries(aphrodite PRIVATE kaler_math)

enable_testing()
add_executable(kaler_tests tests/test_math.cpp)
target_link_libraries(kaler_tests PRIVATE kaler_math)
add_test(NAME kaler_math_tests COMMAND kaler_tests)

add_executable(test_dataloader tests/test_dataloader.cpp)
target_link_libraries(test_dataloader PRIVATE kaler_math)
add_test(NAME dataloader_tests COMMAND test_dataloader)
