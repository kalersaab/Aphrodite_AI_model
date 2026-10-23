#include "kaler/nn/Activations.hpp"

#include <cmath>

namespace kaler::nn {

double relu(double x) noexcept { return x > 0.0 ? x : 0.0; }
double tanhActivation(double x) noexcept { return std::tanh(x); }

kaler::math::Matrix relu(const kaler::math::Matrix& input) {
    kaler::math::Matrix out(input.rows(), input.cols());
    for (std::size_t r = 0; r < input.rows(); ++r)
        for (std::size_t c = 0; c < input.cols(); ++c)
            out(r, c) = relu(input(r, c));
    return out;
}

kaler::math::Matrix tanhActivation(const kaler::math::Matrix& input) {
    kaler::math::Matrix out(input.rows(), input.cols());
    for (std::size_t r = 0; r < input.rows(); ++r)
        for (std::size_t c = 0; c < input.cols(); ++c)
            out(r, c) = tanhActivation(input(r, c));
    return out;
}

} // namespace kaler::nn
