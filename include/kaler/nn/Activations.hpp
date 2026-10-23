#pragma once

#include "kaler/math/Matrix.hpp"

namespace kaler::nn {

[[nodiscard]] double relu(double x) noexcept;
[[nodiscard]] double tanhActivation(double x) noexcept;
[[nodiscard]] kaler::math::Matrix relu(const kaler::math::Matrix& input);
[[nodiscard]] kaler::math::Matrix tanhActivation(const kaler::math::Matrix& input);

} // namespace kaler::nn
