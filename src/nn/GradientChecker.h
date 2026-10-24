#ifndef KALER_GRADIENT_CHECKER_H
#define KALER_GRADIENT_CHECKER_H

#include "Parameter.h"
#include "kaler/math/Matrix.hpp"
#include <functional>
#include <iostream>
#include <cmath>

namespace kaler {
namespace nn {

class GradientChecker {
public:
    using LossFunction = std::function<double()>;
    
    static bool check_gradients(Parameter& param,
                               LossFunction loss_fn,
                               double epsilon = 1e-4,
                               double tolerance = 1e-3);
    
    static void print_gradient_comparison(const math::Matrix& analytical,
                                         const math::Matrix& numerical);
};

} // namespace nn
} // namespace kaler

#endif // KALER_GRADIENT_CHECKER_H
