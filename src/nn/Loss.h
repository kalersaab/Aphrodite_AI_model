#ifndef KALER_LOSS_H
#define KALER_LOSS_H

#include "kaler/math/Matrix.hpp"
#include <cmath>
#include <algorithm>

namespace kaler {
namespace nn {

class Loss {
public:
    static math::Matrix softmax(const math::Matrix& logits);
    
    static double cross_entropy(const math::Matrix& logits, const math::Matrix& targets);
    
    static math::Matrix cross_entropy_grad(const math::Matrix& logits, const math::Matrix& targets);
};

} // namespace nn
} // namespace kaler

#endif // KALER_LOSS_H
