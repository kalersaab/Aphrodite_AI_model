#ifndef KALER_PARAMETER_H
#define KALER_PARAMETER_H

#include "kaler/math/Matrix.hpp"
#include <memory>
#include <string>

namespace kaler {
namespace nn {

class Parameter {
public:
    Parameter(size_t rows, size_t cols, const std::string& name = "");
    
    void xavier_init();
    void zero_init();
    void random_init(double min = -0.1, double max = 0.1);
    
    math::Matrix& data() { return data_; }
    const math::Matrix& data() const { return data_; }
    
    math::Matrix& grad() { return grad_; }
    const math::Matrix& grad() const { return grad_; }
    
    void zero_grad();
    
    const std::string& name() const { return name_; }

private:
    math::Matrix data_;
    math::Matrix grad_;
    std::string name_;
};

} // namespace nn
} // namespace kaler

#endif // KALER_PARAMETER_H
