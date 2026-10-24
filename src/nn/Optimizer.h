#ifndef KALER_OPTIMIZER_H
#define KALER_OPTIMIZER_H

#include "Parameter.h"
#include <vector>
#include <memory>

namespace kaler {
namespace nn {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    
    void add_parameter(Parameter* param) {
        parameters_.push_back(param);
    }

protected:
    std::vector<Parameter*> parameters_;
};

class SGD : public Optimizer {
public:
    explicit SGD(double learning_rate = 0.01)
        : learning_rate_(learning_rate) {}
    
    void step() override;
    void zero_grad() override;

private:
    double learning_rate_;
};

class Adam : public Optimizer {
public:
    explicit Adam(double learning_rate = 0.001,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double epsilon = 1e-8)
        : learning_rate_(learning_rate)
        , beta1_(beta1)
        , beta2_(beta2)
        , epsilon_(epsilon)
        , t_(0) {}
    
    void step() override;
    void zero_grad() override;

public:
    void set_learning_rate(double learning_rate) { learning_rate_ = learning_rate; }
    double learning_rate() const { return learning_rate_; }

private:
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    int t_;
    
    std::vector<math::Matrix> m_;
    std::vector<math::Matrix> v_;
};

} // namespace nn
} // namespace kaler

#endif // KALER_OPTIMIZER_H
