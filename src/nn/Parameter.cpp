#include "Parameter.h"
#include <random>
#include <cmath>

namespace kaler {
namespace nn {

Parameter::Parameter(size_t rows, size_t cols, const std::string& name)
    : data_(rows, cols, 0.0)
    , grad_(rows, cols, 0.0)
    , name_(name) {
}

void Parameter::xavier_init() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    double fan_in = static_cast<double>(data_.cols());
    double fan_out = static_cast<double>(data_.rows());
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    
    std::uniform_real_distribution<double> dist(-limit, limit);
    
    for (size_t i = 0; i < data_.rows(); ++i) {
        for (size_t j = 0; j < data_.cols(); ++j) {
            const_cast<double&>(data_(i, j)) = dist(gen);
        }
    }
}

void Parameter::zero_init() {
    for (size_t i = 0; i < data_.rows(); ++i) {
        for (size_t j = 0; j < data_.cols(); ++j) {
            const_cast<double&>(data_(i, j)) = 0.0;
        }
    }
}

void Parameter::random_init(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);
    
    for (size_t i = 0; i < data_.rows(); ++i) {
        for (size_t j = 0; j < data_.cols(); ++j) {
            const_cast<double&>(data_(i, j)) = dist(gen);
        }
    }
}

void Parameter::zero_grad() {
    for (size_t i = 0; i < grad_.rows(); ++i) {
        for (size_t j = 0; j < grad_.cols(); ++j) {
            const_cast<double&>(grad_(i, j)) = 0.0;
        }
    }
}

} // namespace nn
} // namespace kaler
