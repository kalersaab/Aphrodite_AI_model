#include "kaler/math/Matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kaler::math {
namespace {
void requireSameShape(const Matrix& a, const Matrix& b, const char* operation) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::invalid_argument(std::string(operation) + ": matrix shape mismatch");
    }
}
}

Matrix::Matrix(std::size_t rows, std::size_t cols, double fill)
    : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> values)
    : rows_(rows), cols_(cols), data_(values) {
    if (data_.size() != rows_ * cols_) {
        throw std::invalid_argument("Matrix initializer size does not match rows * cols");
    }
}

double& Matrix::operator()(std::size_t row, std::size_t col) {
    if (row >= rows_ || col >= cols_) throw std::out_of_range("Matrix index out of range");
    return data_[row * cols_ + col];
}

const double& Matrix::operator()(std::size_t row, std::size_t col) const {
    if (row >= rows_ || col >= cols_) throw std::out_of_range("Matrix index out of range");
    return data_[row * cols_ + col];
}

Matrix Matrix::zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::identity(std::size_t size) {
    Matrix result(size, size);
    for (std::size_t i = 0; i < size; ++i) result(i, i) = 1.0;
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) result(c, r) = (*this)(r, c);
    }
    return result;
}

Matrix Matrix::softmaxRows() const {
    Matrix result(rows_, cols_);
    if (cols_ == 0) return result;

    for (std::size_t r = 0; r < rows_; ++r) {
        double maxValue = (*this)(r, 0);
        for (std::size_t c = 1; c < cols_; ++c) {
            maxValue = std::max(maxValue, (*this)(r, c));
        }

        double sum = 0.0;
        for (std::size_t c = 0; c < cols_; ++c) {
            const double e = std::exp((*this)(r, c) - maxValue);
            result(r, c) = e;
            sum += e;
        }

        if (sum == 0.0 || !std::isfinite(sum)) {
            throw std::runtime_error("softmax encountered a non-finite normalization sum");
        }
        for (std::size_t c = 0; c < cols_; ++c) result(r, c) /= sum;
    }
    return result;
}

Matrix operator+(const Matrix& a, const Matrix& b) {
    requireSameShape(a, b, "addition");
    Matrix out(a.rows(), a.cols());
    for (std::size_t r = 0; r < a.rows(); ++r)
        for (std::size_t c = 0; c < a.cols(); ++c)
            out(r, c) = a(r, c) + b(r, c);
    return out;
}

Matrix operator-(const Matrix& a, const Matrix& b) {
    requireSameShape(a, b, "subtraction");
    Matrix out(a.rows(), a.cols());
    for (std::size_t r = 0; r < a.rows(); ++r)
        for (std::size_t c = 0; c < a.cols(); ++c)
            out(r, c) = a(r, c) - b(r, c);
    return out;
}

Matrix operator*(const Matrix& a, double scalar) {
    Matrix out(a.rows(), a.cols());
    for (std::size_t r = 0; r < a.rows(); ++r)
        for (std::size_t c = 0; c < a.cols(); ++c)
            out(r, c) = a(r, c) * scalar;
    return out;
}

Matrix operator*(double scalar, const Matrix& a) { return a * scalar; }

Matrix matmul(const Matrix& a, const Matrix& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("matmul: inner dimensions must match");
    }

    Matrix out(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t k = 0; k < a.cols(); ++k) {
            const double aik = a(i, k);
            for (std::size_t j = 0; j < b.cols(); ++j) {
                out(i, j) += aik * b(k, j);
            }
        }
    }
    return out;
}

} // namespace kaler::math
