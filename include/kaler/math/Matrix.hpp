#pragma once

#include <cstddef>
#include <initializer_list>
#include <vector>

namespace kaler::math {

class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0);
    Matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> values);

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    double& operator()(std::size_t row, std::size_t col);
    const double& operator()(std::size_t row, std::size_t col) const;

    [[nodiscard]] const std::vector<double>& data() const noexcept { return data_; }

    [[nodiscard]] Matrix transpose() const;
    [[nodiscard]] Matrix softmaxRows() const;

    static Matrix zeros(std::size_t rows, std::size_t cols);
    static Matrix identity(std::size_t size);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& a, double scalar);
Matrix operator*(double scalar, const Matrix& a);
Matrix matmul(const Matrix& a, const Matrix& b);

} // namespace kaler::math
