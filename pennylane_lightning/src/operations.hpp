#include "pybind11/eigen.h"
#include <iostream>
#include <Eigen/Dense>
//#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include <unsupported/Eigen/CXX11/Tensor>
//#include <bitset>

using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::VectorXcd;
using Eigen::Tensor;

Eigen::Tensor<std::complex<double>,2> X() {
    Eigen::Tensor<std::complex<double>,2> X(2, 2);
    X(0, 0) = 0;
    X(0, 1) = 1;
    X(1, 0) = 1;
    X(1, 1) = 0;
    return X;
}



