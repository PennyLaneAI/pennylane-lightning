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

Eigen::Tensor<std::complex<double>,2> Identity() {
    Eigen::Tensor<std::complex<double>,2> X(2, 2);
    X.setValues({{1, 0}, {0, 1}});
    return X;
}


Eigen::Tensor<std::complex<double>,2> X() {
    Eigen::Tensor<std::complex<double>,2> X(2, 2);
    X.setValues({{0, 1}, {1, 0}});
    return X;
}


Eigen::Tensor<std::complex<double>,2> Y() {
    Eigen::Tensor<std::complex<double>,2> Y(2, 2);

    std::complex<double> Fst(0, -1);
    std::complex<double> Snd(0, 1);
    Y.setValues({{0, Fst}, {Snd, 0}});
    return Y;
}

Eigen::Tensor<std::complex<double>,2> Z() {
    Eigen::Tensor<std::complex<double>,2> Z(2, 2);
    Z.setValues({{1, 0}, {0, -1}});
    return Z;
}
