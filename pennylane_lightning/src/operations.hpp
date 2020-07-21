#include <iostream>
#include <Eigen/Dense>
//#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include <unsupported/Eigen/CXX11/Tensor>
//#include <bitset>

using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::VectorXcd;
using Eigen::Tensor;

using Gate_1q = Eigen::Tensor<std::complex<double>, 2>;
using Gate_2q = Eigen::Tensor<std::complex<double>, 4>;
using Gate_3q = Eigen::Tensor<std::complex<double>, 6>;


const double SQRT_2 = sqrt(2);

Gate_1q Identity() {
    Gate_1q X(2, 2);
    X.setValues({{1, 0}, {0, 1}});
    return X;
}


Gate_1q X() {
    Gate_1q X(2, 2);
    X.setValues({{0, 1}, {1, 0}});
    return X;
}


Gate_1q Y() {
    Gate_1q Y(2, 2);

    std::complex<double> Fst(0, -1);
    std::complex<double> Snd(0, 1);
    Y.setValues({{0, Fst}, {Snd, 0}});
    return Y;
}

Gate_1q Z() {
    Gate_1q Z(2, 2);
    Z.setValues({{1, 0}, {0, -1}});
    return Z;
}

Gate_1q H() {
    Gate_1q H(2, 2);
    H.setValues({{1/SQRT_2, 1/SQRT_2}, {1/SQRT_2, -1/SQRT_2}});
    return H;
}
