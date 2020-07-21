#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
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


extern const double SQRT_2 = sqrt(2);
extern const std::complex<double> IMAG(0, 1);
extern const std::complex<double> NEGATIVE_IMAG(0, -1);

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
    Y.setValues({{0, NEGATIVE_IMAG}, {IMAG, 0}});
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

Gate_1q S() {
    Gate_1q S(2, 2);
    S.setValues({{1, 0}, {0, IMAG}});
    return S;
}

Gate_1q T() {
    Gate_1q S(2, 2);

    const std::complex<double> exponent(0, -M_PI/4);
    S.setValues({{1, 0}, {0, std::pow(M_E, exponent)}});
    return S;
}

Gate_1q RX(const double& parameter) {
    Gate_1q RX(2, 2);

    const std::complex<double> c (std::cos(parameter / 2), 0);
    const std::complex<double> js (0, std::sin(-parameter / 2));

    RX.setValues({{c, js}, {js, c}});
    return RX;
}

Gate_1q RY(const double& parameter) {
    Gate_1q RY(2, 2);

    const double c = std::cos(parameter / 2);
    const double s = std::sin(parameter / 2);

    RY.setValues({{c, -s}, {s, c}});
    return RY;
}

Gate_1q RZ(const double& parameter) {
    Gate_1q RZ(2, 2);

    const std::complex<double> exponent(0, -parameter/2);
    const std::complex<double> exponent_second(0, parameter/2);
    const std::complex<double> first = std::pow(M_E, exponent);
    const std::complex<double> second = std::pow(M_E, exponent_second);

    RZ.setValues({{first, 0}, {0, second}});
    return RZ;
}
