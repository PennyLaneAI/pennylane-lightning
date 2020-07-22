#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// Generating aliases for Eigen Tensor types of rank 1-25
#define GENERATE_TENSOR_DEF(Argument) using Tensor_##Argument = Eigen::Tensor<std::complex<double>, Argument>;

GENERATE_TENSOR_DEF(1)
GENERATE_TENSOR_DEF(2)
GENERATE_TENSOR_DEF(3)
GENERATE_TENSOR_DEF(4)
GENERATE_TENSOR_DEF(5)
GENERATE_TENSOR_DEF(6)
GENERATE_TENSOR_DEF(7)
GENERATE_TENSOR_DEF(8)
GENERATE_TENSOR_DEF(9)
GENERATE_TENSOR_DEF(10)
GENERATE_TENSOR_DEF(11)
GENERATE_TENSOR_DEF(12)
GENERATE_TENSOR_DEF(13)
GENERATE_TENSOR_DEF(14)
GENERATE_TENSOR_DEF(15)
GENERATE_TENSOR_DEF(16)
GENERATE_TENSOR_DEF(17)
GENERATE_TENSOR_DEF(18)
GENERATE_TENSOR_DEF(19)
GENERATE_TENSOR_DEF(20)
GENERATE_TENSOR_DEF(21)
GENERATE_TENSOR_DEF(22)
GENERATE_TENSOR_DEF(23)
GENERATE_TENSOR_DEF(24)
GENERATE_TENSOR_DEF(25)

using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::VectorXcd;
using Eigen::Tensor;

using State_1q = Eigen::Tensor<std::complex<double>, 1>;
using State_2q = Eigen::Tensor<std::complex<double>, 2>;
using State_3q = Eigen::Tensor<std::complex<double>, 3>;

using Gate_1q = Eigen::Tensor<std::complex<double>, 2>;
using Gate_2q = Eigen::Tensor<std::complex<double>, 4>;
using Gate_3q = Eigen::Tensor<std::complex<double>, 6>;

using Pairs = Eigen::IndexPair<int>;
using Pairs_1q = Eigen::array<Pairs, 1>;
using Pairs_2q = Eigen::array<Pairs, 2>;


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
    Gate_1q T(2, 2);

    const std::complex<double> exponent(0, -M_PI/4);
    T.setValues({{1, 0}, {0, std::pow(M_E, exponent)}});
    return T;
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

Gate_2q CNOT() {
    Gate_2q CNOT(2,2,2,2);
    CNOT.setValues({{{{1, 0},{0, 0}},{{0, 1},{0, 0}}},{{{0, 0},{0, 1}},{{0, 0},{1, 0}}}});
    return CNOT;
}
