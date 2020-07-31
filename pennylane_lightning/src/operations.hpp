// Copyright 2020 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * \rst
 * Contains tensor representations of supported gates in ``lightning.qubit``.
 * \endrst
 */
#pragma once

#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

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


const double SQRT_2 = sqrt(2);
const std::complex<double> IMAG(0, 1);
const std::complex<double> NEGATIVE_IMAG(0, -1);

/**
* Generates the identity gate.
*
* @return the identity tensor
*/
Gate_1q Identity() {
    Gate_1q X(2, 2);
    X.setValues({{1, 0}, {0, 1}});
    return X;
}

/**
* Generates the X gate.
*
* @return the X tensor
*/
Gate_1q X() {
    Gate_1q X(2, 2);
    X.setValues({{0, 1}, {1, 0}});
    return X;
}

/**
* Generates the Y gate.
*
* @return the Y tensor
*/
Gate_1q Y() {
    Gate_1q Y(2, 2);
    Y.setValues({{0, NEGATIVE_IMAG}, {IMAG, 0}});
    return Y;
}

/**
* Generates the Z gate.
*
* @return the Z tensor
*/
Gate_1q Z() {
    Gate_1q Z(2, 2);
    Z.setValues({{1, 0}, {0, -1}});
    return Z;
}

/**
* Generates the H gate.
*
* @return the H tensor
*/
Gate_1q H() {
    Gate_1q H(2, 2);
    H.setValues({{1/SQRT_2, 1/SQRT_2}, {1/SQRT_2, -1/SQRT_2}});
    return H;
}

/**
* Generates the S gate.
*
* @return the S tensor
*/
Gate_1q S() {
    Gate_1q S(2, 2);
    S.setValues({{1, 0}, {0, IMAG}});
    return S;
}

/**
* Generates the T gate.
*
* @return the T tensor
*/
Gate_1q T() {
    Gate_1q T(2, 2);

    const std::complex<double> exponent(0, M_PI/4);
    T.setValues({{1, 0}, {0, std::pow(M_E, exponent)}});
    return T;
}

/**
* Generates the X rotation gate.
*
* @param parameter the rotation angle
* @return the RX tensor
*/
Gate_1q RX(const double& parameter) {
    Gate_1q RX(2, 2);

    const std::complex<double> c (std::cos(parameter / 2), 0);
    const std::complex<double> js (0, std::sin(-parameter / 2));

    RX.setValues({{c, js}, {js, c}});
    return RX;
}

/**
* Generates the Y rotation gate.
*
* @param parameter the rotation angle
* @return the RY tensor
*/
Gate_1q RY(const double& parameter) {
    Gate_1q RY(2, 2);

    const double c = std::cos(parameter / 2);
    const double s = std::sin(parameter / 2);

    RY.setValues({{c, -s}, {s, c}});
    return RY;
}

/**
* Generates the Z rotation gate.
*
* @param parameter the rotation angle
* @return the RZ tensor
*/
Gate_1q RZ(const double& parameter) {
    Gate_1q RZ(2, 2);

    const std::complex<double> exponent(0, -parameter/2);
    const std::complex<double> exponent_second(0, parameter/2);
    const std::complex<double> first = std::pow(M_E, exponent);
    const std::complex<double> second = std::pow(M_E, exponent_second);

    RZ.setValues({{first, 0}, {0, second}});
    return RZ;
}

/**
* Generates the phase-shift gate.
*
* @param parameter the phase shift
* @return the phase-shift tensor
*/
Gate_1q PhaseShift(const double& parameter) {
    Gate_1q PhaseShift(2, 2);

    const std::complex<double> exponent(0, parameter);
    const std::complex<double> shift = std::pow(M_E, exponent);

    PhaseShift.setValues({{1, 0}, {0, shift}});
    return PhaseShift;
}

/**
* Generates the arbitrary single qubit rotation gate.
*
* The rotation is achieved through three separate rotations:
* \f$R(\phi, \theta, \omega)= RZ(\omega)RY(\theta)RZ(\phi)\f$.
*
* @param phi the first rotation angle
* @param theta the second rotation angle
* @param omega the third rotation angle
* @return the rotation tensor
*/
Gate_1q Rot(const double& phi, const double& theta, const double& omega) {
    Gate_1q Rot(2, 2);

    const std::complex<double> e00(0, (-phi - omega)/2);
    const std::complex<double> e10(0, (-phi + omega)/2);
    const std::complex<double> e01(0, (phi - omega)/2);
    const std::complex<double> e11(0, (phi + omega)/2);

    const std::complex<double> exp00 = std::pow(M_E, e00);
    const std::complex<double> exp10 = std::pow(M_E, e10);
    const std::complex<double> exp01 = std::pow(M_E, e01);
    const std::complex<double> exp11 = std::pow(M_E, e11);

    const double c = std::cos(theta / 2);
    const double s = std::sin(theta / 2);

    Rot.setValues({{exp00 * c, -exp01 * s}, {exp10 * s, exp11 * c}});

    return Rot;
}

/**
* Generates the CNOT gate.
*
* @return the CNOT tensor
*/
Gate_2q CNOT() {
    Gate_2q CNOT(2,2,2,2);
    CNOT.setValues({{{{1, 0},{0, 0}},{{0, 1},{0, 0}}},{{{0, 0},{0, 1}},{{0, 0},{1, 0}}}});
    return CNOT;
}

/**
* Generates the SWAP gate.
*
* @return the SWAP tensor
*/
Gate_2q SWAP() {
    Gate_2q SWAP(2,2,2,2);
    SWAP.setValues({{{{1, 0},{0, 0}},{{0, 0},{1, 0}}},{{{0, 1},{0, 0}},{{0, 0},{0, 1}}}});
    return SWAP;
}

/**
* Generates the CZ gate.
*
* @return the CZ tensor
*/
Gate_2q CZ() {
    Gate_2q CZ(2,2,2,2);
    CZ.setValues({{{{1, 0},{0, 0}},{{0, 1},{0, 0}}},{{{0, 0},{1, 0}},{{0, 0},{0, -1}}}});
    return CZ;
}

/**
* Generates the Toffoli gate.
*
* @return the Toffoli tensor
*/
Gate_3q Toffoli() {
    Gate_3q Toffoli(2,2,2,2,2,2);
    Toffoli.setValues({{{{{{1, 0},{0, 0}},{{0, 0},{0, 0}}},{{{0, 1},{0, 0}},{{0, 0},{0, 0}}}},
            {{{{0, 0},{1, 0}},{{0, 0},{0, 0}}},{{{0, 0},{0, 1}},{{0, 0},{0, 0}}}}
        },
        {   {{{{0, 0},{0, 0}},{{1, 0},{0, 0}}},{{{0, 0},{0, 0}},{{0, 1},{0, 0}}}},
            {{{{0, 0},{0, 0}},{{0, 0},{0, 1}}},{{{0, 0},{0, 0}},{{0, 0},{1, 0}}}}
        }});
    return Toffoli;
}

/**
* Generates the CSWAP gate.
*
* @return the CSWAP tensor
*/
Gate_3q CSWAP() {
    Gate_3q CSWAP(2,2,2,2,2,2);
    CSWAP.setValues({{{{{{1, 0},{0, 0}},{{0, 0},{0, 0}}},{{{0, 1},{0, 0}},{{0, 0},{0, 0}}}},
            {{{{0, 0},{1, 0}},{{0, 0},{0, 0}}},{{{0, 0},{0, 1}},{{0, 0},{0, 0}}}}
        },
        {   {{{{0, 0},{0, 0}},{{1, 0},{0, 0}}},{{{0, 0},{0, 0}},{{0, 0},{1, 0}}}},
            {{{{0, 0},{0, 0}},{{0, 1},{0, 0}}},{{{0, 0},{0, 0}},{{0, 0},{0, 1}}}}
        }});
    return CSWAP;
}

/**
* Generates the controlled-X rotation gate.
*
* @param parameter the rotation angle
* @return the CRX tensor
*/
Gate_2q CRX(const double& parameter) {
    Gate_2q CRX(2, 2, 2, 2);

    const std::complex<double> c (std::cos(parameter / 2), 0);
    const std::complex<double> js (0, std::sin(-parameter / 2));

    CRX.setValues({{{{1, 0},{0, 0}},{{0, 1},{0, 0}}},{{{0, 0},{c, js}},{{0, 0},{js, c}}}});
    return CRX;
}

/**
* Generates the controlled-Y rotation gate.
*
* @param parameter the rotation angle
* @return the CRY tensor
*/
Gate_2q CRY(const double& parameter) {
    Gate_2q CRY(2, 2, 2, 2);

    const double c = std::cos(parameter / 2);
    const double s = std::sin(parameter / 2);

    CRY.setValues({{{{1, 0},{0, 0}},{{0, 1},{0, 0}}},{{{0, 0},{c, -s}},{{0, 0},{s, c}}}});
    return CRY;
}

/**
* Generates the controlled-Z rotation gate.
*
* @param parameter the rotation angle
* @return the CRZ tensor
*/
Gate_2q CRZ(const double& parameter) {
    Gate_2q CRZ(2, 2, 2, 2);

    const std::complex<double> exponent(0, -parameter/2);
    const std::complex<double> exponent_second(0, parameter/2);
    const std::complex<double> first = std::pow(M_E, exponent);
    const std::complex<double> second = std::pow(M_E, exponent_second);

    CRZ.setValues({{{{1, 0},{0, 0}},{{0, 1},{0, 0}}},{{{0, 0},{first, 0}},{{0, 0},{0, second}}}});
    return CRZ;
}

/**
* Generates the controlled rotation gate.
*
* This gate implements a rotation on a target qubit depending on a control qubit. The rotation
* on the target qubit is achieved through three separate rotations:
* \f$R(\phi, \theta, \omega)= RZ(\omega)RY(\theta)RZ(\phi)\f$.
*
* @param phi the first rotation angle
* @param theta the second rotation angle
* @param omega the third rotation angle
* @return the controlled rotation tensor
*/
Gate_2q CRot(const double& phi, const double& theta, const double& omega) {
    Gate_2q CRot(2,2,2,2);

    const std::complex<double> e00(0, (-phi - omega)/2);
    const std::complex<double> e10(0, (-phi + omega)/2);
    const std::complex<double> e01(0, (phi - omega)/2);
    const std::complex<double> e11(0, (phi + omega)/2);

    const std::complex<double> exp00 = std::pow(M_E, e00);
    const std::complex<double> exp10 = std::pow(M_E, e10);
    const std::complex<double> exp01 = std::pow(M_E, e01);
    const std::complex<double> exp11 = std::pow(M_E, e11);

    const double c = std::cos(theta / 2);
    const double s = std::sin(theta / 2);

    CRot.setValues({{{{1, 0},{0, 0}},{{0, 1},{0, 0}}},{{{0, 0},{exp00 * c, -exp01 * s}},
            {{0, 0},{exp10 * s, exp11 * c}}
        }});
    return CRot;
}


// Creating aliases based on the function signatures of each operation
typedef Gate_1q (*pfunc_1q)();
typedef Gate_1q (*pfunc_1q_one_param)(const double&);
typedef Gate_1q (*pfunc_1q_three_params)(const double&, const double&, const double&);

typedef Gate_2q (*pfunc_2q)();
typedef Gate_2q (*pfunc_2q_one_param)(const double&);
typedef Gate_2q (*pfunc_2q_three_params)(const double&, const double&, const double&);

typedef Gate_3q (*pfunc_3q)();

// Defining the operation maps
const std::map<std::string, pfunc_1q> OneQubitOps = {
    {"Identity", Identity},
    {"PauliX", X},
    {"PauliY", Y},
    {"PauliZ", Z},
    {"Hadamard", H},
    {"S", S},
    {"T", T}
};

const std::map<std::string, pfunc_1q_one_param> OneQubitOpsOneParam = {
    {"RX", RX},
    {"RY", RY},
    {"RZ", RZ},
    {"PhaseShift", PhaseShift}
};

const std::map<std::string, pfunc_1q_three_params> OneQubitOpsThreeParams = {
    {"Rot", Rot}
};


const std::map<std::string, pfunc_2q> TwoQubitOps = {
    {"CNOT", CNOT},
    {"SWAP", SWAP},
    {"CZ", CZ}
};

const std::map<std::string, pfunc_2q_one_param> TwoQubitOpsOneParam = {
    {"CRX", CRX},
    {"CRY", CRY},
    {"CRZ", CRZ}
};

const std::map<std::string, pfunc_2q_three_params> TwoQubitOpsThreeParams = {
    {"CRot", CRot}
};

const std::map<std::string, pfunc_3q> ThreeQubitOps = {
    {"Toffoli", Toffoli},
    {"CSWAP", CSWAP}
};
