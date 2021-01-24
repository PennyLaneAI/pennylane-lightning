// Copyright 2021 Xanadu Quantum Technologies Inc.

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
#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <map>

#include "typedefs.hpp"

const double SQRT_2 = sqrt(2);
const std::complex<double> IMAG(0, 1);
const std::complex<double> NEGATIVE_IMAG(0, -1);

Gate_Xq<1> Identity() {
    Gate_Xq<1> X(2, 2);
    X.setValues({{1, 0}, {0, 1}});
    return X;
}

Gate_Xq<1> X() {
    Gate_Xq<1> X(2, 2);
    X.setValues({{0, 1}, {1, 0}});
    return X;
}

Gate_Xq<1> Y() {
    Gate_Xq<1> Y(2, 2);
    Y.setValues({{0, NEGATIVE_IMAG}, {IMAG, 0}});
    return Y;
}

Gate_Xq<1> Z() {
    Gate_Xq<1> Z(2, 2);
    Z.setValues({{1, 0}, {0, -1}});
    return Z;
}

Gate_Xq<1> H() {
    Gate_Xq<1> H(2, 2);
    H.setValues({{1/SQRT_2, 1/SQRT_2}, {1/SQRT_2, -1/SQRT_2}});
    return H;
}

Gate_Xq<1> S() {
    Gate_Xq<1> S(2, 2);
    S.setValues({{1, 0}, {0, IMAG}});
    return S;
}

Gate_Xq<1> T() {
    Gate_Xq<1> T(2, 2);

    const std::complex<double> exponent(0, M_PI/4);
    T.setValues({{1, 0}, {0, std::pow(M_E, exponent)}});
    return T;
}

Gate_Xq<1> RX(const double& parameter) {
    Gate_Xq<1> RX(2, 2);

    const std::complex<double> c (std::cos(parameter / 2), 0);
    const std::complex<double> js (0, std::sin(-parameter / 2));

    RX.setValues({{c, js}, {js, c}});
    return RX;
}

Gate_Xq<1> RY(const double& parameter) {
    Gate_Xq<1> RY(2, 2);

    const double c = std::cos(parameter / 2);
    const double s = std::sin(parameter / 2);

    RY.setValues({{c, -s}, {s, c}});
    return RY;
}

Gate_Xq<1> RZ(const double& parameter) {
    Gate_Xq<1> RZ(2, 2);

    const std::complex<double> exponent(0, -parameter/2);
    const std::complex<double> exponent_second(0, parameter/2);
    const std::complex<double> first = std::pow(M_E, exponent);
    const std::complex<double> second = std::pow(M_E, exponent_second);

    RZ.setValues({{first, 0}, {0, second}});
    return RZ;
}

Gate_Xq<1> PhaseShift(const double& parameter) {
    Gate_Xq<1> PhaseShift(2, 2);

    const std::complex<double> exponent(0, parameter);
    const std::complex<double> shift = std::pow(M_E, exponent);

    PhaseShift.setValues({{1, 0}, {0, shift}});
    return PhaseShift;
}

Gate_Xq<1> Rot(const double& phi, const double& theta, const double& omega) {
    Gate_Xq<1> Rot(2, 2);

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

Gate_Xq<2> CNOT() {
    Gate_Xq<2> CNOT(2,2,2,2);
    CNOT.setValues({{{{1, 0},{0, 0}},{{0, 1},{0, 0}}},{{{0, 0},{0, 1}},{{0, 0},{1, 0}}}});
    return CNOT;
}

Gate_Xq<2> SWAP() {
    Gate_Xq<2> SWAP(2,2,2,2);
    SWAP.setValues({{{{1, 0},{0, 0}},{{0, 0},{1, 0}}},{{{0, 1},{0, 0}},{{0, 0},{0, 1}}}});
    return SWAP;
}

Gate_Xq<2> CZ() {
    Gate_Xq<2> CZ(2,2,2,2);
    CZ.setValues({{{{1, 0},{0, 0}},{{0, 1},{0, 0}}},{{{0, 0},{1, 0}},{{0, 0},{0, -1}}}});
    return CZ;
}

Gate_Xq<3> Toffoli() {
    Gate_Xq<3> Toffoli(2,2,2,2,2,2);
    Toffoli.setValues({{{{{{1, 0},{0, 0}},{{0, 0},{0, 0}}},{{{0, 1},{0, 0}},{{0, 0},{0, 0}}}},
            {{{{0, 0},{1, 0}},{{0, 0},{0, 0}}},{{{0, 0},{0, 1}},{{0, 0},{0, 0}}}}
        },
        {   {{{{0, 0},{0, 0}},{{1, 0},{0, 0}}},{{{0, 0},{0, 0}},{{0, 1},{0, 0}}}},
            {{{{0, 0},{0, 0}},{{0, 0},{0, 1}}},{{{0, 0},{0, 0}},{{0, 0},{1, 0}}}}
        }});
    return Toffoli;
}

Gate_Xq<3> CSWAP() {
    Gate_Xq<3> CSWAP(2,2,2,2,2,2);
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
Gate_Xq<2> CRX(const double& parameter) {
    Gate_Xq<2> CRX(2, 2, 2, 2);

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
Gate_Xq<2> CRY(const double& parameter) {
    Gate_Xq<2> CRY(2, 2, 2, 2);

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
Gate_Xq<2> CRZ(const double& parameter) {
    Gate_Xq<2> CRZ(2, 2, 2, 2);

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
Gate_Xq<2> CRot(const double& phi, const double& theta, const double& omega) {
    Gate_Xq<2> CRot(2,2,2,2);

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

// Defining the operation maps
const std::map<std::string, pfunc_Xq<1>> OneQubitOps = {
    {"Identity", Identity},
    {"PauliX", X},
    {"PauliY", Y},
    {"PauliZ", Z},
    {"Hadamard", H},
    {"S", S},
    {"T", T}
};

const std::map<std::string, pfunc_Xq_one_param<1>> OneQubitOpsOneParam = {
    {"RX", RX},
    {"RY", RY},
    {"RZ", RZ},
    {"PhaseShift", PhaseShift}
};

const std::map<std::string, pfunc_Xq_three_params<1>> OneQubitOpsThreeParams = {
    {"Rot", Rot}
};


const std::map<std::string, pfunc_Xq<2>> TwoQubitOps = {
    {"CNOT", CNOT},
    {"SWAP", SWAP},
    {"CZ", CZ}
};

const std::map<std::string, pfunc_Xq_one_param<2>> TwoQubitOpsOneParam = {
    {"CRX", CRX},
    {"CRY", CRY},
    {"CRZ", CRZ}
};

const std::map<std::string, pfunc_Xq_three_params<2>> TwoQubitOpsThreeParams = {
    {"CRot", CRot}
};

const std::map<std::string, pfunc_Xq<3>> ThreeQubitOps = {
    {"Toffoli", Toffoli},
    {"CSWAP", CSWAP}
};
