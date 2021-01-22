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
#include <map>

#include "typedefs.hpp"

const double SQRT_2 = sqrt(2);
const std::complex<double> IMAG(0, 1);
const std::complex<double> NEGATIVE_IMAG(0, -1);

/**
* Generates the identity gate.
*
* @return the identity tensor
*/
Gate_Xq<1> Identity();

/**
* Generates the X gate.
*
* @return the X tensor
*/
Gate_Xq<1> X();

/**
* Generates the Y gate.
*
* @return the Y tensor
*/
Gate_Xq<1> Y();

/**
* Generates the Z gate.
*
* @return the Z tensor
*/
Gate_Xq<1> Z();

/**
* Generates the H gate.
*
* @return the H tensor
*/
Gate_Xq<1> H();

/**
* Generates the S gate.
*
* @return the S tensor
*/
Gate_Xq<1> S();

/**
* Generates the T gate.
*
* @return the T tensor
*/
Gate_Xq<1> T();

/**
* Generates the X rotation gate.
*
* @param parameter the rotation angle
* @return the RX tensor
*/
Gate_Xq<1> RX(const double& parameter);

/**
* Generates the Y rotation gate.
*
* @param parameter the rotation angle
* @return the RY tensor
*/
Gate_Xq<1> RY(const double& parameter);

/**
* Generates the Z rotation gate.
*
* @param parameter the rotation angle
* @return the RZ tensor
*/
Gate_Xq<1> RZ(const double& parameter);

/**
* Generates the phase-shift gate.
*
* @param parameter the phase shift
* @return the phase-shift tensor
*/
Gate_Xq<1> PhaseShift(const double& parameter);

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
Gate_Xq<1> Rot(const double& phi, const double& theta, const double& omega);

/**
* Generates the CNOT gate.
*
* @return the CNOT tensor
*/
Gate_Xq<2> CNOT();

/**
* Generates the SWAP gate.
*
* @return the SWAP tensor
*/
Gate_Xq<2> SWAP();

/**
* Generates the CZ gate.
*
* @return the CZ tensor
*/
Gate_Xq<2> CZ();

/**
* Generates the Toffoli gate.
*
* @return the Toffoli tensor
*/
Gate_Xq<3> Toffoli();

/**
* Generates the CSWAP gate.
*
* @return the CSWAP tensor
*/
Gate_Xq<3> CSWAP();

/**
* Generates the controlled-X rotation gate.
*
* @param parameter the rotation angle
* @return the CRX tensor
*/
Gate_Xq<2> CRX(const double& parameter);

/**
* Generates the controlled-Y rotation gate.
*
* @param parameter the rotation angle
* @return the CRY tensor
*/
Gate_Xq<2> CRY(const double& parameter);

/**
* Generates the controlled-Z rotation gate.
*
* @param parameter the rotation angle
* @return the CRZ tensor
*/
Gate_Xq<2> CRZ(const double& parameter);

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
Gate_Xq<2> CRot(const double& phi, const double& theta, const double& omega);

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
