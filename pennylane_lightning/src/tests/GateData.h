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
#pragma once
#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include "../rework/typedefs.hpp"


using Pennylane::CplxType;
using std::vector;

// Useful constants
const std::complex<double> IMAG(0, 1);
const std::complex<double> NEGATIVE_IMAG(0, -1);
const double SQRT_2 = sqrt(2);
const std::complex<double> exponent(0, M_PI/4);

// Non-parametrized single qubit gates
static const vector<CplxType> PauliX = {0,1,1,0};
static const vector<CplxType> PauliY = {0, NEGATIVE_IMAG, IMAG, 0};
static const vector<CplxType> PauliZ = {1,0,0,-1};
static const vector<CplxType> Hadamard = {1/SQRT_2, 1/SQRT_2, 1/SQRT_2, -1/SQRT_2};
static const vector<CplxType> S = {1, 0, 0, IMAG};
static const vector<CplxType> T = {1, 0, 0, std::pow(M_E, exponent)};


// Parametrized single qubit gates
vector<CplxType> RX(double parameter){
    const std::complex<double> c (std::cos(parameter / 2), 0);
    const std::complex<double> js (0, std::sin(-parameter / 2));
    return {c, js, js, c};
}

vector<CplxType> RY(double parameter){
    const double c = std::cos(parameter / 2);
    const double s = std::sin(parameter / 2);
    return {c, -s, s, c};
}

vector<CplxType> RZ(double parameter){
    const std::complex<double> exponent(0, -parameter/2);
    const std::complex<double> exponent_second(0, parameter/2);
    const std::complex<double> first = std::pow(M_E, exponent);
    const std::complex<double> second = std::pow(M_E, exponent_second);
    return {first, 0, 0, second};
}

static const vector<CplxType> CNOT = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0};

static const vector<CplxType> Toffoli = {
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1, 0 };

