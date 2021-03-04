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
vector<CplxType> RX(const vector<double> & pars){
    double parameter = pars.at(0);

    const std::complex<double> c (std::cos(parameter / 2), 0);
    const std::complex<double> js (0, std::sin(-parameter / 2));
    return {c, js, js, c};
}

vector<CplxType> RY(const vector<double> & pars){
    double parameter = pars.at(0);

    const double c = std::cos(parameter / 2);
    const double s = std::sin(parameter / 2);
    return {c, -s, s, c};
}

vector<CplxType> RZ(const vector<double> & pars){

    double parameter = pars.at(0);
    const std::complex<double> exponent(0, -parameter/2);
    const std::complex<double> exponent_second(0, parameter/2);
    const std::complex<double> first = std::pow(M_E, exponent);
    const std::complex<double> second = std::pow(M_E, exponent_second);
    return {first, 0, 0, second};
}

vector<CplxType> PhaseShift(const vector<double> & pars){

    double parameter = pars.at(0);

    const std::complex<double> exponent(0, parameter);
    const std::complex<double> shift = std::pow(M_E, exponent);

    return {1, 0, 0, shift};
}

vector<CplxType> Rot(const vector<double> & pars){

    double phi = pars.at(0);
    double theta = pars.at(1);
    double omega = pars.at(2);

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

    return {exp00 * c, -exp01 * s, exp10 * s, exp11 * c};
}


// Defining the operation maps
using pfunc_params = std::function<vector<CplxType>(const vector<double>&)>;


static const vector<CplxType> CNOT = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
    0, 0, 1, 0};

static const vector<CplxType> SWAP = {
    1, 0, 0, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    0, 0, 0, 1};

static const vector<CplxType> CZ = {
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, -1};

static const vector<CplxType> Toffoli = {
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 1, 0 };

static const vector<CplxType> CSWAP = {
    1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1 };

