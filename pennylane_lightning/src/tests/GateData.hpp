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

#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <type_traits>

using std::vector;

namespace {

template <class DataPrecision = double> class GateUtilities {
    using CplxType = std::complex<DataPrecision>;

  public:
    // Type alias for the functions of parametrized matrices
    using pfunc_params =
        std::function<vector<CplxType>(const vector<DataPrecision> &)>;

    inline static const vector<CplxType> CNOT{
        {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}};

    inline static const vector<CplxType> SWAP{
        {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0},
        {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}};

    inline static const vector<CplxType> CZ{
        {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-1, 0}};

    inline static const vector<CplxType> Toffoli{
        {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}};

    inline static const vector<CplxType> CSWAP{
        {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}};

    inline static const CplxType IMAG{0, 1};
    inline static const CplxType NEGATIVE_IMAG{0, -1};
    inline static const DataPrecision SQRT_2{std::sqrt(2)};
    inline static const DataPrecision INV_SQRT_2{1 / SQRT_2};

    // Non-parametrized single qubit gates
    inline static const vector<CplxType> PauliX{{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    inline static const vector<CplxType> PauliY{
        {0, 0}, NEGATIVE_IMAG, IMAG, {0, 0}};
    inline static const vector<CplxType> PauliZ{
        {1, 0}, {0, 0}, {0, 0}, {-1, 0}};
    inline static const vector<CplxType> Hadamard{
        {INV_SQRT_2, 0}, {INV_SQRT_2, 0}, {INV_SQRT_2, 0}, {-INV_SQRT_2, 0}};
    inline static const vector<CplxType> S{{1, 0}, 0, 0, IMAG};
    inline static const vector<CplxType> T{
        {1, 0}, 0, 0, std::pow(M_E, CplxType{0, M_PI / 4})};

    // Parametrized single qubit gates
    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecisionStatic>>
    RX(const vector<DataPrecisionStatic> &pars) {
        DataPrecisionStatic parameter = pars.at(0);

        const std::complex<DataPrecisionStatic> c{std::cos(parameter / 2), 0};
        const std::complex<DataPrecisionStatic> js{0, std::sin(-parameter / 2)};
        return {c, js, js, c};
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecision>>
    RY(const vector<DataPrecision> &pars) {
        DataPrecision parameter = pars.at(0);

        const std::complex<DataPrecisionStatic> c{std::cos(parameter / 2), 0};
        const std::complex<DataPrecisionStatic> s{std::sin(parameter / 2), 0};
        return {c, -s, s, c};
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecisionStatic>>
    RZ(const vector<double> &pars) {
        DataPrecisionStatic parameter = pars.at(0);
        const std::complex<DataPrecisionStatic> phase{0, -parameter / 2};
        const std::complex<DataPrecisionStatic> phase_second{0, parameter / 2};
        const std::complex<DataPrecisionStatic> first{std::pow(M_E, phase)};
        const std::complex<DataPrecisionStatic> second{
            std::pow(M_E, phase_second)};
        return {first, {0, 0}, {0, 0}, second};
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecisionStatic>>
    PhaseShift(const vector<DataPrecisionStatic> &pars) {
        DataPrecisionStatic parameter = pars.at(0);

        const std::complex<DataPrecisionStatic> phase{0, parameter};
        const std::complex<DataPrecisionStatic> shift{std::pow(M_E, phase)};

        return {{1, 0}, {0, 0}, {0, 0}, shift};
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<DataPrecisionStatic>
    Rot(const vector<DataPrecisionStatic> &pars) {
        const DataPrecisionStatic phi = pars.at(0);
        const DataPrecisionStatic theta = pars.at(1);
        const DataPrecisionStatic omega = pars.at(2);

        const std::complex<DataPrecisionStatic> e00{0, (-phi - omega) / 2};
        const std::complex<DataPrecisionStatic> e10{0, (-phi + omega) / 2};
        const std::complex<DataPrecisionStatic> e01{0, (phi - omega) / 2};
        const std::complex<DataPrecisionStatic> e11{0, (phi + omega) / 2};

        const std::complex<DataPrecisionStatic> exp00{std::pow(M_E, e00)};
        const std::complex<DataPrecisionStatic> exp10{std::pow(M_E, e10)};
        const std::complex<DataPrecisionStatic> exp01{std::pow(M_E, e01)};
        const std::complex<DataPrecisionStatic> exp11{std::pow(M_E, e11)};

        const DataPrecisionStatic c{std::cos(theta / 2)};
        const DataPrecisionStatic s{std::sin(theta / 2)};

        return {exp00 * c, -exp01 * s, exp10 * s, exp11 * c};
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecisionStatic>>
    CRX(const vector<DataPrecisionStatic> &pars) {
        DataPrecisionStatic parameter = pars.at(0);

        const std::complex<DataPrecisionStatic> c{std::cos(parameter / 2), 0};
        const std::complex<DataPrecisionStatic> js{0, std::sin(-parameter / 2)};

        vector<std::complex<DataPrecisionStatic>> matrix{
            {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
            {0, 0}, {0, 0}, c,      js,     {0, 0}, {0, 0}, js,     c};

        return matrix;
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecisionStatic>>
    CRY(const vector<DataPrecisionStatic> &pars) {
        DataPrecision parameter = pars.at(0);

        const DataPrecisionStatic c{std::cos(parameter / 2)};
        const DataPrecisionStatic s{std::sin(parameter / 2)};

        vector<std::complex<DataPrecisionStatic>> matrix = {
            {1, 0}, {0, 0}, {0, 0}, {0, 0},  {0, 0}, {1, 0}, {0, 0}, {0, 0},
            {0, 0}, {0, 0}, {c, 0}, {-s, 0}, {0, 0}, {0, 0}, {s, 0}, {c, 0}};

        return matrix;
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecision>>
    CRZ(const vector<DataPrecision> &pars) {
        DataPrecision parameter = pars.at(0);

        const std::complex<DataPrecisionStatic> phase{0, -parameter / 2};
        const std::complex<DataPrecisionStatic> phase_second{0, parameter / 2};
        const std::complex<DataPrecisionStatic> first = std::pow(M_E, phase);
        const std::complex<DataPrecisionStatic> second =
            std::pow(M_E, phase_second);

        vector<std::complex<DataPrecision>> matrix = {
            {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
            {0, 0}, {0, 0}, first,  {0, 0}, {0, 0}, {0, 0}, {0, 0}, second};
        return matrix;
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecision>>
    CRot(const vector<DataPrecisionStatic> &pars) {
        DataPrecisionStatic phi = pars.at(0);
        DataPrecisionStatic theta = pars.at(1);
        DataPrecisionStatic omega = pars.at(2);

        const std::complex<DataPrecisionStatic> e00(0, (-phi - omega) / 2);
        const std::complex<DataPrecisionStatic> e10(0, (-phi + omega) / 2);
        const std::complex<DataPrecisionStatic> e01(0, (phi - omega) / 2);
        const std::complex<DataPrecisionStatic> e11(0, (phi + omega) / 2);

        const std::complex<DataPrecisionStatic> exp00 = std::pow(M_E, e00);
        const std::complex<DataPrecisionStatic> exp10 = std::pow(M_E, e10);
        const std::complex<DataPrecisionStatic> exp01 = std::pow(M_E, e01);
        const std::complex<DataPrecisionStatic> exp11 = std::pow(M_E, e11);

        const DataPrecisionStatic c{std::cos(theta / 2)};
        const DataPrecisionStatic s{std::sin(theta / 2)};

        vector<std::complex<DataPrecisionStatic>> matrix = {
            {1, 0}, {0, 0}, {0, 0},    {0, 0},   {0, 0},    {1, 0},
            {0, 0}, {0, 0}, {0, 0},    {0, 0},   exp00 * c, -exp01 * s,
            {0, 0}, {0, 0}, exp10 * s, exp11 * c};

        return matrix;
    }
};

} // namespace