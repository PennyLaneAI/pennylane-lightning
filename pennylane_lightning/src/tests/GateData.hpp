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
#include <functional>
#include <iostream>
#include <type_traits>

#include "../typedefs.hpp"

// using Pennylane::CplxType;
using std::vector;

namespace {

template <class DataPrecision = double> class GateUtilities {
    using CplxType = std::complex<DataPrecision>;

  public:
    // Type alias for the functions of parametrized matrices
    using pfunc_params =
        std::function<vector<CplxType>(const vector<DataPrecision> &)>;

    constexpr vector<CplxType> CNOT{
        {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}};

    constexpr vector<CplxType> SWAP{
        {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0},
        {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}};

    constexpr vector<CplxType> CZ{
        {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {-1, 0}};

    constexpr vector<CplxType> Toffoli{
        {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}};

    constexpr vector<CplxType> CSWAP{
        {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
        {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}};

    constexpr CplxType IMAG{0, 1};
    constexpr CplxType NEGATIVE_IMAG{0, -1};
    constexpr DataPrecision SQRT_2{Sqrt(2)};
    constexpr DataPrecision INV_SQRT_2{1 / SQRT_2};

    // Non-parametrized single qubit gates
    constexpr vector<CplxType> PauliX{{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    constexpr vector<CplxType> PauliY{{0, 0}, NEGATIVE_IMAG, IMAG, {0, 0}};
    constexpr vector<CplxType> PauliZ{{1, 0}, {0, 0}, {0, 0}, {-1, 0}};
    constexpr vector<CplxType> Hadamard{
        {INV_SQRT_2, 0}, {INV_SQRT_2, 0}, {INV_SQRT_2, 0}, {-INV_SQRT_2, 0}};
    constexpr vector<CplxType> S{{1, 0}, 0, 0, IMAG};
    constexpr vector<CplxType> T{{1, 0}, 0, 0, Pow(M_E, CplxType{0, M_PI / 4})};

    template <class DataPrecisionStatic = DataPrecision>
    static DataPrecisionStatic Pow(DataPrecisionStatic base,
                                   DataPrecisionStatic expon) {
        if constexpr (std::is_same_v<DataPrecisionStatic, double>)
            return std::pow(base, expon);
        else
            return std::powf(base, expon);
    }
    template <class DataPrecisionStatic = DataPrecision>
    static DataPrecisionStatic Cos(DataPrecisionStatic value) {
        if constexpr (std::is_same_v<DataPrecisionStatic, double>)
            return std::cos(value);
        else
            return std::cosf(value);
    }
    template <class DataPrecisionStatic = DataPrecision>
    static DataPrecisionStatic Sin(DataPrecisionStatic value) {
        if constexpr (std::is_same_v<DataPrecisionStatic, double>)
            return std::sin(value);
        else
            return std::sinf(value);
    }
    template <class DataPrecisionStatic = DataPrecision>
    static DataPrecisionStatic Sqrt(DataPrecisionStatic value) {
        if constexpr (std::is_same_v<DataPrecisionStatic, double>)
            return std::sqrt(value);
        else
            return std::sqrtf(value);
    }

    // Parametrized single qubit gates
    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecisionStatic>>
    RX(const vector<DataPrecisionStatic> &pars) {
        DataPrecisionStatic parameter = pars.at(0);

        const std::complex<DataPrecisionStatic> c{
            Cos<DataPrecisionStatic>(parameter / 2), 0};
        const std::complex<DataPrecisionStatic> js{
            0, Sin<DataPrecisionStatic>(-parameter / 2)};
        return {c, js, js, c};
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecision>>
    RY(const vector<DataPrecision> &pars) {
        DataPrecision parameter = pars.at(0);

        const std::complex<DataPrecisionStatic> c{
            Cos<DataPrecisionStatic>(parameter / 2), 0};
        const std::complex<DataPrecisionStatic> s{
            Sin<DataPrecisionStatic>(parameter / 2), 0};
        return {c, -s, s, c};
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecisionStatic>>
    RZ(const vector<double> &pars) {
        DataPrecisionStatic parameter = pars.at(0);
        const std::complex<DataPrecisionStatic> phase{0, -parameter / 2};
        const std::complex<DataPrecisionStatic> phase_second{0, parameter / 2};
        const std::complex<DataPrecisionStatic> first{
            Pow<DataPrecisionStatic>(M_E, phase)};
        const std::complex<DataPrecisionStatic> second{
            Pow<DataPrecisionStatic>(M_E, phase_second)};
        return {first, {0, 0}, {0, 0}, second};
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecisionStatic>>
    PhaseShift(const vector<DataPrecisionStatic> &pars) {
        DataPrecisionStatic parameter = pars.at(0);

        const std::complex<DataPrecisionStatic> phase{0, parameter};
        const std::complex<DataPrecisionStatic> shift{
            Pow<DataPrecisionStatic>(M_E, phase)};

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

        const std::complex<DataPrecisionStatic> exp00{
            Pow<DataPrecisionStatic>(M_E, e00)};
        const std::complex<DataPrecisionStatic> exp10{
            Pow<DataPrecisionStatic>(M_E, e10)};
        const std::complex<DataPrecisionStatic> exp01{
            Pow<DataPrecisionStatic>(M_E, e01)};
        const std::complex<DataPrecisionStatic> exp11{
            Pow<DataPrecisionStatic>(M_E, e11)};

        const DataPrecisionStatic c{Cos<DataPrecisionStatic>(theta / 2)};
        const DataPrecisionStatic s{Sin<DataPrecisionStatic>(theta / 2)};

        return {exp00 * c, -exp01 * s, exp10 * s, exp11 * c};
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecisionStatic>>
    CRX(const vector<DataPrecisionStatic> &pars) {
        DataPrecisionStatic parameter = pars.at(0);

        const std::complex<DataPrecisionStatic> c{
            Cos<DataPrecisionStatic>(parameter / 2), 0};
        const std::complex<DataPrecisionStatic> js{
            0, Sin<DataPrecisionStatic>(-parameter / 2)};

        vector<std::complex<DataPrecisionStatic>> matrix{
            {1, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0},
            {0, 0}, {0, 0}, c,      js,     {0, 0}, {0, 0}, js,     c};

        return matrix;
    }

    template <class DataPrecisionStatic = DataPrecision>
    static vector<std::complex<DataPrecisionStatic>>
    CRY(const vector<DataPrecisionStatic> &pars) {
        DataPrecision parameter = pars.at(0);

        const DataPrecisionStatic c{Cos<DataPrecisionStatic>(parameter / 2)};
        const DataPrecisionStatic s{Sin<DataPrecisionStatic>(parameter / 2)};

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
        const std::complex<DataPrecisionStatic> first =
            Pow<DataPrecisionStatic>(M_E, phase);
        const std::complex<DataPrecisionStatic> second =
            Pow<DataPrecisionStatic>(M_E, phase_second);

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

        const std::complex<DataPrecisionStatic> exp00 =
            Pow<DataPrecisionStatic>(M_E, e00);
        const std::complex<DataPrecisionStatic> exp10 =
            Pow<DataPrecisionStatic>(M_E, e10);
        const std::complex<DataPrecisionStatic> exp01 =
            Pow<DataPrecisionStatic>(M_E, e01);
        const std::complex<DataPrecisionStatic> exp11 =
            Pow<DataPrecisionStatic>(M_E, e11);

        const DataPrecisionStatic c{Cos<DataPrecisionStatic>(theta / 2)};
        const DataPrecisionStatic s{Sin<DataPrecisionStatic>(theta / 2)};

        vector<std::complex<DataPrecisionStatic>> matrix = {
            {1, 0}, {0, 0}, {0, 0},    {0, 0},   {0, 0},    {1, 0},
            {0, 0}, {0, 0}, {0, 0},    {0, 0},   exp00 * c, -exp01 * s,
            {0, 0}, {0, 0}, exp10 * s, exp11 * c};

        return matrix;
    }
};

} // namespace