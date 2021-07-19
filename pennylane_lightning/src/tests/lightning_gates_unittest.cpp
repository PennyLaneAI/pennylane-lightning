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
#include <functional>

#include "../StateVector.hpp"

#include "GateData.hpp"
#include "TestingUtils.hpp"
#include "gtest/gtest.h"

using std::function;
using std::string;
using std::unique_ptr;
using std::vector;

namespace test_gates {

using PrecisionT = double;
using CplxType = std::complex<PrecisionT>;

const vector<PrecisionT> ZERO_PARAM = {};
const vector<PrecisionT> ONE_PARAM = {0.123};
const vector<PrecisionT> THREE_PARAMS = {0.123, 2.345, 1.4321};

// -------------------------------------------------------------------------------------------------------------
// Non-parametrized gates

class MatrixNoParamFixture
    : public ::testing::TestWithParam<
          std::tuple<vector<CplxType>, vector<CplxType>>> {};

TEST_P(MatrixNoParamFixture, CheckMatrix) {
    const vector<CplxType> have_gate = std::get<0>(GetParam());
    const vector<CplxType> want_gate = std::get<1>(GetParam());

    const vector<double> params = {};

    EXPECT_EQ(have_gate, want_gate);
}

INSTANTIATE_TEST_SUITE_P(
    GateMatrix, MatrixNoParamFixture,
    ::testing::Values(
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getPauliX(),
                        GateUtilities<PrecisionT>::PauliX),
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getPauliY(),
                        GateUtilities<PrecisionT>::PauliY),
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getPauliZ(),
                        GateUtilities<PrecisionT>::PauliZ),
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getHadamard(),
                        GateUtilities<PrecisionT>::Hadamard),
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getS(),
                        GateUtilities<PrecisionT>::S),
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getT(),
                        GateUtilities<PrecisionT>::T),
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getCNOT(),
                        GateUtilities<PrecisionT>::CNOT),
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getSWAP(),
                        GateUtilities<PrecisionT>::SWAP),
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getCZ(),
                        GateUtilities<PrecisionT>::CZ),
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getToffoli(),
                        GateUtilities<PrecisionT>::Toffoli),
        std::make_tuple(Pennylane::StateVector<PrecisionT>::getCSWAP(),
                        GateUtilities<PrecisionT>::CSWAP)));

// -------------------------------------------------------------------------------------------------------------
// Parametrized gates

class MatrixWithParamsFixture
    : public ::testing::TestWithParam<
          std::tuple<GateUtilities<PrecisionT>::pfunc_params, vector<CplxType>,
                     vector<PrecisionT>>> {};

TEST_P(MatrixWithParamsFixture, CheckMatrix) {

    const GateUtilities<PrecisionT>::pfunc_params have_gate_func =
        std::get<0>(GetParam());
    const vector<CplxType> want_gate = std::get<1>(GetParam());
    const vector<double> gate_params = std::get<2>(GetParam());

    EXPECT_EQ(have_gate_func(gate_params), want_gate);
}

INSTANTIATE_TEST_SUITE_P(
    GateMatrix, MatrixWithParamsFixture,
    ::testing::Values(
        std::make_tuple(&Pennylane::StateVector<PrecisionT>::getRX,
                        GateUtilities<PrecisionT>::RX, ONE_PARAM),
        std::make_tuple(&Pennylane::StateVector<PrecisionT>::getRY,
                        GateUtilities<PrecisionT>::RY, ONE_PARAM),
        std::make_tuple(&Pennylane::StateVector<PrecisionT>::getRZ,
                        GateUtilities<PrecisionT>::RZ, ONE_PARAM),
        std::make_tuple(&Pennylane::StateVector<PrecisionT>::getPhaseShift,
                        PhaseShift, ONE_PARAM),
        std::make_tuple(&Pennylane::StateVector<PrecisionT>::getRot,
                        GateUtilities<PrecisionT>::Rot, THREE_PARAMS),
        std::make_tuple(&Pennylane::StateVector<PrecisionT>::getCRX,
                        GateUtilities<PrecisionT>::CRX, ONE_PARAM),
        std::make_tuple(&Pennylane::StateVector<PrecisionT>::getCRY,
                        GateUtilities<PrecisionT>::CRY, ONE_PARAM),
        std::make_tuple(&Pennylane::StateVector<PrecisionT>::getCRZ,
                        GateUtilities<PrecisionT>::CRZ, ONE_PARAM),
        std::make_tuple(&Pennylane::StateVector<PrecisionT>::getCRot,
                        GateUtilities<PrecisionT>::CRot, THREE_PARAMS)));

// -------------------------------------------------------------------------------------------------------------
// Parameter length validation

class NumParamsThrowsFixture
    : public ::testing::TestWithParam<std::tuple<string, vector<PrecisionT>>> {
};

TEST_P(NumParamsThrowsFixture, CheckParamLength) {
    const string gate_name = std::get<0>(GetParam());
    const vector<PrecisionT> params = std::get<1>(GetParam());
}

const vector<string> non_param_gates = {
    "PauliX", "PauliY", "PauliZ", "Hadamard", "S",    "T",
    "CNOT",   "SWAP",   "CZ",     "Toffoli",  "CSWAP"};
const vector<vector<double>> many_params = {ONE_PARAM, THREE_PARAMS};

INSTANTIATE_TEST_SUITE_P(
    NoParameterGateChecks, NumParamsThrowsFixture,
    ::testing::Combine(::testing::ValuesIn(non_param_gates),
                       ::testing::ValuesIn(many_params)));

const vector<string> param_gates = {"RX",  "RY",  "RZ",  "PhaseShift", "Rot",
                                    "CRX", "CRY", "CRZ", "CRot"};
const vector<vector<double>> zero_params = {ZERO_PARAM};

INSTANTIATE_TEST_SUITE_P(ParametrizedGateChecks, NumParamsThrowsFixture,
                         ::testing::Combine(::testing::ValuesIn(param_gates),
                                            ::testing::ValuesIn(zero_params)));

} // namespace test_gates
