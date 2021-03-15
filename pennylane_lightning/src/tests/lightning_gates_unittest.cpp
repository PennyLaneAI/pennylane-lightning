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
#include "gtest/gtest.h"
#include "../Gates.hpp"
#include "GateData.hpp"
#include "TestingUtils.hpp"

#include <functional>

using std::unique_ptr;
using std::vector;
using std::string;
using std::function;

using Pennylane::CplxType;

namespace test_gates{

const vector<double> ZERO_PARAM = {};
const vector<double> ONE_PARAM = {0.123};
const vector<double> THREE_PARAMS = {0.123, 2.345, 1.4321};


// -------------------------------------------------------------------------------------------------------------
// Non-parametrized gates

class MatrixNoParamFixture : public ::testing::TestWithParam<std::tuple<string, vector<CplxType> > > {
};

TEST_P(MatrixNoParamFixture, CheckMatrix) {
    const string gate_name = std::get<0>(GetParam());
    const vector<CplxType> matrix = std::get<1>(GetParam());

    const vector<double> params = {};

    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, params);
    EXPECT_EQ(gate->asMatrix(), matrix);
}

INSTANTIATE_TEST_SUITE_P (
        GateMatrix,
        MatrixNoParamFixture,
        ::testing::Values(
                std::make_tuple("PauliX", PauliX),
                std::make_tuple("PauliY", PauliY),
                std::make_tuple("PauliZ", PauliZ),
                std::make_tuple("Hadamard", Hadamard),
                std::make_tuple("S", S),
                std::make_tuple("T", T),
                std::make_tuple("CNOT", CNOT),
                std::make_tuple("SWAP", SWAP),
                std::make_tuple("CZ", CZ),
                std::make_tuple("Toffoli", Toffoli),
                std::make_tuple("CSWAP", CSWAP)
));

// -------------------------------------------------------------------------------------------------------------
// Parametrized gates

class MatrixWithParamsFixture : public ::testing::TestWithParam<std::tuple<string, pfunc_params, vector<double> >> {
};

TEST_P(MatrixWithParamsFixture, CheckMatrix) {
    const string gate_name = std::get<0>(GetParam());
    pfunc_params func = std::get<1>(GetParam());
    const vector<double> params = std::get<2>(GetParam());


    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, params);
    EXPECT_EQ(gate->asMatrix(), func(params));
}

INSTANTIATE_TEST_SUITE_P (
        GateMatrix,
        MatrixWithParamsFixture,
        ::testing::Values(
                std::make_tuple("RX", RX, ONE_PARAM),
                std::make_tuple("RY", RY, ONE_PARAM),
                std::make_tuple("RZ", RZ, ONE_PARAM),
                std::make_tuple("PhaseShift", PhaseShift, ONE_PARAM),
                std::make_tuple("Rot", Rot, THREE_PARAMS),
                std::make_tuple("CRX", CRX, ONE_PARAM),
                std::make_tuple("CRY", CRY, ONE_PARAM),
                std::make_tuple("CRZ", CRZ, ONE_PARAM),
                std::make_tuple("CRot", CRot, THREE_PARAMS)
));


// -------------------------------------------------------------------------------------------------------------
// Parameter length validation

class NumParamsThrowsFixture : public ::testing::TestWithParam<std::tuple<string, vector<double> > > {
};

TEST_P(NumParamsThrowsFixture, CheckParamLength) {
    const string gate_name = std::get<0>(GetParam());
    const vector<double> params = std::get<1>(GetParam());

    EXPECT_THROW_WITH_MESSAGE_SUBSTRING(Pennylane::constructGate(gate_name, params), std::invalid_argument, gate_name);
}

const vector<string> non_param_gates = {"PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "CNOT", "SWAP", "CZ", "Toffoli", "CSWAP"};
const vector<vector<double>> many_params = {ONE_PARAM, THREE_PARAMS};

INSTANTIATE_TEST_SUITE_P (
        NoParameterGateChecks,
        NumParamsThrowsFixture,
        ::testing::Combine(
            ::testing::ValuesIn(non_param_gates),
            ::testing::ValuesIn(many_params)
    ));

const vector<string> param_gates = {"RX", "RY", "RZ", "PhaseShift", "Rot", "CRX", "CRY", "CRZ", "CRot"};
const vector<vector<double>> zero_params = { ZERO_PARAM };

INSTANTIATE_TEST_SUITE_P (
        ParametrizedGateChecks,
        NumParamsThrowsFixture,
        ::testing::Combine(
            ::testing::ValuesIn(param_gates),
            ::testing::ValuesIn(zero_params)
    ));

TEST(DispatchTable, constructGateThrows) {
    const string test_gate_name = "Non-existent gate";
    EXPECT_THROW_WITH_MESSAGE_SUBSTRING(Pennylane::constructGate(test_gate_name, {}), std::invalid_argument, test_gate_name);
}

TEST(QubitUnitary, constructGate) {
    vector<CplxType> mx = {1,0,0,1};
    Pennylane::QubitUnitary qubit_unitary(1, mx);
    ASSERT_EQ(qubit_unitary.numQubits, 1);
    ASSERT_EQ(qubit_unitary.asMatrix(), mx);
}


}
