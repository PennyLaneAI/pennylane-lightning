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
#include "../rework/Gates.hpp"
#include "GateData.h"
#include "TestingUtils.h"

#include <functional>

using std::unique_ptr;
using std::vector;
using std::string;
using std::function;

using Pennylane::CplxType;

using getParametrizedGateMatrix = function< vector<CplxType>(double)>;

namespace test_gates{

// -------------------------------------------------------------------------------------------------------------
// Non-parametrized gates
class GateMatrixNoParamTestFixture : public ::testing::TestWithParam<std::tuple<string, vector<CplxType> > > {
};

TEST_P(GateMatrixNoParamTestFixture, CheckMatrix) {
    const string gate_name = std::get<0>(GetParam());
    const vector<CplxType> matrix = std::get<1>(GetParam());

    const vector<double> params = {};

    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, params);
    EXPECT_EQ(gate->asMatrix(), matrix);
}

INSTANTIATE_TEST_SUITE_P (
        GateTests,
        GateMatrixNoParamTestFixture,
        ::testing::Values(
                std::make_tuple("PauliX", PauliX),
                std::make_tuple("PauliY", PauliY),
                std::make_tuple("PauliZ", PauliZ),
                std::make_tuple("Hadamard", Hadamard),
                std::make_tuple("S", S),
                std::make_tuple("T", T),
                std::make_tuple("CNOT", CNOT),
                std::make_tuple("Toffoli", Toffoli)));

// -------------------------------------------------------------------------------------------------------------
// Parametrized gates

/*
class GateMatrixWithParamsTestFixture : public ::testing::TestWithParam<std::tuple<string, getParametrizedGateMatrix, vector<double> > {
};

TEST_P(GateMatrixWithParamsTestFixture, CheckMatrix) {
    const string gate_name = std::get<0>(GetParam());
    auto func = std::get<1>(GetParam());
    const vector<double> params = std::get<2>(GetParam());

    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, params);
    EXPECT_EQ(gate->asMatrix(), func(params[0]));
}

INSTANTIATE_TEST_SUITE_P (
        GateTests,
        GateMatrixWithParamsTestFixture,
        ::testing::Values(
                std::make_tuple("RX", RX),
));
*/

TEST(constructGate, RX){
    const string gate_name = "RX";
    vector<double> params = {0.3};

    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, params);

    EXPECT_EQ(gate->asMatrix(), RX(params.at(0)));
}

TEST(constructGate, RY){
    const string gate_name = "RY";
    vector<double> params = {0.3};

    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, params);

    EXPECT_EQ(gate->asMatrix(), RY(params.at(0)));
}

TEST(constructGate, RZ){
    const string gate_name = "RZ";
    vector<double> params = {0.3};

    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, params);

    EXPECT_EQ(gate->asMatrix(), RZ(params.at(0)));
}


// -------------------------------------------------------------------------------------------------------------
// Parameter length validation

// Pair the gate name with its matrix from GateData
class ValidateParamLengthGateFixture : public ::testing::TestWithParam<std::tuple<string, vector<double> > > {
};

TEST_P(ValidateParamLengthGateFixture, CheckParamLength) {
    const string gate_name = std::get<0>(GetParam());
    const vector<double> params = std::get<1>(GetParam());
    //EXPECT_THROW(Pennylane::constructGate(gate_name, params), std::invalid_argument);
    EXPECT_THROW_WITH_MESSAGE_SUBSTRING(Pennylane::constructGate(gate_name, params), std::invalid_argument, gate_name);
}

const vector<double> ZERO_PARAM = {};
const vector<double> ONE_PARAM = {0.123};
const vector<double> TWO_PARAMS = {0.123, 2.345};

INSTANTIATE_TEST_SUITE_P (
        ValidationTests,
        ValidateParamLengthGateFixture,
        ::testing::Values(
                std::make_tuple("PauliX", ONE_PARAM),
                std::make_tuple("PauliX", TWO_PARAMS),
                std::make_tuple("PauliY", ONE_PARAM),
                std::make_tuple("PauliY", TWO_PARAMS),
                std::make_tuple("PauliZ", ONE_PARAM),
                std::make_tuple("PauliZ", TWO_PARAMS),

                std::make_tuple("CNOT", ONE_PARAM),
                std::make_tuple("CNOT", TWO_PARAMS),
                std::make_tuple("Toffoli", ONE_PARAM),
                std::make_tuple("Toffoli", TWO_PARAMS)));

}
