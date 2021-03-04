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

using getParametrizedMatrix = function< vector<CplxType>(double)>;

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
                std::make_tuple("Toffoli", Toffoli)));

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
                std::make_tuple("Rot", Rot, THREE_PARAMS)
));


// -------------------------------------------------------------------------------------------------------------
// Parameter length validation

// Pair the gate name with its matrix from GateData
class NumParamsThrowsFixture : public ::testing::TestWithParam<std::tuple<string, vector<double> > > {
};

TEST_P(NumParamsThrowsFixture, CheckParamLength) {
    const string gate_name = std::get<0>(GetParam());
    const vector<double> params = std::get<1>(GetParam());
    //EXPECT_THROW(Pennylane::constructGate(gate_name, params), std::invalid_argument);
    EXPECT_THROW_WITH_MESSAGE_SUBSTRING(Pennylane::constructGate(gate_name, params), std::invalid_argument, gate_name);
}

INSTANTIATE_TEST_SUITE_P (
        GateChecks,
        NumParamsThrowsFixture,
        ::testing::Values(
                std::make_tuple("PauliX", ONE_PARAM),
                std::make_tuple("PauliX", THREE_PARAMS),
                std::make_tuple("PauliY", ONE_PARAM),
                std::make_tuple("PauliY", THREE_PARAMS),
                std::make_tuple("PauliZ", ONE_PARAM),
                std::make_tuple("PauliZ", THREE_PARAMS),

                std::make_tuple("CNOT", ONE_PARAM),
                std::make_tuple("CNOT", THREE_PARAMS),
                std::make_tuple("Toffoli", ONE_PARAM),
                std::make_tuple("Toffoli", THREE_PARAMS)));

}
