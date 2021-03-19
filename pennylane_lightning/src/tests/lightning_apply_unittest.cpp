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
#include "../Apply.hpp"
#include "GateData.hpp"

using std::vector;
using std::string;
using std::unique_ptr;

const vector<double> ONE_PARAM = {0.123};

namespace test_apply {

class getIndicesAfterExclusionTestFixture :public ::testing::TestWithParam<std::tuple<unsigned int, unsigned int>> {
};

TEST_P(getIndicesAfterExclusionTestFixture, CheckgetIndicesAfterExclusionResults) {
    unsigned int index = std::get<0>(GetParam());
    unsigned int qubits = std::get<1>(GetParam());

    vector<vector<unsigned int>> inputs = {
        {0, 1},
        {2},
        {1, 2},
        {1, 2, 4},
        {3, 2, 4},
    };

    vector<vector<unsigned int>> outputs = {
        {2},
        {0, 1},
        {0},
        {0, 3},
        {0, 1, 5},
    };

    vector<unsigned int> input = inputs[index];
    vector<unsigned int> output = outputs[index];
    ASSERT_EQ(output, Pennylane::getIndicesAfterExclusion(input, qubits));
}

INSTANTIATE_TEST_SUITE_P (
        getIndicesAfterExclusionTests,
        getIndicesAfterExclusionTestFixture,
        ::testing::Values(
                std::make_tuple(0, 3),
                std::make_tuple(1, 3),
                std::make_tuple(2, 3),
                std::make_tuple(3, 5),
                std::make_tuple(4, 6)
                ));

}

// -------------------------------------------------------------------------------------------------------------
// Test applyDerivative function

class applyDerivativeFixture : public ::testing::TestWithParam<std::tuple<string, pfunc_params, vector<double> >> {
};

TEST_P(applyDerivativeFixture, CheckApplyDerivative) {
    const string gate_name = std::get<0>(GetParam());
    pfunc_params func = std::get<1>(GetParam());
    const vector<double> params = std::get<2>(GetParam());


    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, params);

    // two qubits, apply on first
    CplxType[] stateVec = {1.0, 0.0, 0.0, 0.0};
    size_t stateVecSize = sizeof(stateVec) / sizeof(stateVec[0]);
    StateVector state = StateVector(stateVec, stateVecSize);

    vector<size_t> internalIndices{0};
    vector<size_t> externalIndices{1};

    expectedStateVec = std::get<3>(GetParam());
    size_t expectedStateVecSize = sizeof(expectedStateVec) / sizeof(expectedStateVec[0]);
    StateVector expectedState = StateVector(expectedStateVec, expectedStateVecSize);

    gate->applyDerivative(state, internalIndices, externalIndices);
    EXPECT_EQ(state, expectedState);
}
CplxType EXPECTED_STATES[][] = {
    {CplxType(0.03073062, 0.0), CplxType(0.0, 0.49905474)},
    {CplxType(0.03073062, 0.0), CplxType(-0.49905474, 0.0)},
    {CplxType(0.03073062, 0.49905474), CplxType(0.0, 0.0)},
    {CplxType(0.0, 0.0), CplxType(0.0, 0.0)}
}
INSTANTIATE_TEST_SUITE_P (
        GateMatrix,
        MatrixWithParamsFixture,
        ::testing::Values(
                std::make_tuple("RX", RX, ONE_PARAM, EXPECTED_STATES[0]),
                std::make_tuple("RY", RY, ONE_PARAM, EXPECTED_STATES[0]),
                std::make_tuple("RZ", RZ, ONE_PARAM, EXPECTED_STATES[0]),
                std::make_tuple("PhaseShift", PhaseShift, ONE_PARAM, EXPECTED_STATES[0]),
                // std::make_tuple("CRX", CRX, ONE_PARAM),
                // std::make_tuple("CRY", CRY, ONE_PARAM),
                // std::make_tuple("CRZ", CRZ, ONE_PARAM),
));