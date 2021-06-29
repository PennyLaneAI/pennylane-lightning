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
#include <memory>
#include <math.h>

using std::vector;
using Pennylane::CplxType;

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

namespace test_apply_generator {

class applyGeneratorFixture :public ::testing::TestWithParam<std::tuple<std::string, vector<double>, vector<CplxType>, vector<CplxType>>> {
};


vector<CplxType> vec = {1, 0};
Pennylane::StateVector expected(vec.data(), vec.size());

TEST_P(applyGeneratorFixture, applyGeneratorPTest) {
    std::string opLabel = std::get<0>(GetParam());
    vector<double> par = std::get<1>(GetParam());
    vector<CplxType> input = std::get<2>(GetParam());
    vector<CplxType> expected = std::get<3>(GetParam());

    std::unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(opLabel, par);

    Pennylane::StateVector init_state(input.data(), input.size());
    int qubits = int(log2(input.size()));
    vector<unsigned int> op_wires = (qubits == 1) ? vector<unsigned int>{0} : vector<unsigned int>{0, 1};

    Pennylane::applyGateGenerator(init_state, gate, op_wires, qubits);
    for(unsigned int i=0; i<init_state.length; ++i){
        ASSERT_NEAR(init_state.arr[i].real(), expected[i].real(), 1e-5);
        ASSERT_NEAR(init_state.arr[i].imag(), expected[i].imag(), 1e-5);
    }
}

INSTANTIATE_TEST_SUITE_P (
        applyGeneratorTests,
        applyGeneratorFixture,
        ::testing::Values(

                // Single-qubit elementary rotations
                std::make_tuple("RX", vector<double>{3.14}, vector<CplxType>{1,0}, vector<CplxType>{0, 1}),
                std::make_tuple("RY", vector<double>{3.14}, vector<CplxType>{1,0}, vector<CplxType>{0, CplxType(0,1)}),
                std::make_tuple("RZ", vector<double>{3.14}, vector<CplxType>{1,0}, vector<CplxType>{1,0}),
                std::make_tuple("RX", vector<double>{3.14}, vector<CplxType>{0,1}, vector<CplxType>{1, 0}),
                std::make_tuple("RY", vector<double>{3.14}, vector<CplxType>{0,1}, vector<CplxType>{CplxType(0,-1), 0}),
                std::make_tuple("RZ", vector<double>{3.14}, vector<CplxType>{0,1}, vector<CplxType>{0,-1}),

                // Controlled two-qubit rotations
                std::make_tuple("CRZ", vector<double>{3.14}, vector<CplxType>{1,0,0,0}, vector<CplxType>{0,0,0,0}),
                std::make_tuple("CRZ", vector<double>{3.14}, vector<CplxType>{0,1,0,0}, vector<CplxType>{0,0,0,0}),
                std::make_tuple("CRZ", vector<double>{3.14}, vector<CplxType>{0,0,1,0}, vector<CplxType>{0,0,1,0}),
                std::make_tuple("CRZ", vector<double>{3.14}, vector<CplxType>{0,0,0,1}, vector<CplxType>{0,0,0,-1})
                ));

}
