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
#include "gmock/gmock.h"
#include "../Apply.hpp"
#include "../StateVector.hpp"
#include <vector>

using std::string;
using std::array;
using std::vector;
using Pennylane::CplxType;
using Pennylane::StateVector;

namespace test_apply_x {

    class applyXFixture : public ::testing::TestWithParam<std::tuple<std::vector<CplxType>, std::vector<CplxType> > > {
    };

    TEST_P(applyXFixture, CheckParamLength) {
        vector<CplxType> vec = std::get<0>(GetParam());
        const vector<CplxType> expected = std::get<1>(GetParam());

        StateVector state = StateVector(vec.data(), vec.size());
        apply_x(state, 0, 1);
        //ASSERT_THAT(result, ::testing::ElementsAre(expected));
        ASSERT_TRUE(memcmp(state.arr, expected.data(), 2)==0 );
    }

    const vector<string> non_param_gates = {"PauliX", "PauliY", "PauliZ", "Hadamard", "S", "T", "CNOT", "SWAP", "CZ", "Toffoli", "CSWAP"};
    const vector<vector<double>> many_params = {{0.3}};

    INSTANTIATE_TEST_SUITE_P (
            applyXChecks,
            applyXFixture,
            ::testing::Values(
                    std::make_tuple(vector<CplxType>{1,0}, vector<CplxType>{0,1})
    ));
}
