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
#include "../Optimize.hpp"
#include <iostream>

using std::unique_ptr;
using std::vector;
using std::string;
using std::function;

using Pennylane::CplxType;
using Pennylane::AbstractGate;

namespace test_optimize{

TEST(light_optimize, get_extended_matrix) {
    unique_ptr<AbstractGate> paulix = Pennylane::constructGate("PauliX", {});
    vector<CplxType> mx = paulix->asMatrix();

    vector<unsigned int> wires1 = {0,1};
    vector<unsigned int> wires2 = {2,3};
    vector<unsigned int> wires3 = {4};
    get_extended_matrix(std::move(paulix), mx, wires1, wires2,wires3);
    ASSERT_EQ(1, 1);
}

TEST(light_optimize, create_identity) {
    vector<CplxType> mx = Pennylane::create_identity(2);
    vector<CplxType> expected = {1,0,0,1};

    ASSERT_EQ(mx, expected);
}

class CreateIdentity : public ::testing::TestWithParam<std::tuple<unsigned int, vector<CplxType> > > {
};

TEST_P(CreateIdentity, CreateIdentity) {
    const unsigned int dim = std::get<0>(GetParam());
    const vector<CplxType> expected =std::get<1>(GetParam());

    vector<CplxType> mx = Pennylane::create_identity(dim);

    ASSERT_EQ(mx, expected);
}

INSTANTIATE_TEST_SUITE_P (
        IdentityTests,
        CreateIdentity,
        ::testing::Values(
                std::make_tuple(2, vector<CplxType>{1,0,0,1}),
                std::make_tuple(4, vector<CplxType>{1,0,0,0,
                                                    0,1,0,0,
                                                    0,0,1,0,
                                                    0,0,0,1})
    ));

}
