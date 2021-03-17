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
/*
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
*/

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

class SeparateControlTarget : public ::testing::TestWithParam<std::tuple<string, INDICES, std::tuple<INDICES, INDICES >> > {
};

TEST_P(SeparateControlTarget, SeparateControlTarget) {
    const string op = std::get<0>(GetParam());
    const INDICES wires = std::get<1>(GetParam());
    auto expected =std::get<2>(GetParam());

    auto res_wires = Pennylane::separate_control_and_target(op, wires);

    ASSERT_EQ(res_wires, expected);
}

INSTANTIATE_TEST_SUITE_P (
        SeparateControlTargetTests,
        SeparateControlTarget,
        ::testing::Values(
                              // Gate   all wires                  control wires  target wires
                std::make_tuple("RY", INDICES{1}, std::make_tuple(INDICES{}, INDICES{1})),
                std::make_tuple("CNOT", INDICES{0,1}, std::make_tuple(INDICES{0}, INDICES{1})),
                std::make_tuple("CNOT", INDICES{1,0}, std::make_tuple(INDICES{1}, INDICES{0})),
                std::make_tuple("SWAP", INDICES{0,1}, std::make_tuple(INDICES{}, INDICES{0,1})),
                std::make_tuple("SWAP", INDICES{1,0}, std::make_tuple(INDICES{}, INDICES{1,0})),
                std::make_tuple("Toffoli", INDICES{0,1,2}, std::make_tuple(INDICES{0,1}, INDICES{2})),
                std::make_tuple("Toffoli", INDICES{1,0,2}, std::make_tuple(INDICES{1,0}, INDICES{2})),
                std::make_tuple("CSWAP", INDICES{0,2,1}, std::make_tuple(INDICES{0}, INDICES{2,1})),
                std::make_tuple("CSWAP", INDICES{2,1,0}, std::make_tuple(INDICES{2}, INDICES{1,0}))
    ));

class GetNewQubitList : public ::testing::TestWithParam<std::tuple<string, INDICES, string, INDICES, INDICES,INDICES > > {
};

TEST_P(GetNewQubitList, GetNewQubitList) {
    const string op1 = std::get<0>(GetParam());
    const INDICES wires1 = std::get<1>(GetParam());

    const string op2 = std::get<2>(GetParam());
    const INDICES wires2 = std::get<3>(GetParam());

    const INDICES control_expected = std::get<4>(GetParam());
    const INDICES target_expected = std::get<5>(GetParam());

    auto all_res = Pennylane::get_new_qubit_list(op1, wires1, op2, wires2);
    const INDICES control_result = std::get<0>(all_res);
    const INDICES target_result = std::get<1>(all_res);

    //std::tie(control_result, target_result) = all_res;

    ASSERT_EQ(control_result, control_expected);
    ASSERT_EQ(target_result, target_expected);
}

INSTANTIATE_TEST_SUITE_P (
        GetNewQubitListTests,
        GetNewQubitList,
        ::testing::Values(
                std::make_tuple("RY", INDICES{1}, "RY", INDICES{1}, INDICES{}, INDICES{1}),
                std::make_tuple("CNOT", INDICES{0,1}, "RY", INDICES{1}, INDICES{}, INDICES{1,0}),
                std::make_tuple("CNOT", INDICES{0,1}, "SWAP", INDICES{1,2}, INDICES{}, INDICES{1,0,2}),
                std::make_tuple("CNOT", INDICES{0,1}, "SWAP", INDICES{1,0}, INDICES{}, INDICES{1,0}),
                std::make_tuple("Toffoli", INDICES{0,1,2}, "SWAP", INDICES{1,0}, INDICES{}, INDICES{2,0,1})
    ));

}



