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

class GetExtendedMatrix : public ::testing::TestWithParam<std::tuple<string, INDICES, INDICES, INDICES, INDICES, vector<CplxType>  > > {
};

TEST_P(GetExtendedMatrix, GetExtendedMatrix) {
    const string gate_name = std::get<0>(GetParam());
    unique_ptr<AbstractGate> gate = Pennylane::constructGate(gate_name, {});
    vector<CplxType> mx;

    INDICES new_controls = std::get<1>(GetParam());
    INDICES new_targets = std::get<2>(GetParam());
    INDICES first_controls = std::get<3>(GetParam());
    INDICES first_targets = std::get<4>(GetParam());
    Pennylane::get_extended_matrix(std::move(gate), mx, new_controls, new_targets, first_controls, first_targets);

    auto expected = std::get<5>(GetParam());
    ASSERT_EQ(mx, expected);
}

INSTANTIATE_TEST_SUITE_P (
        GetExtendedMatrixTests,
        GetExtendedMatrix,
        ::testing::Values(
                std::make_tuple("PauliX", INDICES{},  INDICES{0,1}, INDICES{}, INDICES{0}, vector<CplxType>{0,0,1,0,
                                                                                                            0,0,0,1,
                                                                                                            1,0,0,0,
                                                                                                            0,1,0,0}),
                std::make_tuple("PauliX", INDICES{},  INDICES{0,1}, INDICES{}, INDICES{1}, vector<CplxType>{0,1,0,0,
                                                                                                            1,0,0,0,
                                                                                                            0,0,0,1,
                                                                                                            0,0,1,0}),
                std::make_tuple("CNOT", INDICES{},  INDICES{0,1}, INDICES{}, INDICES{0, 1}, vector<CplxType>{1,0,0,0,
                                                                                                             0,1,0,0,
                                                                                                             0,0,0,1,
                                                                                                             0,0,1,0}),
                std::make_tuple("CZ", INDICES{},  INDICES{0,1}, INDICES{}, INDICES{0, 1}, vector<CplxType>{1,0,0,0,
                                                                                                             0,1,0,0,
                                                                                                             0,0,1,0,
                                                                                                             0,0,0,-1}),
                std::make_tuple("CNOT", INDICES{},  INDICES{0,1,2}, INDICES{}, INDICES{0,1}, vector<CplxType>{1, 0, 0, 0, 0, 0, 0, 0,
                                                                                                            0, 1, 0, 0, 0, 0, 0, 0,
                                                                                                            0, 0, 1, 0, 0, 0, 0, 0,
                                                                                                            0, 0, 0, 1, 0, 0, 0, 0,
                                                                                                            0, 0, 0, 0, 0, 0, 1, 0,
                                                                                                            0, 0, 0, 0, 0, 0, 0, 1,
                                                                                                            0, 0, 0, 0, 1, 0, 0, 0,
                                                                                                            0, 0, 0, 0, 0, 1, 0, 0
}),
                std::make_tuple("CNOT", INDICES{},  INDICES{0,1,2}, INDICES{}, INDICES{1,0}, vector<CplxType>{1, 0, 0, 0, 0, 0, 0, 0,
                                                                                                                0, 1, 0, 0, 0, 0, 0, 0,
                                                                                                                0, 0, 0, 0, 0, 0, 1, 0,
                                                                                                                0, 0, 0, 0, 0, 0, 0, 1,
                                                                                                                0, 0, 0, 0, 1, 0, 0, 0,
                                                                                                                0, 0, 0, 0, 0, 1, 0, 0,
                                                                                                                0, 0, 1, 0, 0, 0, 0, 0,
                                                                                                                0, 0, 0, 1, 0, 0, 0, 0
}),
                std::make_tuple("CNOT", INDICES{},  INDICES{0,1,2}, INDICES{}, INDICES{1,2}, vector<CplxType>{1, 0, 0, 0, 0, 0, 0, 0,
                                                                                                            0, 1, 0, 0, 0, 0, 0, 0,
                                                                                                            0, 0, 0, 1, 0, 0, 0, 0,
                                                                                                            0, 0, 1, 0, 0, 0, 0, 0,
                                                                                                            0, 0, 0, 0, 1, 0, 0, 0,
                                                                                                            0, 0, 0, 0, 0, 1, 0, 0,
                                                                                                            0, 0, 0, 0, 0, 0, 0, 1,
                                                                                                            0, 0, 0, 0, 0, 0, 1, 0
}),
                std::make_tuple("CNOT", INDICES{},  INDICES{0,1,2}, INDICES{}, INDICES{2,1}, vector<CplxType>{1, 0, 0, 0, 0, 0, 0, 0,
                                                                                                            0, 0, 0, 1, 0, 0, 0, 0,
                                                                                                            0, 0, 1, 0, 0, 0, 0, 0,
                                                                                                            0, 1, 0, 0, 0, 0, 0, 0,
                                                                                                            0, 0, 0, 0, 1, 0, 0, 0,
                                                                                                            0, 0, 0, 0, 0, 0, 0, 1,
                                                                                                            0, 0, 0, 0, 0, 0, 1, 0,
                                                                                                            0, 0, 0, 0, 0, 1, 0, 0
})/*,

                std::make_tuple("CNOT", INDICES{},  INDICES{0,1,2}, INDICES{}, INDICES{0,2}, vector<CplxType>{1, 0, 0, 0, 0, 0, 0, 0,
                                                                                                            0, 1, 0, 0, 0, 0, 0, 0,
                                                                                                            0, 0, 0, 0, 0, 1, 0, 0,
                                                                                                            0, 0, 0, 0, 1, 0, 0, 0,
                                                                                                            0, 0, 1, 0, 0, 0, 0, 0,
                                                                                                            0, 0, 0, 1, 0, 0, 0, 0,
                                                                                                            0, 0, 0, 0, 0, 0, 0, 1,
                                                                                                            0, 0, 0, 0, 0, 0, 1, 0
})
*/


    ));

class Merge : public ::testing::TestWithParam<std::tuple<string, string, INDICES, INDICES, vector<CplxType>  > > {
};

TEST_P(Merge, Merge) {
    string label1 = std::get<0>(GetParam());
    string label2 = std::get<1>(GetParam());

    INDICES wires1 = std::get<2>(GetParam());
    INDICES wires2 = std::get<3>(GetParam());

    const vector<CplxType> expected = std::get<4>(GetParam());

    unique_ptr<AbstractGate> gate1 = Pennylane::constructGate(label1, {});
    unique_ptr<AbstractGate> gate2 = Pennylane::constructGate(label2, {});

    auto gate = Pennylane::merge(move(gate1), label1, wires1, move(gate2), label2, wires2);
    auto res_matrix = gate->asMatrix();

    ASSERT_EQ(res_matrix, expected);
}

INSTANTIATE_TEST_SUITE_P (
        MergeTests,
        Merge,
        ::testing::Values(
            std::make_tuple("PauliX", "PauliX", INDICES{0}, INDICES{0}, vector<CplxType>{1,0,0,1}),
            std::make_tuple("PauliX", "PauliX", INDICES{1}, INDICES{1}, vector<CplxType>{1,0,0,1}),
            std::make_tuple("PauliX", "PauliX", INDICES{0}, INDICES{1}, vector<CplxType>{
                                                    0, 0, 0, 1,
                                                    0, 0, 1, 0,
                                                    0, 1, 0, 0,
                                                    1, 0, 0, 0}),
            std::make_tuple("PauliX", "PauliX", INDICES{1}, INDICES{0}, vector<CplxType>{
                                                     0, 0, 0, 1,
                                                     0, 0, 1, 0,
                                                     0, 1, 0, 0,
                                                     1, 0, 0, 0}),
            std::make_tuple("CNOT", "CNOT", INDICES{0, 1}, INDICES{0,1}, vector<CplxType>{
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    0, 0, 0, 1}),
            std::make_tuple("CZ", "CZ", INDICES{0, 1}, INDICES{0,1}, vector<CplxType>{
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    0, 0, 0, 1}),
            std::make_tuple("CZ", "PauliX", INDICES{0, 1}, INDICES{2}, vector<CplxType>{
                                                 0,  1,  0,  0,  0,  0,  0,  0,
                                                 1,  0,  0,  0,  0,  0,  0,  0,
                                                 0,  0,  0,  1,  0,  0,  0,  0,
                                                 0,  0,  1,  0,  0,  0,  0,  0,
                                                 0,  0,  0,  0,  0,  1,  0,  0,
                                                 0,  0,  0,  0,  1,  0,  0,  0,
                                                 0,  0,  0,  0,  0,  0,  0, -1,
                                                 0,  0,  0,  0,  0,  0, -1,  0}),
            std::make_tuple("CZ", "PauliX", INDICES{1,2}, INDICES{0}, vector<CplxType>{
                                                 0,  0,  0,  0,  1,  0,  0,  0,
                                                 0,  0,  0,  0,  0,  1,  0,  0,
                                                 0,  0,  0,  0,  0,  0,  1,  0,
                                                 0,  0,  0,  0,  0,  0,  0, -1,
                                                 1,  0,  0,  0,  0,  0,  0,  0,
                                                 0,  1,  0,  0,  0,  0,  0,  0,
                                                 0,  0,  1,  0,  0,  0,  0,  0,
                                                 0,  0,  0, -1,  0,  0,  0,  0})

    ));

unique_ptr<AbstractGate> aux_func(vector<unique_ptr<AbstractGate>> && gates, const string& label, vector<unsigned int>& wires) {
    return Pennylane::merge(std::move(gates[0]), label, wires, std::move(gates[1]), label, wires);
}

TEST(MergeThroughPtrs, MergeThroughPtrs) {
    string label = "PauliX";
    unique_ptr<AbstractGate> gate1 = Pennylane::constructGate("PauliX", {});
    unique_ptr<AbstractGate> gate2 = Pennylane::constructGate("PauliX", {});

    vector<unique_ptr<AbstractGate>> gates;
    gates.push_back(std::move(gate1));
    gates.push_back(std::move(gate2));

    INDICES wires = {0};
    auto gate = aux_func(std::move(gates), label, wires);

    vector<CplxType> expected = {1,0,0,1};
    ASSERT_EQ(gate->asMatrix(), expected);
}

class OptimizeLight : public ::testing::TestWithParam<std::tuple<vector<string>, vector<INDICES> , unsigned int, vector<vector<CplxType> >, vector<INDICES> >> {
};

TEST_P(OptimizeLight, OptimizeLight) {
    vector<unique_ptr<AbstractGate>> gates;

    vector<string> gate_names = std::get<0>(GetParam());
    for (auto gate : gate_names){
        gates.push_back(std::move(Pennylane::constructGate(gate, {})));
    }

    vector<INDICES> wires = std::get<1>(GetParam());
    const unsigned int num_expected_gates = std::get<2>(GetParam());
    auto expected_matrices = std::get<3>(GetParam());
    auto expected_wires = std::get<4>(GetParam());

    auto num_w1 = wires[0].size();
    auto num_w2 = wires[1].size();
    const unsigned int num_qubits = num_w1 >= num_w2 ? num_w1 : num_w2;

    Pennylane::optimize_light(std::move(gates), gate_names, wires, num_qubits);
    ASSERT_EQ(gates.size(), num_expected_gates);
    ASSERT_EQ(gates[0]->asMatrix(), expected_matrices[0]);

    ASSERT_EQ(wires.size(), 1);
    ASSERT_EQ(wires[0], expected_wires[0]);
}

INSTANTIATE_TEST_SUITE_P (
        OptimizeLightNonParamTests,
        OptimizeLight,
        ::testing::Values(
            // Unitarity
            std::make_tuple(vector<string>{"PauliX", "PauliX"}, vector<INDICES>{{0}, {0}}, 1, vector<vector<CplxType>>{{1,0,0,1}}, vector<INDICES>{{0}}),
            std::make_tuple(vector<string>{"PauliY", "PauliY"}, vector<INDICES>{{0}, {0}}, 1, vector<vector<CplxType>>{{1,0,0,1}}, vector<INDICES>{{0}}),
            std::make_tuple(vector<string>{"PauliZ", "PauliZ"}, vector<INDICES>{{0}, {0}}, 1, vector<vector<CplxType>>{{1,0,0,1}}, vector<INDICES>{{0}}),

            // Note: we process gates first in first out, so we compute U = v[1] @ v[0]
            // Merging Paulis
            std::make_tuple(vector<string>{"PauliX", "PauliY"}, vector<INDICES>{{0}, {0}}, 1, vector<vector<CplxType>>{{CplxType(0, -1),0,0,CplxType(0, 1)}}, vector<INDICES>{{0}}),
            std::make_tuple(vector<string>{"PauliY", "PauliZ"}, vector<INDICES>{{0}, {0}}, 1, vector<vector<CplxType>>{{0,CplxType(0, -1),CplxType(0, -1),0}}, vector<INDICES>{{0}}),
            std::make_tuple(vector<string>{"PauliZ", "PauliX"}, vector<INDICES>{{0}, {0}}, 1, vector<vector<CplxType>>{{0,-1,1,0}}, vector<INDICES>{{0}}),

            // Merging two-qubit with one-qubit gate (note: same target qubits)
            //TODO: separate control targets to allow the following case:
            /*
            std::make_tuple(vector<string>{"PauliX", "CZ"}, vector<INDICES>{{1}, {0,1}}, 1, vector<vector<CplxType>>{{
                                                                                0,  1,  0,  0,
                                                                                1,  0,  0,  0,
                                                                                0,  0,  0,  1,
                                                                                0,  0, -1,  0
                                                                                }}),
            */
            std::make_tuple(vector<string>{"CZ", "PauliX"}, vector<INDICES>{{0,1}, {1}}, 1, vector<vector<CplxType>>{{
                                                                                0,  1,  0,  0,
                                                                                1,  0,  0,  0,
                                                                                0,  0,  0, -1,
                                                                                0,  0,  1,  0
                                                                                }}, vector<INDICES>{{0,1}})
        ));

class OptimizeLightParamOps : public ::testing::TestWithParam<std::tuple<vector<string>, vector<vector<double>>, vector<INDICES> , unsigned int, vector<vector<CplxType> >, vector<INDICES> >> {
};

TEST_P(OptimizeLightParamOps, OptimizeLightParametrizedOps) {
    vector<string> gate_names = std::get<0>(GetParam());
    vector<vector<double>> params = std::get<1>(GetParam());
    vector<INDICES> wires = std::get<2>(GetParam());

    const unsigned int num_expected_gates = std::get<3>(GetParam());
    auto expected_matrices = std::get<4>(GetParam());
    auto expected_wires = std::get<5>(GetParam());

    vector<unique_ptr<AbstractGate>> gates;

    for (int i=0; i<gate_names.size(); ++i){
        gates.push_back(std::move(Pennylane::constructGate(gate_names[i], params[i])));
    }

    auto num_w1 = wires[0].size();
    auto num_w2 = wires[1].size();
    const unsigned int num_qubits = num_w1 >= num_w2 ? num_w1 : num_w2;

    Pennylane::optimize_light(std::move(gates), gate_names, wires, num_qubits);
    ASSERT_EQ(gates.size(), num_expected_gates);
    ASSERT_EQ(gates[0]->asMatrix(), expected_matrices[0]);

    ASSERT_EQ(wires.size(), 1);
    ASSERT_EQ(wires[0], expected_wires[0]);
}

vector<vector<double>> test_params = {{0.5432}, {0.5432}, {0.5432}};
vector<INDICES> test_wires = {{0}, {0}, {0}};
vector<INDICES> expected_wires = {{0}};

CplxType param = 0.5432;
auto half_param = param/CplxType(2.0);
CplxType c1 = CplxType(0.5)*(cos(half_param) + cos(CplxType(3.0)*half_param));
CplxType c2 = -((CplxType(1.0) + CplxType(0, 1)) + cos(param)) + sin(half_param);
CplxType c3 = CplxType(0.5)*((CplxType(1.0) - CplxType(0, 2))*sin(half_param) + sin(CplxType(3.0)*half_param));

/*
INSTANTIATE_TEST_SUITE_P (
        OptimizeLightParamOpsTests,
        OptimizeLightParamOps,
        ::testing::Values(
            // Unitarity
            std::make_tuple(vector<string>{"RY", "RX", "RY"}, test_params, test_wires, 1, vector<vector<CplxType>>{{c1,c2,c3,c1}}, expected_wires)
        ));
*/

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

class SetBlock : public ::testing::TestWithParam<std::tuple<vector<CplxType>, size_t, size_t, vector<CplxType>, size_t, vector<CplxType> > > {
};

TEST_P(SetBlock, SetBlock) {
    auto big_mx = std::get<0>(GetParam());
    auto dim = std::get<1>(GetParam());

    auto start_index = std::get<2>(GetParam());

    auto block_mx = std::get<3>(GetParam());
    auto block_dim = std::get<4>(GetParam());

    auto expected = std::get<5>(GetParam());

    Pennylane::set_block(big_mx.data(), dim, start_index, block_mx.data(), block_dim);
    ASSERT_EQ(big_mx, expected);
}

INSTANTIATE_TEST_SUITE_P (
        SetBlockTests,
        SetBlock,
        ::testing::Values(
                std::make_tuple(vector<CplxType>{0,0,0,0}, 2, 0, vector<CplxType>{1}, 1, vector<CplxType>{1,0,0,0}),
                std::make_tuple(vector<CplxType>{0,0,0,0}, 2, 2, vector<CplxType>{1}, 1, vector<CplxType>{0,0,1,0}),
                std::make_tuple(vector<CplxType>{0,0,0,0}, 2, 0, vector<CplxType>{1,0,0,1}, 2, vector<CplxType>{1,0,0,1}),

                std::make_tuple(vector<CplxType>{0,0,0,0,
                                                 0,0,0,0,
                                                 0,0,0,0,
                                                 0,0,0,0}, 4, 0, vector<CplxType>{1,0,0,1}, 2,
                                 vector<CplxType>{1,0,0,0,
                                                  0,1,0,0,
                                                  0,0,0,0,
                                                  0,0,0,0}),


                std::make_tuple(vector<CplxType>{0,0,0,0,
                                                 0,0,0,0,
                                                 0,0,0,0,
                                                 0,0,0,0}, 4, 2,
                                vector<CplxType>{1,0,
                                                 0,1}, 2,
                                vector<CplxType>{0,0,1,0,
                                                 0,0,0,1,
                                                 0,0,0,0,
                                                 0,0,0,0}),


                std::make_tuple(vector<CplxType>{0,0,0,0,
                                                 0,0,0,0,
                                                 0,0,0,0,
                                                 0,0,0,0}, 4, 8,
                                vector<CplxType>{1,0,
                                                 0,1}, 2,
                                vector<CplxType>{0,0,0,0,
                                                 0,0,0,0,
                                                 1,0,0,0,
                                                 0,1,0,0}),


                std::make_tuple(vector<CplxType>{0,0,0,0,
                                                 0,0,0,0,
                                                 0,0,0,0,
                                                 0,0,0,0}, 4, 10,
                                vector<CplxType>{1,0,
                                                 0,1}, 2,
                                vector<CplxType>{0,0,0,0,
                                                 0,0,0,0,
                                                 0,0,1,0,
                                                 0,0,0,1})
    ));

class SwapRows : public ::testing::TestWithParam<std::tuple<vector<CplxType>, size_t, size_t, size_t, vector<CplxType> > > {
};

TEST_P(SwapRows, SwapRows) {
    auto mx = std::get<0>(GetParam());
    auto dim = std::get<1>(GetParam());

    auto row1 = std::get<2>(GetParam());
    auto row2 = std::get<3>(GetParam());

    auto expected = std::get<4>(GetParam());

    Pennylane::swap_rows(mx.data(), dim, row1, row2);
    ASSERT_EQ(mx, expected);
}

INSTANTIATE_TEST_SUITE_P (
        SwapRowsTests,
        SwapRows,
        ::testing::Values(
                std::make_tuple(vector<CplxType>{1,2,
                                                 3,4},
                                                 2, 0, 1,
                                vector<CplxType>{3,4,
                                                 1,2}),


                std::make_tuple(vector<CplxType>{1,2,
                                                 3,4},
                                                 2, 1, 0,
                                vector<CplxType>{3,4,
                                                 1,2}),


                std::make_tuple(vector<CplxType>{1,2,3,
                                                 4,5,6,
                                                 7,8,9},
                                                 3, 0, 2,
                                vector<CplxType>{7,8,9,
                                                 4,5,6,
                                                 1,2,3}),

                std::make_tuple(vector<CplxType>{1,2,3,
                                                 4,5,6,
                                                 7,8,9},
                                                 3, 2, 1,
                                vector<CplxType>{1,2,3,
                                                 7,8,9,
                                                 4,5,6})
    ));

class SwapCols : public ::testing::TestWithParam<std::tuple<vector<CplxType>, size_t, size_t, size_t, vector<CplxType> > > {
};

TEST_P(SwapCols, SwapCols) {
    auto mx = std::get<0>(GetParam());
    auto dim = std::get<1>(GetParam());

    auto col1 = std::get<2>(GetParam());
    auto col2 = std::get<3>(GetParam());

    auto expected = std::get<4>(GetParam());

    Pennylane::swap_cols(mx.data(), dim, col1, col2);
    ASSERT_EQ(mx, expected);
}

INSTANTIATE_TEST_SUITE_P (
        SwapColsTests,
        SwapCols,
        ::testing::Values(
                std::make_tuple(vector<CplxType>{1,2,
                                                 3,4},
                                                 2, 0, 1,
                                vector<CplxType>{2,1,
                                                 4,3}),


                std::make_tuple(vector<CplxType>{1,2,
                                                 3,4},
                                                 2, 1, 0,
                                vector<CplxType>{2,1,
                                                 4,3}),


                std::make_tuple(vector<CplxType>{1,2,3,
                                                 4,5,6,
                                                 7,8,9},
                                                 3, 0, 2,
                                vector<CplxType>{3,2,1,
                                                 6,5,4,
                                                 9,8,7}),


                std::make_tuple(vector<CplxType>{1,2,3,
                                                 4,5,6,
                                                 7,8,9},
                                                 3, 2, 1,
                                vector<CplxType>{1,3,2,
                                                 4,6,5,
                                                 7,9,8})
    ));

