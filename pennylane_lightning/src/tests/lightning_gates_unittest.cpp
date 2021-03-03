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

using std::unique_ptr;
using std::vector;
using std::string;

using Pennylane::CplxType;

namespace test_gates{

TEST(constructGate, MatrixNoParam){
    const string gate_name = "PauliX";

    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, {});

    EXPECT_EQ(gate->asMatrix(), PauliX);
}

TEST(constructGate, RX){
    const string gate_name = "RX";
    vector<double> params = {0.3};

    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, params);

    EXPECT_EQ(gate->asMatrix(), RX(params.at(0)));
}

TEST(constructGate, CNOT){
    const string gate_name = "CNOT";
    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, {});
    EXPECT_EQ(gate->asMatrix(), CNOT);
}

// TODO: add tests for input validation error

}
