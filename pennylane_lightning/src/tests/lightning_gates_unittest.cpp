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

using std::unique_ptr;
using std::vector;
using std::string;

using Pennylane::CplxType;

namespace test_gates {

TEST(constructGate, PauliX){
    const vector<double> params (5, 10.0);

    const string gate_name = "PauliX";

    unique_ptr<Pennylane::AbstractGate> gate = Pennylane::constructGate(gate_name, params);

    //const vector<CplxType> gate_matrix = gate->asMatrix();
    const vector<CplxType> target_matrix{0, 1, 1, 0 };

    //std::cout<<gate_matrix<<"\n";
    //std::cout<<target_matrix<<"\n";

    //EXPECT_EQ(gate_matrix, target_matrix);
}

}