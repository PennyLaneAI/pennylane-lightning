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
    get_extended_matrix(std::move(paulix), mx);
    ASSERT_EQ(1, 1);
}
}
