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
#include "../rework/Util.hpp"

namespace test_utils {

TEST(exp2, fixed_example) {
    std::vector<unsigned int> inputs = {1, 2, 5, 8};
    std::vector<size_t> outputs = {2, 4, 32, 256};

    for (size_t i = 0; i < inputs.size(); i++) {
        size_t result = Pennylane::exp2(inputs[i]);
        EXPECT_TRUE(result == outputs[i]);
    }
}

TEST(maxDecimalForQubit, fixed_example) {
    std::vector<std::vector<unsigned int>> inputs = {
        {0, 3},
        {1, 3},
        {2, 3},
        {0, 4},
        {2, 4},
        {2, 5},
    };
    std::vector<size_t> outputs = {4, 2, 1, 8, 2, 4};

    for (size_t i = 0; i < inputs.size(); i++) {
        size_t result = Pennylane::maxDecimalForQubit(inputs[i][0], inputs[i][1]);
        EXPECT_TRUE(result == outputs[i]);
    }
}

}
