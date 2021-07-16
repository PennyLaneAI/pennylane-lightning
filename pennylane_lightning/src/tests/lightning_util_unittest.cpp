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
#include <tuple>

#include "../Util.hpp"
#include "gtest/gtest.h"

namespace test_utils {

class Exp2TestFixture : public ::testing::TestWithParam<std::tuple<int, int>> {
};

TEST_P(Exp2TestFixture, CheckExp2Results) {
    int input = std::get<0>(GetParam());
    size_t expected = std::get<1>(GetParam());
    ASSERT_EQ(expected, Pennylane::exp2(input));
}

INSTANTIATE_TEST_SUITE_P(Exp2Tests, Exp2TestFixture,
                         ::testing::Values(std::make_tuple(1, 2),
                                           std::make_tuple(2, 4),
                                           std::make_tuple(5, 32),
                                           std::make_tuple(8, 256)));

class maxDecimalForQubitTestFixture
    : public ::testing::TestWithParam<
          std::tuple<unsigned int, unsigned int, size_t>> {};

TEST_P(maxDecimalForQubitTestFixture, CheckMaxDecimalResults) {
    unsigned int qubitIndex = std::get<0>(GetParam());
    unsigned int qubits = std::get<1>(GetParam());
    size_t expected = std::get<2>(GetParam());
    ASSERT_EQ(expected, Pennylane::maxDecimalForQubit(qubitIndex, qubits));
}

INSTANTIATE_TEST_SUITE_P(
    maxDecimalForQubitTests, maxDecimalForQubitTestFixture,
    ::testing::Values(std::make_tuple(0, 3, 4), std::make_tuple(1, 3, 2),
                      std::make_tuple(2, 3, 1), std::make_tuple(0, 4, 8),
                      std::make_tuple(2, 4, 2), std::make_tuple(2, 5, 4)));

} // namespace test_utils
