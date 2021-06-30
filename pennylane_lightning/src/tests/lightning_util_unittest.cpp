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
#include "../Util.hpp"

#include <tuple>

using std::vector;
using Pennylane::CplxType;
using Pennylane::StateVector;

namespace test_utils {

class Exp2TestFixture :public ::testing::TestWithParam<std::tuple<int, int>> {
};

TEST_P(Exp2TestFixture, CheckExp2Results) {
    int input = std::get<0>(GetParam());
    size_t expected = std::get<1>(GetParam());
    ASSERT_EQ(expected, Pennylane::exp2(input));
}

INSTANTIATE_TEST_SUITE_P (
        Exp2Tests,
        Exp2TestFixture,
        ::testing::Values(
                std::make_tuple(1, 2),
                std::make_tuple(2, 4),
                std::make_tuple(5, 32),
                std::make_tuple(8, 256)));

class maxDecimalForQubitTestFixture :public ::testing::TestWithParam<std::tuple<unsigned int, unsigned int, size_t>> {
};

TEST_P(maxDecimalForQubitTestFixture, CheckMaxDecimalResults) {
    unsigned int qubitIndex = std::get<0>(GetParam());
    unsigned int qubits = std::get<1>(GetParam());
    size_t expected = std::get<2>(GetParam());
    ASSERT_EQ(expected, Pennylane::maxDecimalForQubit(qubitIndex, qubits));
}

INSTANTIATE_TEST_SUITE_P (
        maxDecimalForQubitTests,
        maxDecimalForQubitTestFixture,
        ::testing::Values(
                std::make_tuple(0, 3, 4),
                std::make_tuple(1, 3, 2),
                std::make_tuple(2, 3, 1),
                std::make_tuple(0, 4, 8),
                std::make_tuple(2, 4, 2),
                std::make_tuple(2, 5, 4)));


class innerProductFixture :public ::testing::TestWithParam<std::tuple<vector<CplxType>, vector<CplxType>, CplxType>> {
};

TEST_P(innerProductFixture, innerProductPTest) {

    vector<CplxType> lambda_state = std::get<0>(GetParam());
    vector<CplxType> mu_state = std::get<1>(GetParam());
    CplxType expected = std::get<2>(GetParam());

    int length = lambda_state.size();

    Pennylane::StateVector lambda(lambda_state.data(), length);
    Pennylane::StateVector mu(mu_state.data(), length);
    auto res = Pennylane::inner_product(lambda, mu);
    ASSERT_NEAR(expected.real(), res.real(), 1e-10);
    ASSERT_NEAR(expected.imag(), res.imag(), 1e-10);
}

INSTANTIATE_TEST_SUITE_P (
        innerProductTests,
        innerProductFixture,
        ::testing::Values(
                std::make_tuple(vector<CplxType>{1,0}, vector<CplxType>{1,0}, 1),
                std::make_tuple(vector<CplxType>{1,0}, vector<CplxType>{0,1}, 0),
                std::make_tuple(vector<CplxType>{1,0,0,0}, vector<CplxType>{1,0,0,0}, 1),
                std::make_tuple(vector<CplxType>{CplxType(0,1),0}, vector<CplxType>{CplxType(0,1),0}, 1),
                std::make_tuple(vector<CplxType>{0.5,0.25}, vector<CplxType>{0.1,0.2}, 0.05 + 0.05),
                std::make_tuple(
                                vector<CplxType>{CplxType(0.1,0.2), CplxType(0.3,0.4)},
                                vector<CplxType>{CplxType(0.5,0.6), CplxType(0.7,0.8)},
                                CplxType(0.7,-0.08)
                                )
                ));

}
