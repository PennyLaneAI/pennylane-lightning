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
#include "TestingUtils.hpp"

#include <tuple>

using Pennylane::CplxType;
using std::vector;

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
                                              // matrix, dim, start idx, block matrix, block dim, result
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
                                                 0,0,0,1}),
                std::make_tuple(vector<CplxType>{1,0,0,0,
                                                 0,1,0,0,
                                                 0,0,1,0,
                                                 0,0,0,1}, 4, 0,
                                vector<CplxType>{1,0,0,0,
                                                 0,1,0,0,
                                                 0,0,0,1,
                                                 0,0,1,0}, 4,
                                vector<CplxType>{1,0,0,0,
                                                 0,1,0,0,
                                                 0,0,0,1,
                                                 0,0,1,0})
    ));

class SetBlockThrowsFixture : public ::testing::TestWithParam<std::tuple<vector<CplxType>, size_t, size_t, vector<CplxType>, size_t >> {
};

TEST_P(SetBlockThrowsFixture, OutOfBounds) {
    auto big_mx = std::get<0>(GetParam());
    auto dim = std::get<1>(GetParam());

    auto start_index = std::get<2>(GetParam());

    auto block_mx = std::get<3>(GetParam());
    auto block_dim = std::get<4>(GetParam());

    std::string msg = "The block of the matrix determined by the start index needs to be greater than or equal to the dimension of the submatrix.";
    EXPECT_THROW_WITH_MESSAGE(Pennylane::set_block(big_mx.data(), dim, start_index, block_mx.data(), block_dim), std::invalid_argument, msg);
}

INSTANTIATE_TEST_SUITE_P (
        IncorrectStartingIndexOrSubmatrixSize,
        SetBlockThrowsFixture,
        ::testing::Values(
                std::make_tuple(vector<CplxType>{0,0,0,0}, 2, 2, vector<CplxType>{1,0,0,1}, 2),
                std::make_tuple(vector<CplxType>{0,0,0,0}, 2, 4, vector<CplxType>{1}, 1)
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
