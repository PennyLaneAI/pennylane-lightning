// Copyright 2020 Xanadu Quantum Technologies Inc.

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
#include "../operations.hpp"
#include "../lightning_qubit.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <math.h>       /* sqrt */

const double tol = 1.0e-6f;

using Matrix_2q = Eigen::Matrix<std::complex<double>, 4, 4>;
using Vector_3q = Eigen::Matrix<std::complex<double>, 8, 1>;

template<class State>
Eigen::VectorXcd vectorize(State state) {
    Eigen::Map<Eigen::VectorXcd> out(state.data(), state.size());
    return out;
}

namespace one_qubit_ops {


TEST(PauliX, ApplyToZero) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    auto operation = X();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    expected_output_state.setValues({0, 1});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(PauliX, ApplyToPlus) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, 1/SQRT_2});

    auto operation = X();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    expected_output_state.setValues({1/SQRT_2, 1/SQRT_2});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(PauliY, ApplyToZero) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    auto operation = Y();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    std::complex<double> Val(0, 1);
    expected_output_state.setValues({0, Val});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(PauliY, ApplyToPlus) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, 1/SQRT_2});

    auto operation = Y();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    std::complex<double> first(0, -1/SQRT_2);
    std::complex<double> second(0, 1/SQRT_2);
    expected_output_state.setValues({first, second});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(PauliZ, ApplyToZero) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    auto operation = Z();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    expected_output_state.setValues({1, 0});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(PauliZ, ApplyToPlus) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, 1/SQRT_2});

    auto operation = Z();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    expected_output_state.setValues({1/SQRT_2, -1/SQRT_2});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(Hadamard, ApplyToZero) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    auto operation = H();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    expected_output_state.setValues({1/SQRT_2, 1/SQRT_2});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(Hadamard, ApplyToMinus) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, -1/SQRT_2});

    auto operation = H();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    expected_output_state.setValues({0, 1});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(SGate, ApplyToZero) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    auto operation = S();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    expected_output_state.setValues({1, 0});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(SGate, ApplyToPlus) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, 1/SQRT_2});

    auto operation = S();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    std::complex<double> imag_val(0, 1/SQRT_2);
    expected_output_state.setValues({1/SQRT_2, imag_val});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(TGate, ApplyToZero) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    auto operation = T();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    expected_output_state.setValues({1, 0});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(TGate, ApplyToPlus) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, 1/SQRT_2});

    auto operation = T();
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    const std::complex<double> exponent(0, M_PI/4);
    std::complex<double> val = std::pow(M_E, exponent)/SQRT_2;
    expected_output_state.setValues({1/SQRT_2, val});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(RXGate, ApplyToZeroPiHalf) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    double par = M_PI/2;

    auto operation = RX(par);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    const std::complex<double> first(1/SQRT_2, 0);
    const std::complex<double> second(0, -1/SQRT_2);
    expected_output_state.setValues({first, second});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(RXGate, ApplyToZeroPi) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    double par = M_PI;

    auto operation = RX(par);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    const std::complex<double> second(0, -1);
    expected_output_state.setValues({0, second});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}


TEST(RXGate, ApplyToPlusPiHalf) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, 1/SQRT_2});

    double par = M_PI/2;

    auto operation = RX(par);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    const std::complex<double> val(0.5, -0.5);
    expected_output_state.setValues({val, val});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}


TEST(RYGate, ApplyToZeroPiHalf) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    double par = M_PI/2;

    auto operation = RY(par);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    expected_output_state.setValues({1/SQRT_2, 1/SQRT_2});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(RYGate, ApplyToZeroPi) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    double par = M_PI;

    auto operation = RY(par);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    expected_output_state.setValues({0, 1});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}


TEST(RYGate, ApplyToPlusPiHalf) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, 1/SQRT_2});

    double par = M_PI/2;

    auto operation = RY(par);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    expected_output_state.setValues({0, 1});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}


TEST(RZGate, ApplyToZeroPiHalf) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    double par = M_PI/2;

    auto operation = RZ(par);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    std::complex<double> val(1/SQRT_2, -1/SQRT_2);
    expected_output_state.setValues({val, 0});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(RZGate, ApplyToOnePi) {

    State_1q input_state(2);
    input_state.setValues({0, 1});

    double par = M_PI;

    auto operation = RZ(par);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    const std::complex<double> val(0, 1);
    expected_output_state.setValues({0, val});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}


TEST(RZGate, ApplyToPlusHalfPi) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, 1/SQRT_2});

    double par = M_PI/2;

    auto operation = RZ(par);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);
    const std::complex<double> first(0.5, -0.5);
    const std::complex<double> second(0.5, 0.5);
    expected_output_state.setValues({first, second});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}


TEST(RotGate, ApplyToZeroPiHalfZeroZero) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    const double par = M_PI/2;

    auto operation = Rot(par, 0, 0);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    std::complex<double> val(1/SQRT_2, -1/SQRT_2);
    expected_output_state.setValues({val, 0});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}


TEST(RotGate, ApplyToZeroZeroPiHalfZero) {

    State_1q input_state(2);
    input_state.setValues({1, 0});

    const double par = M_PI/2;

    auto operation = Rot(0, par, 0);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    expected_output_state.setValues({1/SQRT_2, 1/SQRT_2});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}


TEST(RotGate, ApplyToPlusZeroZeroPiHalf) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, 1/SQRT_2});

    const double par = M_PI/2;

    auto operation = Rot(0, 0, par);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    std::complex<double> val1(0.5, -0.5);
    std::complex<double> val2(0.5, 0.5);
    expected_output_state.setValues({val1, val2});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(RotGate, ApplyToZeroPiHalfNegPiHalfPiHalf) {

    State_1q input_state(2);
    input_state.setValues({1,0});

    const double par1 = M_PI/2;
    const double par2 = -M_PI/2;
    const double par3 = M_PI/2;

    auto operation = Rot(par1, par2, par3);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    std::complex<double> val(0, -1/SQRT_2);
    expected_output_state.setValues({val, -1/SQRT_2});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}


TEST(RotGate, ApplyToPlusNegPiHalfPiPi) {

    State_1q input_state(2);
    input_state.setValues({1/SQRT_2, 1/SQRT_2});

    const double par1 = -M_PI/2;
    const double par2 = M_PI;
    const double par3 = M_PI;

    auto operation = Rot(par1, par2, par3);
    Pairs_1q product_dims = { Pairs(1, 0) };
    State_1q output_state = operation.contract(input_state, product_dims);

    State_1q expected_output_state(2);

    std::complex<double> val1(0.5, 0.5);
    std::complex<double> val2(-0.5, 0.5);
    expected_output_state.setValues({val1, val2});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(PhaseShift, ApplyToZeroAndOne) {
    State_1q input_state_0(2);
    State_1q input_state_1(2);
    input_state_0.setValues({1, 0});
    input_state_1.setValues({0, 1});


    std::complex<double> const1(0.99500417, 0.09983342);
    State_1q expected_state_0(2);
    State_1q expected_state_1(2);
    expected_state_0.setValues({1, 0});
    expected_state_1.setValues({0, const1});

    vector<int> w{0};
    vector<float> p{0.1};
    auto output_state_0 = contract_1q_op(input_state_0, "PhaseShift", w, p);
    auto output_state_1 = contract_1q_op(input_state_1, "PhaseShift", w, p);

    // Casting to a vector for comparison
    auto expected_state_vector_0 = vectorize(expected_state_0);
    auto expected_state_vector_1 = vectorize(expected_state_1);
    auto output_state_vector_0 = vectorize(output_state_0);
    auto output_state_vector_1 = vectorize(output_state_1);

    EXPECT_TRUE(expected_state_vector_0.isApprox(output_state_vector_0, tol));
    EXPECT_TRUE(expected_state_vector_1.isApprox(output_state_vector_1, tol));
}

}  // namespace one_qubit_ops

namespace two_qubit_ops {


TEST(CNOT, ApplyToZero) {

    State_2q input_state(2,2);
    input_state.setValues({{1, 0},{0,0}});

    auto operation = CNOT();
    Pairs_2q product_dims = { Pairs(2, 0), Pairs(3, 1) };
    State_2q output_state = operation.contract(input_state, product_dims);

    State_2q expected_output_state(2,2);
    expected_output_state.setValues({{1, 0},{0,0}});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(CNOT, ApplyToOneZero) {

    State_2q input_state(2,2);
    input_state.setValues({{0, 0},{1,0}});

    auto operation = CNOT();
    Pairs_2q product_dims = { Pairs(2, 0), Pairs(3, 1) };
    State_2q output_state = operation.contract(input_state, product_dims);

    State_2q expected_output_state(2,2);
    expected_output_state.setValues({{0, 0},{0,1}});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(CNOT, ApplyToBellState) {

    State_2q input_state(2,2);
    input_state.setValues({{1/SQRT_2,0},{0, 1/SQRT_2}});

    auto operation = CNOT();
    Pairs_2q product_dims = { Pairs(2, 0), Pairs(3, 1) };
    State_2q output_state = operation.contract(input_state, product_dims);

    State_2q expected_output_state(2,2);
    expected_output_state.setValues({{1/SQRT_2,0},{1/SQRT_2, 0}});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(CNOT, ApplyToThreeQubitControlThird) {

    State_3q input_state(2,2,2);
    input_state.setValues({{{0,0},{0,0}},{{0,0},{0,1}}});

    auto operation = CNOT();
    Pairs_2q product_dims = { Pairs(3, 2), Pairs(2, 1) };
    State_3q output_state = operation.contract(input_state, product_dims);

    State_3q expected_output_state(2,2,2);

    // The output dimensions are ordered according to the output of the tensor
    // contraction (no shuffling takes place)
    expected_output_state.setValues({{{0,0},{0,0}},{{0,1},{0,0}}});

    // Casting to a vector for comparison
    auto output_state_vector = vectorize(output_state);
    auto expected_vector = vectorize(expected_output_state);

    EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(SWAP, ToMatrix) {

    auto operation = SWAP();
    auto operation_matrix = Map<Matrix_2q> (operation.data());

    Matrix_2q target_matrix;
    target_matrix << 1, 0, 0, 0,
                  0, 0, 1, 0,
                  0, 1, 0, 0,
                  0, 0, 0, 1;

    EXPECT_TRUE(operation_matrix.isApprox(target_matrix, tol));

}

TEST(CZ, ToMatrix) {

    auto operation = CZ();
    auto operation_matrix = Map<Matrix_2q> (operation.data());

    Matrix_2q target_matrix;
    target_matrix << 1, 0, 0, 0,
                  0, 1, 0, 0,
                  0, 0, 1, 0,
                  0, 0, 0, -1;

    EXPECT_TRUE(operation_matrix.isApprox(target_matrix, tol));

}

TEST(CRots, ApplyTo00) {

    State_2q input_state(2,2);
    input_state.setValues({{1, 0},{0, 0}});

    State_2q expected_output_state(2,2);
    expected_output_state.setValues({{1, 0},{0, 0}});

    vector<int> w{0, 1};
    vector<float> p{0.1};
    auto output_state_X = contract_2q_op(input_state, "CRX", w, p);
    auto output_state_Y = contract_2q_op(input_state, "CRY", w, p);
    auto output_state_Z = contract_2q_op(input_state, "CRZ", w, p);

    vector<int> w2{0, 1};
    vector<float> p2{0.1, 0.2, 0.3};
    auto output_state_Rot = contract_2q_op(input_state, "CRot", w2, p2);

    // Casting to a vector for comparison
    auto output_state_vector_X = vectorize(output_state_X);
    auto output_state_vector_Y = vectorize(output_state_Y);
    auto output_state_vector_Z = vectorize(output_state_Z);
    auto expected_vector = vectorize(expected_output_state);
    auto output_state_vector_Rot = vectorize(output_state_Rot);

    EXPECT_TRUE(expected_vector.isApprox(output_state_vector_X, tol));
    EXPECT_TRUE(expected_vector.isApprox(output_state_vector_Y, tol));
    EXPECT_TRUE(expected_vector.isApprox(output_state_vector_Z, tol));
    EXPECT_TRUE(expected_vector.isApprox(output_state_vector_Rot, tol));
}

TEST(CRots, ApplyTo01) {

    State_2q input_state(2,2);
    input_state.setValues({{0, 1},{0, 0}});

    State_2q expected_output_state(2,2);
    expected_output_state.setValues({{0, 1},{0, 0}});

    vector<int> w{0, 1};
    vector<float> p{0.1};
    auto output_state_X = contract_2q_op(input_state, "CRX", w, p);
    auto output_state_Y = contract_2q_op(input_state, "CRY", w, p);
    auto output_state_Z = contract_2q_op(input_state, "CRZ", w, p);

    vector<int> w2{0, 1};
    vector<float> p2{0.1, 0.2, 0.3};
    auto output_state_Rot = contract_2q_op(input_state, "CRot", w2, p2);

    // Casting to a vector for comparison
    auto output_state_vector_X = vectorize(output_state_X);
    auto output_state_vector_Y = vectorize(output_state_Y);
    auto output_state_vector_Z = vectorize(output_state_Z);
    auto expected_vector = vectorize(expected_output_state);
    auto output_state_vector_Rot = vectorize(output_state_Rot);

    EXPECT_TRUE(expected_vector.isApprox(output_state_vector_X, tol));
    EXPECT_TRUE(expected_vector.isApprox(output_state_vector_Y, tol));
    EXPECT_TRUE(expected_vector.isApprox(output_state_vector_Z, tol));
    EXPECT_TRUE(expected_vector.isApprox(output_state_vector_Rot, tol));
}

TEST(CRots, ApplyTo10) {

    State_2q input_state(2,2);
    input_state.setValues({{0, 0},{1, 0}});

    float phi(0.1);
    auto cos = std::cos(phi / 2);
    auto sin = std::sin(phi / 2);

    complex<double> cos_real(cos, 0);
    complex<double> sin_imag(0, sin);
    complex<double> sin_real(sin, 0);

    State_2q expected_output_state_X(2,2);
    expected_output_state_X.setValues({{0, 0},{cos_real, -sin_imag}});

    State_2q expected_output_state_Y(2,2);
    expected_output_state_Y.setValues({{0, 0},{cos_real, sin_real}});

    State_2q expected_output_state_Z(2,2);
    expected_output_state_Z.setValues({{0, 0},{cos_real-sin_imag, 0}});

    vector<float> p2{0.4, 0.1, 0.3};
    complex<double> imag_phi_plus_omega(0, -(p2[0] + p2[2]) / 2);
    complex<double> imag_phi_minus_omega(0, -(p2[0] - p2[2]) / 2);
    auto exp_plus = std::pow(M_E, imag_phi_plus_omega);
    auto exp_minus = std::pow(M_E, imag_phi_minus_omega);

    State_2q expected_output_state_Rot(2,2);
    expected_output_state_Rot.setValues({{0, 0},{cos_real * exp_plus, sin_real * exp_minus}});

    vector<int> w{0, 1};
    vector<float> p{phi};
    auto output_state_X = contract_2q_op(input_state, "CRX", w, p);
    auto output_state_Y = contract_2q_op(input_state, "CRY", w, p);
    auto output_state_Z = contract_2q_op(input_state, "CRZ", w, p);

    vector<int> w2{0, 1};
    auto output_state_Rot = contract_2q_op(input_state, "CRot", w2, p2);

    // Casting to a vector for comparison
    auto output_state_vector_X = vectorize(output_state_X);
    auto output_state_vector_Y = vectorize(output_state_Y);
    auto output_state_vector_Z = vectorize(output_state_Z);
    auto output_state_vector_Rot = vectorize(output_state_Rot);

    auto expected_vector_X = vectorize(expected_output_state_X);
    auto expected_vector_Y = vectorize(expected_output_state_Y);
    auto expected_vector_Z = vectorize(expected_output_state_Z);
    auto expected_vector_Rot = vectorize(expected_output_state_Rot);

    EXPECT_TRUE(expected_vector_X.isApprox(output_state_vector_X, tol));
    EXPECT_TRUE(expected_vector_Y.isApprox(output_state_vector_Y, tol));
    EXPECT_TRUE(expected_vector_Z.isApprox(output_state_vector_Z, tol));
    EXPECT_TRUE(expected_vector_Rot.isApprox(output_state_vector_Rot, tol));
}

TEST(CRots, ApplyTo11) {

    State_2q input_state(2,2);
    input_state.setValues({{0, 0},{0, 1}});

    float phi(0.1);
    auto cos = std::cos(phi / 2);
    auto sin = std::sin(phi / 2);

    complex<double> cos_real(cos, 0);
    complex<double> sin_imag(0, sin);
    complex<double> sin_real(sin, 0);

    State_2q expected_output_state_X(2,2);
    expected_output_state_X.setValues({{0, 0},{-sin_imag, cos_real}});

    State_2q expected_output_state_Y(2,2);
    expected_output_state_Y.setValues({{0, 0},{-sin_real, cos_real}});

    State_2q expected_output_state_Z(2,2);
    expected_output_state_Z.setValues({{0, 0},{0, cos_real+sin_imag}});

    vector<float> p2{0.4, 0.1, 0.3};
    complex<double> imag_phi_plus_omega(0, (p2[0] + p2[2]) / 2);
    complex<double> imag_phi_minus_omega(0, (p2[0] - p2[2]) / 2);
    auto exp_plus = std::pow(M_E, imag_phi_plus_omega);
    auto exp_minus = std::pow(M_E, imag_phi_minus_omega);

    State_2q expected_output_state_Rot(2,2);
    expected_output_state_Rot.setValues({{0, 0},{-sin_real * exp_minus, cos_real * exp_plus}});

    vector<int> w{0, 1};
    vector<float> p{phi};
    auto output_state_X = contract_2q_op(input_state, "CRX", w, p);
    auto output_state_Y = contract_2q_op(input_state, "CRY", w, p);
    auto output_state_Z = contract_2q_op(input_state, "CRZ", w, p);

    vector<int> w2{0, 1};
    auto output_state_Rot = contract_2q_op(input_state, "CRot", w2, p2);

    // Casting to a vector for comparison
    auto output_state_vector_X = vectorize(output_state_X);
    auto output_state_vector_Y = vectorize(output_state_Y);
    auto output_state_vector_Z = vectorize(output_state_Z);
    auto output_state_vector_Rot = vectorize(output_state_Rot);

    auto expected_vector_X = vectorize(expected_output_state_X);
    auto expected_vector_Y = vectorize(expected_output_state_Y);
    auto expected_vector_Z = vectorize(expected_output_state_Z);
    auto expected_vector_Rot = vectorize(expected_output_state_Rot);

    EXPECT_TRUE(expected_vector_X.isApprox(output_state_vector_X, tol));
    EXPECT_TRUE(expected_vector_Y.isApprox(output_state_vector_Y, tol));
    EXPECT_TRUE(expected_vector_Z.isApprox(output_state_vector_Z, tol));
    EXPECT_TRUE(expected_vector_Rot.isApprox(output_state_vector_Rot, tol));

}

}  // namespace two_qubit_ops

namespace three_qubit_ops {

TEST(Toffoli, ApplyToAll) {
    State_3q input_state_000(2, 2, 2);
    input_state_000.setValues({{{1, 0},{0, 0}}, {{0, 0},{0, 0}}});
    State_3q input_state_001(2, 2, 2);
    input_state_001.setValues({{{0, 1},{0, 0}}, {{0, 0},{0, 0}}});
    State_3q input_state_010(2, 2, 2);
    input_state_010.setValues({{{0, 0},{1, 0}}, {{0, 0},{0, 0}}});
    State_3q input_state_011(2, 2, 2);
    input_state_011.setValues({{{0, 0},{0, 1}}, {{0, 0},{0, 0}}});
    State_3q input_state_100(2, 2, 2);
    input_state_100.setValues({{{0, 0},{0, 0}}, {{1, 0},{0, 0}}});
    State_3q input_state_101(2, 2, 2);
    input_state_101.setValues({{{0, 0},{0, 0}}, {{0, 1},{0, 0}}});
    State_3q input_state_110(2, 2, 2);
    input_state_110.setValues({{{0, 0},{0, 0}}, {{0, 0},{1, 0}}});
    State_3q input_state_111(2, 2, 2);
    input_state_111.setValues({{{0, 0},{0, 0}}, {{0, 0},{0, 1}}});

    std::vector<State_3q> input_states{
        input_state_000,
        input_state_001,
        input_state_010,
        input_state_011,
        input_state_100,
        input_state_101,
        input_state_110,
        input_state_111,
    };
    std::vector<State_3q> output_states;

    vector<int> w{0, 1, 2};

    for (int i=0; i < 8; i++) {
        output_states.push_back(contract_3q_op(input_states[i], "Toffoli", w));
    }

    State_3q target_state_110(2, 2, 2);
    target_state_110.setValues({{{0, 0},{0, 0}}, {{0, 0},{0, 1}}});
    State_3q target_state_111(2, 2, 2);
    target_state_111.setValues({{{0, 0},{0, 0}}, {{0, 0},{1, 0}}});

    auto expected_states = input_states;
    expected_states[6] = target_state_110;
    expected_states[7] = target_state_111;

    for (int i=0; i < 8; i++) {
        auto expected_vector = vectorize(expected_states[i]);
        auto output_state = vectorize(output_states[i]);
        EXPECT_TRUE(expected_vector.isApprox(output_state, tol));
    }
}

TEST(CSWAP, ApplyToAll) {
    State_3q input_state_000(2, 2, 2);
    input_state_000.setValues({{{1, 0},{0, 0}}, {{0, 0},{0, 0}}});
    State_3q input_state_001(2, 2, 2);
    input_state_001.setValues({{{0, 1},{0, 0}}, {{0, 0},{0, 0}}});
    State_3q input_state_010(2, 2, 2);
    input_state_010.setValues({{{0, 0},{1, 0}}, {{0, 0},{0, 0}}});
    State_3q input_state_011(2, 2, 2);
    input_state_011.setValues({{{0, 0},{0, 1}}, {{0, 0},{0, 0}}});
    State_3q input_state_100(2, 2, 2);
    input_state_100.setValues({{{0, 0},{0, 0}}, {{1, 0},{0, 0}}});
    State_3q input_state_101(2, 2, 2);
    input_state_101.setValues({{{0, 0},{0, 0}}, {{0, 1},{0, 0}}});
    State_3q input_state_110(2, 2, 2);
    input_state_110.setValues({{{0, 0},{0, 0}}, {{0, 0},{1, 0}}});
    State_3q input_state_111(2, 2, 2);
    input_state_111.setValues({{{0, 0},{0, 0}}, {{0, 0},{0, 1}}});

    std::vector<State_3q> input_states{
        input_state_000,
        input_state_001,
        input_state_010,
        input_state_011,
        input_state_100,
        input_state_101,
        input_state_110,
        input_state_111,
    };
    std::vector<State_3q> output_states;

    vector<int> w{0, 1, 2};

    for (int i=0; i < 8; i++) {
        output_states.push_back(contract_3q_op(input_states[i], "CSWAP", w));
    }

    State_3q target_state_101(2, 2, 2);
    target_state_101.setValues({{{0, 0},{0, 0}}, {{0, 0},{1, 0}}});
    State_3q target_state_110(2, 2, 2);
    target_state_110.setValues({{{0, 0},{0, 0}}, {{0, 1},{0, 0}}});

    auto expected_states = input_states;
    expected_states[5] = target_state_101;
    expected_states[6] = target_state_110;

    for (int i=0; i < 8; i++) {
        auto expected_vector = vectorize(expected_states[i]);
        auto output_state = vectorize(output_states[i]);
        EXPECT_TRUE(expected_vector.isApprox(output_state, tol));
    }
}

} // namespace three_qubit_ops

namespace auxiliary_functions {

TEST(CalcTensInd, OneWireOneQubit) {
    std::vector<int> tensor_indices({0});
    std::vector<int> wires({0});
    std::vector<int> output_tensor_indices = calculate_tensor_indices(wires, tensor_indices);

    EXPECT_TRUE(tensor_indices == output_tensor_indices);
}

TEST(CalcTensInd, OneWireTwoQubit) {
    std::vector<int> tensor_indices({0, 1});
    std::vector<int> wires({0});
    std::vector<int> output_tensor_indices = calculate_tensor_indices(wires, tensor_indices);

    std::vector<int> expected_tensor_indices({0, 1});

    EXPECT_TRUE(expected_tensor_indices == output_tensor_indices);
}

TEST(CalcTensInd, TwoWireFiveQubitAscOrder) {
    std::vector<int> tensor_indices({0, 1, 2, 3, 4});
    std::vector<int> wires({1,2,4});
    std::vector<int> output_tensor_indices = calculate_tensor_indices(wires, tensor_indices);

    std::vector<int> expected_tensor_indices({1, 2, 4, 0, 3});

    EXPECT_TRUE(expected_tensor_indices == output_tensor_indices);
}


TEST(CalcTensInd, TwoWireFiveQubitRandomOrder) {
    std::vector<int> tensor_indices({0, 1, 2, 3, 4});
    std::vector<int> wires({2, 1, 4});
    std::vector<int> output_tensor_indices = calculate_tensor_indices(wires, tensor_indices);

    std::vector<int> expected_tensor_indices({2, 1, 4, 0, 3});

    EXPECT_TRUE(expected_tensor_indices == output_tensor_indices);

}

TEST(CalcTensInd, TwoWireFiveQubitRandomOrderReverse) {
    std::vector<int> tensor_indices({4, 3, 2, 1, 0});
    std::vector<int> wires({2, 1, 4});
    std::vector<int> output_tensor_indices = calculate_tensor_indices(wires, tensor_indices);

    std::vector<int> expected_tensor_indices({2, 1, 4, 3, 0});

    EXPECT_TRUE(expected_tensor_indices == output_tensor_indices);
}

TEST(QubitPositions, OneElem) {
    std::vector<int> tensor_indices({0});
    std::vector<int> qubit_positions = calculate_qubit_positions(tensor_indices);

    EXPECT_TRUE(tensor_indices == qubit_positions);
}

TEST(QubitPositions, MultipleAscendingOrder) {
    std::vector<int> tensor_indices({0, 1, 2, 3});
    std::vector<int> qubit_positions = calculate_qubit_positions(tensor_indices);

    std::vector<int> expected_qubit_positions({0, 1, 2, 3});

    EXPECT_TRUE(expected_qubit_positions == qubit_positions);
}

TEST(QubitPositions, MultipleRandomOrderUnique) {
    std::vector<int> tensor_indices({1, 2, 0, 3});
    std::vector<int> qubit_positions = calculate_qubit_positions(tensor_indices);

    std::vector<int> expected_qubit_positions({2, 0, 1, 3});

    EXPECT_TRUE(expected_qubit_positions == qubit_positions);
}


TEST(QubitPositions, MultipleRandomOrderUnique2) {
    std::vector<int> tensor_indices({1, 3, 2, 0});
    std::vector<int> qubit_positions = calculate_qubit_positions(tensor_indices);

    std::vector<int> expected_qubit_positions({3, 0, 2, 1});

    EXPECT_TRUE(expected_qubit_positions == qubit_positions);
}
}  // namespace auxiliary_functions
