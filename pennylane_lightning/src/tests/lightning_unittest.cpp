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

const double tol = 1.0e-10f;

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}

TEST(TGate, ApplyToPlus) {

  State_1q input_state(2);
  input_state.setValues({1/SQRT_2, 1/SQRT_2});

  auto operation = T();
  Pairs_1q product_dims = { Pairs(1, 0) };
  State_1q output_state = operation.contract(input_state, product_dims);

  State_1q expected_output_state(2);

  const std::complex<double> exponent(0, -M_PI/4);
  std::complex<double> val = std::pow(M_E, exponent)/SQRT_2;
  expected_output_state.setValues({1/SQRT_2, val});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

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
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_TRUE(output_state_vector.isApprox(expected_vector, tol));
}
}  // namespace two_qubit_ops

namespace auxiliary_functions {

TEST(CalcPerm, OneElemOneQubit) {
    std::vector<int> input_perm({0});
    std::vector<int> output_perm = calc_perm(input_perm, 1);

    EXPECT_TRUE(input_perm == output_perm);
}

TEST(CalcPerm, OneElemTwoQubit) {
    std::vector<int> input_perm({0});
    std::vector<int> output_perm = calc_perm(input_perm, 2);

    std::vector<int> expected_perm({0, 1});

    EXPECT_TRUE(output_perm == expected_perm);
}

TEST(CalcPerm, TwoElemFiveQubitAscOrder) {
    std::vector<int> input_perm({1,2,4});
    std::vector<int> output_perm = calc_perm(input_perm, 5);

    std::vector<int> expected_perm({1,2,4,0,3});

    EXPECT_TRUE(output_perm == expected_perm);
}


TEST(CalcPerm, TwoElemFiveQubitRandomOrder) {
    std::vector<int> input_perm({2,1,4});
    std::vector<int> output_perm = calc_perm(input_perm, 5);

    std::vector<int> expected_perm({2,1,4,0,3});

    EXPECT_TRUE(output_perm == expected_perm);
}

TEST(ArgSort, OneElem) {
    std::vector<int> input_perm({0});
    std::vector<int> output_perm = argsort(input_perm);

    EXPECT_TRUE(input_perm == output_perm);
}

TEST(ArgSort, MultipleAscendingOrder) {
    std::vector<int> input_perm({5,10,15});
    std::vector<int> output_perm = argsort(input_perm);

    std::vector<int> expected({0,1,2});

    EXPECT_TRUE(output_perm == expected);
}

TEST(ArgSort, MultipleRandomOrderUnique) {
    std::vector<int> input_perm({10,15,5});
    std::vector<int> output_perm = argsort(input_perm);

    std::vector<int> expected({2,0,1});

    EXPECT_TRUE(output_perm == expected);
}


TEST(ArgSort, MultipleRandomOrderRepeatedVals) {
    std::vector<int> input_perm({10,15,15,5});
    std::vector<int> output_perm = argsort(input_perm);

    std::vector<int> expected({3,0,1,2});

    EXPECT_TRUE(output_perm == expected);
}
}  // namespace auxiliary_functions
