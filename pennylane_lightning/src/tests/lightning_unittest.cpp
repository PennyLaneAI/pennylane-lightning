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
#include <unsupported/Eigen/CXX11/Tensor>
#include <math.h>       /* sqrt */

const double tol = 1.0e-10f;

namespace onequbitops {


TEST(PauliX, ApplyToZero) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  auto operation = X();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  expected_output_state.setValues({0, 1});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(PauliX, ApplyToPlus) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1/SQRT_2, 1/SQRT_2});

  auto operation = X();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  expected_output_state.setValues({1/SQRT_2, 1/SQRT_2});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(PauliY, ApplyToZeroTestCase) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  auto operation = Y();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  std::complex<double> Val(0, 1);
  expected_output_state.setValues({0, Val});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(PauliY, ApplyToPlus) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1/SQRT_2, 1/SQRT_2});

  auto operation = Y();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);

  std::complex<double> Fst(0, -1/SQRT_2);
  std::complex<double> Snd(0, 1/SQRT_2);
  expected_output_state.setValues({Fst, Snd});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(PauliZ, ApplyToZero) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  auto operation = Z();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  expected_output_state.setValues({1, 0});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(PauliZ, ApplyToPlus) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1/SQRT_2, 1/SQRT_2});

  auto operation = Z();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  expected_output_state.setValues({1/SQRT_2, -1/SQRT_2});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(Hadamard, ApplyToZero) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  auto operation = H();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  expected_output_state.setValues({1/SQRT_2, 1/SQRT_2});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(Hadamard, ApplyToMinus) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1/SQRT_2, -1/SQRT_2});

  auto operation = H();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  expected_output_state.setValues({0, 1});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(SGate, ApplyToZero) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  auto operation = S();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  expected_output_state.setValues({1, 0});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(SGate, ApplyToPlus) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1/SQRT_2, 1/SQRT_2});

  auto operation = S();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  std::complex<double> negative_imag_val(0, 1/SQRT_2);
  expected_output_state.setValues({1/SQRT_2, negative_imag_val});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(TGate, ApplyToZero) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  auto operation = T();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  expected_output_state.setValues({1, 0});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(TGate, ApplyToPlus) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1/SQRT_2, 1/SQRT_2});

  auto operation = T();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);

  const std::complex<double> exponent(0, -M_PI/4);
  std::complex<double> val = std::pow(M_E, exponent)/SQRT_2;
  expected_output_state.setValues({1/SQRT_2, val});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(RXGate, ApplyToZeroPiHalf) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  double par = M_PI/2;

  auto operation = RX(par);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);

  const std::complex<double> first(1/SQRT_2, 0);
  const std::complex<double> second(0, -1/SQRT_2);
  expected_output_state.setValues({first, second});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(RXGate, ApplyToZeroPi) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  double par = M_PI;

  auto operation = RX(par);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);

  const std::complex<double> second(0, -1);
  expected_output_state.setValues({0, second});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}


TEST(RXGate, ApplyToPlus) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1/SQRT_2, 1/SQRT_2});

  double par = M_PI/2;

  auto operation = RX(par);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);

  const std::complex<double> val(0.5, -0.5);
  expected_output_state.setValues({val, val});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}


TEST(RYGate, ApplyToZeroPiHalf) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  double par = M_PI/2;

  auto operation = RY(par);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);

  expected_output_state.setValues({1/SQRT_2, 1/SQRT_2});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(RYGate, ApplyToZeroPi) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  double par = M_PI;

  auto operation = RY(par);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);

  const std::complex<double> second(0, -1);
  expected_output_state.setValues({0, 1});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}


TEST(RYGate, ApplyToPlus) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1/SQRT_2, 1/SQRT_2});

  double par = M_PI/2;

  auto operation = RY(par);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  expected_output_state.setValues({0, 1});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}


TEST(RZGate, ApplyToZeroPiHalf) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1, 0});

  double par = M_PI/2;

  auto operation = RZ(par);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);

  std::complex<double> val(1/SQRT_2, -1/SQRT_2);
  expected_output_state.setValues({val, 0});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}

TEST(RZGate, ApplyToZeroPi) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({0, 1});

  double par = M_PI;

  auto operation = RZ(par);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);

  const std::complex<double> val(0, 1);
  expected_output_state.setValues({0, val});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}


TEST(RZGate, ApplyToPlus) {

  Eigen::Tensor<std::complex<double>,1> input_state(2);
  input_state.setValues({1/SQRT_2, 1/SQRT_2});

  double par = M_PI/2;

  auto operation = RZ(par);
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> output_state = operation.contract(input_state, product_dims);

  Eigen::Tensor<std::complex<double>,1> expected_output_state(2);
  const std::complex<double> first(0.5, -0.5);
  const std::complex<double> second(0.5, 0.5);
  expected_output_state.setValues({first, second});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> output_state_vector(output_state.data(), output_state.size());
  Eigen::Map<Eigen::VectorXcd> expected_vector(expected_output_state.data(), expected_output_state.size());

  EXPECT_EQ(output_state_vector.isApprox(expected_vector, tol), 1);
}
}  // namespace onequbitops
