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

namespace some_collection_of_tests {


TEST(PauliX, ApplyToZero) {

  Eigen::Tensor<std::complex<double>,1> InputState(2);
  InputState.setValues({1, 0});

  auto operation = X();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> OutputState = operation.contract(InputState, product_dims);

  Eigen::Tensor<std::complex<double>,1> ExpectedOutputState(2);
  ExpectedOutputState.setValues({0, 1});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> mis(OutputState.data(), OutputState.size());
  Eigen::Map<Eigen::VectorXcd> mos(ExpectedOutputState.data(), ExpectedOutputState.size());

  EXPECT_EQ(mis.isApprox(mos, tol), 1);
}

TEST(PauliX, ApplyToPlus) {

  double SQRT_2 = sqrt(2);
  Eigen::Tensor<std::complex<double>,1> InputState(2);
  InputState.setValues({1/SQRT_2, 1/SQRT_2});

  auto operation = X();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> OutputState = operation.contract(InputState, product_dims);

  Eigen::Tensor<std::complex<double>,1> ExpectedOutputState(2);
  ExpectedOutputState.setValues({1/SQRT_2, 1/SQRT_2});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> mis(OutputState.data(), OutputState.size());
  Eigen::Map<Eigen::VectorXcd> mos(ExpectedOutputState.data(), ExpectedOutputState.size());

  EXPECT_EQ(mis.isApprox(mos, tol), 1);
}

TEST(PauliY, ApplyToZeroTestCase) {

  Eigen::Tensor<std::complex<double>,1> InputState(2);
  InputState.setValues({1, 0});

  auto operation = Y();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> OutputState = operation.contract(InputState, product_dims);

  Eigen::Tensor<std::complex<double>,1> ExpectedOutputState(2);
  std::complex<double> Val(0, 1);
  ExpectedOutputState.setValues({0, Val});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> mis(OutputState.data(), OutputState.size());
  Eigen::Map<Eigen::VectorXcd> mos(ExpectedOutputState.data(), ExpectedOutputState.size());

  EXPECT_EQ(mis.isApprox(mos, tol), 1);
}

TEST(PauliY, ApplyToPlus) {

  double SQRT_2 = sqrt(2);
  Eigen::Tensor<std::complex<double>,1> InputState(2);
  InputState.setValues({1/SQRT_2, 1/SQRT_2});

  auto operation = Y();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> OutputState = operation.contract(InputState, product_dims);

  Eigen::Tensor<std::complex<double>,1> ExpectedOutputState(2);

  std::complex<double> Fst(0, -1/SQRT_2);
  std::complex<double> Snd(0, 1/SQRT_2);
  ExpectedOutputState.setValues({Fst, Snd});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> mis(OutputState.data(), OutputState.size());
  Eigen::Map<Eigen::VectorXcd> mos(ExpectedOutputState.data(), ExpectedOutputState.size());

  EXPECT_EQ(mis.isApprox(mos, tol), 1);
}

TEST(PauliZ, ApplyToZero) {

  Eigen::Tensor<std::complex<double>,1> InputState(2);
  InputState.setValues({1, 0});

  auto operation = Z();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> OutputState = operation.contract(InputState, product_dims);

  Eigen::Tensor<std::complex<double>,1> ExpectedOutputState(2);
  ExpectedOutputState.setValues({1, 0});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> mis(OutputState.data(), OutputState.size());
  Eigen::Map<Eigen::VectorXcd> mos(ExpectedOutputState.data(), ExpectedOutputState.size());

  EXPECT_EQ(mis.isApprox(mos, tol), 1);
}

TEST(PauliZ, ApplyToPlus) {

  double SQRT_2 = sqrt(2);
  Eigen::Tensor<std::complex<double>,1> InputState(2);
  InputState.setValues({1/SQRT_2, 1/SQRT_2});

  auto operation = Z();
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
  Eigen::Tensor<std::complex<double>, 1> OutputState = operation.contract(InputState, product_dims);

  Eigen::Tensor<std::complex<double>,1> ExpectedOutputState(2);
  ExpectedOutputState.setValues({1/SQRT_2, -1/SQRT_2});

  // Casting to a vector for comparison
  Eigen::Map<Eigen::VectorXcd> mis(OutputState.data(), OutputState.size());
  Eigen::Map<Eigen::VectorXcd> mos(ExpectedOutputState.data(), ExpectedOutputState.size());

  EXPECT_EQ(mis.isApprox(mos, tol), 1);
}

}  // namespace some_collection_of_tests
