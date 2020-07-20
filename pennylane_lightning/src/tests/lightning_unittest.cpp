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

const double tol = 1.0e-10f;

namespace some_collection_of_tests {

TEST(SomeTestName, TestCase) {
// "SomeTestName" and "TestCase" are hierarchical names given by the person who makes the test
  int initial_val = 141;
  int expected = 142;
  
  int val = initial_val + 1; // test some function

  EXPECT_NEAR(expected, val, tol);
}

}  // namespace some_collection_of_tests

