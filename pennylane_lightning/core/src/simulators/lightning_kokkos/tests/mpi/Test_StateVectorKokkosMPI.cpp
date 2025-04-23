// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <complex>
#include <limits> // numeric_limits
#include <random>
#include <type_traits>
#include <vector>

#include <catch2/catch.hpp>

#include "StateVectorKokkosMPI.hpp"
#include "TestHelpers.hpp" // createRandomStateVectorData
#include "mpi.h"

/**
 * @file
 *  Tests for functionality for the class StateVectorKokkosMPI.
 */

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::Util;

using Pennylane::Util::isApproxEqual;
using Pennylane::Util::randomUnitary;

std::mt19937_64 re{1337};
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("StateVectorKokkosMPI::Constructibility",
                   "[Default Constructibility]", StateVectorKokkosMPI<>) {
    SECTION("StateVectorBackend<>") {
        REQUIRE(!std::is_constructible_v<TestType>);
    }
}
