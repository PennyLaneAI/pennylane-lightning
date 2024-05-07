
// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <complex>
#include <vector>

#include "NDPermuter.hpp"
#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Util;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("Util::DefaultPermuter::Constructibility",
                   "[Default Constructibility]", DefaultPermuter<>,
                   DefaultPermuter<8>) {
    SECTION("DefaultPermuter") { REQUIRE(std::is_constructible_v<TestType>); }
}
