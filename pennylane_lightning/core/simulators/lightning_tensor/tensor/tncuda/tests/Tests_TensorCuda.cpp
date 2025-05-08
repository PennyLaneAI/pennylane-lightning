// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <catch2/catch.hpp>
#include <cmath>

#include "DevTag.hpp"
#include "TensorCuda.hpp"

#include "TestHelpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Util;
using namespace Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningTensor::TNCuda;
} // namespace
/// @endcond

TEMPLATE_TEST_CASE("TensorCuda::Constructibility", "[Default Constructibility]",
                   float, double) {
    SECTION("TensorCuda<>") {
        REQUIRE(!std::is_constructible_v<TensorCuda<TestType>()>);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("TensorCuda::Constructibility",
                           "[General Constructibility]", (TensorCuda),
                           (float, double)) {
    using TensorT = TestType;

    SECTION("TensorT<TestType>") { REQUIRE(!std::is_constructible_v<TensorT>); }
    SECTION("TensorT<TestType> {const std::size_t, const "
            "std::vector<std::size_t> &, const "
            "std::vector<std::size_t>&, DevTag<int> &}") {
        REQUIRE(std::is_constructible_v<
                TensorT, const std::size_t, const std::vector<std::size_t> &,
                const std::vector<std::size_t> &, DevTag<int> &>);
    }
}

TEMPLATE_TEST_CASE("TensorCuda::baseMethods", "[TensorCuda]", float, double) {
    const std::size_t rank = 3;
    const std::vector<std::size_t> modes = {0, 1, 2};
    const std::vector<std::size_t> extents = {2, 2, 2};
    const std::size_t length = 8;
    DevTag<int> dev_tag{0, 0};

    TensorCuda<TestType> tensor{rank, modes, extents, dev_tag};

    SECTION("getRank()") { CHECK(tensor.getRank() == rank); }

    SECTION("getModes()") { CHECK(tensor.getModes() == modes); }

    SECTION("getExtents()") { CHECK(tensor.getExtents() == extents); }

    SECTION("getLength()") { CHECK(tensor.getLength() == length); }
}
