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

#include <tuple>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "tncuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda::Util;
} // namespace
/// @endcond

TEST_CASE("MPSShapeCheck", "[TNCuda_utils]") {
    SECTION("Correct incoming MPS shape") {
        std::vector<std::vector<std::size_t>> MPS_shape_dest{
            {2, 2}, {2, 2, 4}, {4, 2, 2}, {2, 2}};

        std::vector<std::vector<std::size_t>> MPS_shape_source{
            {2, 2}, {2, 2, 4}, {4, 2, 2}, {2, 2}};

        REQUIRE_NOTHROW(MPSShapeCheck(MPS_shape_dest, MPS_shape_source));
    }

    SECTION("Incorrect incoming MPS shape, bond dimension") {
        std::vector<std::vector<std::size_t>> MPS_shape_dest{
            {2, 2}, {2, 2, 4}, {4, 2, 2}, {2, 2}};

        std::vector<std::vector<std::size_t>> incorrect_MPS_shape{
            {2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2}};

        REQUIRE_THROWS_WITH(
            MPSShapeCheck(MPS_shape_dest, incorrect_MPS_shape),
            Catch::Matchers::Contains("The incoming MPS does not have the "
                                      "correct layout for lightning.tensor"));
    }
    SECTION("Incorrect incoming MPS shape, physical dimension") {
        std::vector<std::vector<std::size_t>> MPS_shape_dest{
            {2, 2}, {2, 2, 4}, {4, 2, 2}, {2, 2}};

        std::vector<std::vector<std::size_t>> incorrect_shape{
            {4, 2}, {2, 4, 4}, {4, 4, 2}, {2, 4}};

        REQUIRE_THROWS_WITH(
            MPSShapeCheck(MPS_shape_dest, incorrect_shape),
            Catch::Matchers::Contains("The incoming MPS does not have the "
                                      "correct layout for lightning.tensor"));
    }
    SECTION("Incorrect incoming MPS shape, number sites") {
        std::vector<std::vector<std::size_t>> MPS_shape_dest{
            {2, 2}, {2, 2, 4}, {4, 2, 2}, {2, 2}};

        std::vector<std::vector<std::size_t>> incorrect_shape{
            {2, 2}, {2, 2, 2}, {2, 2}};

        REQUIRE_THROWS_WITH(
            MPSShapeCheck(MPS_shape_dest, incorrect_shape),
            Catch::Matchers::Contains("The incoming MPS does not have the "
                                      "correct layout for lightning.tensor"));
    }
}
