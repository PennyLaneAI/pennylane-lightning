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

TEST_CASE("swap_op_wires_queue", "[TNCuda_utils]") {
    SECTION("is_wires_local: true") {
        std::vector<std::size_t> wires = {0, 1, 2, 3};
        REQUIRE(is_wires_local(wires) == true);
    }

    SECTION("is_wires_local: false") {
        std::vector<std::size_t> wires = {0, 1, 3, 4};
        REQUIRE(is_wires_local(wires) == false);
    }

    SECTION("swap_op_wires_queue: local") {
        std::vector<std::size_t> wires = {0, 1, 2, 3};
        auto [target_wires, swap_wires_queue] =
            create_swap_wire_pair_queue(wires);
        REQUIRE(wires == target_wires);
        REQUIRE(swap_wires_queue.empty() == true);
    }

    SECTION("swap_op_wires_queue: non-local [0,1,n_wires-1]") {
        std::vector<std::size_t> wires = {0, 1, 4};

        std::vector<std::size_t> target_wires_ref = {0, 1, 2};
        std::vector<std::vector<std::size_t>> swap_wires_queue_ref = {{4, 3},
                                                                      {3, 2}};
        auto [local_wires, swap_wires_queue] =
            create_swap_wire_pair_queue(wires);
        REQUIRE(local_wires == target_wires_ref);
        REQUIRE(swap_wires_queue.size() == 1);
        REQUIRE(swap_wires_queue[0] == swap_wires_queue_ref);
    }

    SECTION("swap_op_wires_queue: non-local [0,n_wires-2,n_wires-1]") {
        std::vector<std::size_t> wires = {0, 3, 4};

        std::vector<std::size_t> target_wires_ref = {2, 3, 4};
        std::vector<std::vector<std::size_t>> swap_wires_queue_ref = {{0, 1},
                                                                      {1, 2}};
        auto [local_wires, swap_wires_queue] =
            create_swap_wire_pair_queue(wires);
        REQUIRE(local_wires == target_wires_ref);
        REQUIRE(swap_wires_queue.size() == 1);
        REQUIRE(swap_wires_queue[0] == swap_wires_queue_ref);
    }

    SECTION("swap_op_wires_queue: non-local [0,n_wires/2,n_wires-1]") {
        std::vector<std::size_t> wires = {0, 2, 4};
        std::vector<std::size_t> target_wires_ref = {1, 2, 3};
        std::vector<std::vector<std::size_t>> swap_wires_queue_ref0 = {{0, 1}};
        std::vector<std::vector<std::size_t>> swap_wires_queue_ref1 = {{4, 3}};
        auto [local_wires, swap_wires_queue] =
            create_swap_wire_pair_queue(wires);
        REQUIRE(local_wires == target_wires_ref);
        REQUIRE(swap_wires_queue.size() == 2);
        REQUIRE(swap_wires_queue[0] == swap_wires_queue_ref0);
        REQUIRE(swap_wires_queue[1] == swap_wires_queue_ref1);
    }
}
