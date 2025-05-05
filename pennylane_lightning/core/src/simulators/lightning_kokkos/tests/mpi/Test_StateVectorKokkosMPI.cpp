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

/**
 * Test StateVectorKokkosMPI wire-related helper methods
 */
TEMPLATE_TEST_CASE("is_element_in_vector", "[LKMPI]",double, float) {
    StateVectorKokkosMPI<TestType> sv(2);
   std::vector<std::size_t> wires = {0, 1, 4, 5};
   REQUIRE(sv.is_element_in_vector(wires, static_cast<std::size_t>(0)));
   REQUIRE(sv.is_element_in_vector(wires, static_cast<std::size_t>(1)));
   REQUIRE(sv.is_element_in_vector(wires, static_cast<std::size_t>(4)));
   REQUIRE(sv.is_element_in_vector(wires, static_cast<std::size_t>(5)));
   REQUIRE(!sv.is_element_in_vector(wires, static_cast<std::size_t>(2)));

   wires = {};
   REQUIRE(!sv.is_element_in_vector(wires, static_cast<std::size_t>(0)));
}


TEMPLATE_TEST_CASE("get_element_index_in_vector", "[LKMPI]",double, float) {
    StateVectorKokkosMPI<TestType> sv(2);
   std::vector<std::size_t> wires = {0, 1, 4, 5};
   REQUIRE(sv.get_element_index_in_vector(wires, static_cast<std::size_t>(0)) == 0);
   REQUIRE(sv.get_element_index_in_vector(wires, static_cast<std::size_t>(1)) == 1);
   REQUIRE(sv.get_element_index_in_vector(wires, static_cast<std::size_t>(4)) == 2);
   REQUIRE(sv.get_element_index_in_vector(wires, static_cast<std::size_t>(5)) == 3);

   PL_REQUIRE_THROWS_MATCHES(
       sv.get_element_index_in_vector(wires, static_cast<std::size_t>(2)),
       LightningException,"Element not in vector");
}


TEMPLATE_TEST_CASE("Local/Global wire helpers", "[LKMPI]",double, float) {
    StateVectorKokkosMPI<TestType> sv(5);

    // Only run with 4 mpi ranks:
    REQUIRE(sv.get_mpi_size() == 4);

    REQUIRE(sv.get_num_global_wires() == 2);
    REQUIRE(sv.get_num_local_wires() == 3);

    REQUIRE(sv.get_rev_global_wire_index(0) == 1);
    REQUIRE(sv.get_rev_global_wire_index(1) == 0);

    PL_REQUIRE_THROWS_MATCHES(sv.get_rev_global_wire_index(2),LightningException,"Element not in vector");
    PL_REQUIRE_THROWS_MATCHES(sv.get_rev_local_wire_index(1),LightningException,"Element not in vector");

    REQUIRE(sv.get_rev_local_wire_index(2) == 2);
    REQUIRE(sv.get_rev_local_wire_index(3) == 1);
    REQUIRE(sv.get_rev_local_wire_index(4) == 0);

    REQUIRE(sv.get_global_wires() == std::vector<std::size_t>{0, 1});
    REQUIRE(sv.get_local_wires() == std::vector<std::size_t>{2, 3, 4});

    REQUIRE(sv.get_local_wires_indices({2, 4}) == std::vector<std::size_t>{0, 2});
    REQUIRE(sv.get_local_wires_indices({3, 2}) == std::vector<std::size_t>{1, 0});
    PL_REQUIRE_THROWS_MATCHES(sv.get_local_wires_indices({3, 0}),LightningException,"Element not in vector");

    REQUIRE(sv.get_global_wires_indices({0, 1}) == std::vector<std::size_t>{0, 1});
    REQUIRE(sv.get_global_wires_indices({1, 0}) == std::vector<std::size_t>{1, 0});
    PL_REQUIRE_THROWS_MATCHES(sv.get_global_wires_indices({3, 0}),LightningException,"Element not in vector");

    REQUIRE(sv.find_global_wires({1, 3}) == std::vector<std::size_t>{1});
    REQUIRE(sv.find_global_wires({1, 3, 0}) == std::vector<std::size_t>{1, 0});
    REQUIRE(sv.find_global_wires({1, 0}) == std::vector<std::size_t>{1, 0});
    REQUIRE(sv.find_global_wires({1, 0, 5}) == std::vector<std::size_t>{1, 0});

    REQUIRE(sv.is_wires_local({2, 3}) == true);
    REQUIRE(sv.is_wires_local({0, 1}) == false);
    REQUIRE(sv.is_wires_local({2, 0}) == false);
    REQUIRE(sv.is_wires_global({0, 1}) == true);
    REQUIRE(sv.is_wires_global({2, 3}) == false);
    REQUIRE(sv.is_wires_global({2, 0}) == false);

    REQUIRE(sv.get_blk_size() == 8);

    REQUIRE(sv.global_2_local_index(0) == std::pair<std::size_t, std::size_t>{0, 0});
    REQUIRE(sv.global_2_local_index(1) == std::pair<std::size_t, std::size_t>{0, 1});
    REQUIRE(sv.global_2_local_index(7) == std::pair<std::size_t, std::size_t>{0, 7});
    REQUIRE(sv.global_2_local_index(8) == std::pair<std::size_t, std::size_t>{1, 0});
    REQUIRE(sv.global_2_local_index(9) == std::pair<std::size_t, std::size_t>{1, 1});
    REQUIRE(sv.global_2_local_index(30) == std::pair<std::size_t, std::size_t>{3, 6});
    REQUIRE(sv.global_2_local_index(31) == std::pair<std::size_t, std::size_t>{3, 7});
}


TEMPLATE_TEST_CASE("getDataVector", "[LKMPI]",double, float) {
    StateVectorKokkosMPI<TestType> sv(5);
    std::vector<Kokkos::complex<TestType>> reference_data(32, {0.0, 0.0});
    reference_data[0] = 1.0;

    auto data = sv.getDataVector(0);
    if(sv.get_mpi_rank() == 0) {
        for (std::size_t j = 0; j < reference_data.size(); j++) {
            CHECK(real(data[j]) == Approx(real(reference_data[j])));
            CHECK(imag(data[j]) == Approx(imag(reference_data[j])));
        }
    }
}
