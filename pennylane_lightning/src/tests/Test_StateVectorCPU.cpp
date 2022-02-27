#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "StateVectorCPU.hpp"
#include "StateVectorRaw.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;

TEMPLATE_TEST_CASE("StateVectorCPU::StateVectorCPU", "[StateVectorRaw]", float,
                   double) {
    using fp_t = TestType;

    SECTION("StateVectorCPU") {
        REQUIRE(!std::is_constructible_v<StateVectorCPU<>>);
    }
    SECTION("StateVectorCPU<TestType>") {
        REQUIRE(!std::is_constructible_v<StateVectorCPU<TestType>>);
    }
    SECTION("StateVectorCPU<TestType> {size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorCPU<TestType>, size_t>);
        const size_t num_qubits = 4;
        StateVectorCPU<fp_t> sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
    }
    SECTION("StateVectorCPU<TestType> {const StateVectorRaw<TestType>&}") {
        REQUIRE(std::is_constructible_v<StateVectorCPU<TestType>,
                                        const StateVectorRaw<TestType> &>);
    }
    SECTION("StateVectorCPU<TestType> {const StateVectorCPU<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorCPU<TestType>>);
    }
    SECTION("StateVectorCPU<TestType> {StateVectorCPU<TestType>&&}") {
        REQUIRE(std::is_move_constructible_v<StateVectorCPU<TestType>>);
    }
}
