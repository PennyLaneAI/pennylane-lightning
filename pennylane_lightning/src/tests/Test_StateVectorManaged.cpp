#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "StateVectorManaged.hpp"
#include "StateVectorRaw.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;

TEMPLATE_TEST_CASE("StateVectorManaged::StateVectorManaged", "[StateVectorRaw]",
                   float, double) {
    using fp_t = TestType;

    SECTION("StateVectorManaged") {
        REQUIRE(std::is_constructible_v<StateVectorManaged<>>);
    }
    SECTION("StateVectorManaged<TestType>") {
        REQUIRE(std::is_constructible_v<StateVectorManaged<TestType>>);
    }
    SECTION("StateVectorManaged<TestType> {size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorManaged<TestType>>);
        const size_t num_qubits = 4;
        StateVectorManaged<fp_t> sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
        REQUIRE(sv.getDataVector().size() == 16);
    }
    SECTION("StateVectorManaged<TestType> {const StateVectorRaw<TestType>&}") {
        REQUIRE(std::is_constructible_v<StateVectorManaged<TestType>,
                                        const StateVectorRaw<TestType> &>);
    }
    SECTION(
        "StateVectorManaged<TestType> {const StateVectorManaged<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorManaged<TestType>>);
    }
    SECTION("StateVectorManaged<TestType> {StateVectorManaged<TestType>&&}") {
        REQUIRE(std::is_move_constructible_v<StateVectorManaged<TestType>>);
    }
}
