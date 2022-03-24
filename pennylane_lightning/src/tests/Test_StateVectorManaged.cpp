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

TEMPLATE_TEST_CASE("StateVectorManaged::StateVectorManaged",
                   "[StateVectorManaged]", float, double) {
    using PrecisionT = TestType;

    SECTION("StateVectorManaged") {
        REQUIRE(!std::is_constructible_v<StateVectorManaged<>>);
    }
    SECTION("StateVectorManaged<TestType>") {
        REQUIRE(!std::is_constructible_v<StateVectorManaged<TestType>>);
    }
    SECTION("StateVectorManaged<TestType> {size_t}") {
        REQUIRE(std::is_constructible_v<StateVectorManaged<TestType>, size_t>);
        const size_t num_qubits = 4;
        StateVectorManaged<PrecisionT> sv(num_qubits);

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

TEMPLATE_TEST_CASE("StateVectorManaged::applyMatrix", "[StateVectorManaged]",
                   float, double) {
    using PrecisionT = TestType;
    SECTION("Test wrong matrix") {
        std::vector<std::complex<TestType>> m(7, 0.0);
        const size_t num_qubits = 4;
        StateVectorManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS(sv.applyMatrix(m, {0, 1}));
    }

    SECTION("Test wrong wires") {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS(sv.applyMatrix(m, {}));
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyOperations",
                   "[StateVectorManaged]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test wrong matrix") {
        std::vector<std::complex<PrecisionT>> m(7, 0.0);
        const size_t num_qubits = 4;
        StateVectorManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyMatrix(m, {0, 1}),
                            Catch::Contains("does not match with the given"));
    }

    SECTION("Test wrong wires") {
        std::vector<std::complex<PrecisionT>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyMatrix(m.data(), {}),
                            "Number of wires must be larger than 0");
    }
}
