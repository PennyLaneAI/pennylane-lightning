#include "DefaultKernels.hpp"
#include "LinearAlgebra.hpp"
#include "StateVectorManaged.hpp"
#include "StateVectorRaw.hpp"
#include "Util.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

#include "TestHelpers.hpp"
#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

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

TEMPLATE_TEST_CASE("StateVectorManaged::applyMatrix with std::vector",
                   "[StateVectorManaged]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test wrong matrix size") {
        std::vector<std::complex<TestType>> m(7, 0.0);
        const size_t num_qubits = 4;
        StateVectorManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyMatrix(m, {0, 1}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }

    SECTION("Test wrong number of wires") {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyMatrix(m, {0}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyMatrix with a pointer",
                   "[StateVectorManaged]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test wrong matrix") {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyMatrix(m.data(), {}),
                            Catch::Contains("must be larger than 0"));
    }

    SECTION("Test with different number of wires") {
        std::default_random_engine re{1337};
        const size_t num_qubits = 5;
        for (size_t num_wires = 1; num_wires < num_qubits; num_wires++) {
            StateVectorManaged<PrecisionT> sv1(num_qubits);
            StateVectorManaged<PrecisionT> sv2(num_qubits);

            std::vector<size_t> wires(num_wires);
            std::iota(wires.begin(), wires.end(), 0);

            const auto m = Util::randomUnitary<PrecisionT>(re, num_wires);
            sv1.applyMatrix(m, wires);
            Gates::GateImplementationsPI::applyMultiQubitOp<PrecisionT>(
                sv2.getData(), num_qubits, m.data(), wires, false);
            REQUIRE(sv1.getDataVector() ==
                    approx(sv2.getDataVector()).margin(PrecisionT{1e-5}));
        }
    }
}

TEMPLATE_TEST_CASE("StateVectorManaged::applyOperations",
                   "[StateVectorManaged]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test invalid arguments without params") {
        const size_t num_qubits = 4;
        StateVectorManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"PauliX", "PauliY"}, {{0}}, {false, false}),
            Catch::Contains("must all be equal")); // invalid wires
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"PauliX", "PauliY"}, {{0}, {1}}, {false}),
            Catch::Contains("must all be equal")); // invalid inverse
    }

    SECTION("Test invalid arguments with params") {
        const size_t num_qubits = 4;
        StateVectorManaged<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"RX", "RY"}, {{0}}, {false, false},
                               {{0.0}, {0.0}}),
            Catch::Contains("must all be equal")); // invalid wires
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"RX", "RY"}, {{0}, {1}}, {false},
                               {{0.0}, {0.0}}),
            Catch::Contains("must all be equal")); // invalid inverse

        REQUIRE_THROWS_WITH(
            sv.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false},
                               {{0.0}}),
            Catch::Contains("must all be equal")); // invalid params
    }
}
