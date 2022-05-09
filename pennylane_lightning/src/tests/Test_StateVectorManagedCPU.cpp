#include "LinearAlgebra.hpp"
#include "StateVectorManagedCPU.hpp"
#include "StateVectorRawCPU.hpp"
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

TEMPLATE_TEST_CASE("StateVectorManagedCPU::StateVectorManagedCPU",
                   "[StateVectorManagedCPU]", float, double) {
    using PrecisionT = TestType;

    SECTION("StateVectorManagedCPU") {
        REQUIRE(!std::is_constructible_v<StateVectorManagedCPU<>>);
    }
    SECTION("StateVectorManagedCPU<TestType>") {
        REQUIRE(!std::is_constructible_v<StateVectorManagedCPU<TestType>>);
    }
    SECTION("StateVectorManagedCPU<TestType> {size_t}") {
        REQUIRE(
            std::is_constructible_v<StateVectorManagedCPU<TestType>, size_t>);
        const size_t num_qubits = 4;
        StateVectorManagedCPU<PrecisionT> sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
        REQUIRE(sv.getDataVector().size() == 16);
    }
    SECTION("StateVectorManagedCPU<TestType> {const "
            "StateVectorRawCPU<TestType>&}") {
        REQUIRE(std::is_constructible_v<StateVectorManagedCPU<TestType>,
                                        const StateVectorRawCPU<TestType> &>);
    }
    SECTION("StateVectorManagedCPU<TestType> {const "
            "StateVectorManagedCPU<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorManagedCPU<TestType>>);
    }
    SECTION(
        "StateVectorManagedCPU<TestType> {StateVectorManagedCPU<TestType>&&}") {
        REQUIRE(std::is_move_constructible_v<StateVectorManagedCPU<TestType>>);
    }
    SECTION("Aligned 256bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned256;
        StateVectorManagedCPU<PrecisionT> sv(4, Threading::SingleThread,
                                             memory_model);
        /* Even when we allocate 256 bit aligend memory, it is possible that the
         * alignment happens to be 512 bit */
        REQUIRE(((getMemoryModel(sv.getDataVector().data()) ==
                  CPUMemoryModel::Aligned256) ||
                 (getMemoryModel(sv.getDataVector().data()) ==
                  CPUMemoryModel::Aligned512)));
    }

    SECTION("Aligned 512bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned512;
        StateVectorManagedCPU<PrecisionT> sv(4, Threading::SingleThread,
                                             memory_model);
        REQUIRE((getMemoryModel(sv.getDataVector().data()) ==
                 CPUMemoryModel::Aligned512));
    }
}

TEMPLATE_TEST_CASE("StateVectorManagedCPU::applyMatrix with std::vector",
                   "[StateVectorManagedCPU]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test wrong matrix size") {
        std::vector<std::complex<TestType>> m(7, 0.0);
        const size_t num_qubits = 4;
        StateVectorManagedCPU<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyMatrix(m, {0, 1}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }

    SECTION("Test wrong number of wires") {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorManagedCPU<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyMatrix(m, {0}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }
}

TEMPLATE_TEST_CASE("StateVectorManagedCPU::applyMatrix with a pointer",
                   "[StateVectorManagedCPU]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test wrong matrix") {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorManagedCPU<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyMatrix(m.data(), {}),
                            Catch::Contains("must be larger than 0"));
    }

    SECTION("Test with different number of wires") {
        std::default_random_engine re{1337};
        const size_t num_qubits = 5;
        for (size_t num_wires = 1; num_wires < num_qubits; num_wires++) {
            StateVectorManagedCPU<PrecisionT> sv1(num_qubits);
            StateVectorManagedCPU<PrecisionT> sv2(num_qubits);

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

TEMPLATE_TEST_CASE("StateVectorManagedCPU::applyOperations",
                   "[StateVectorManagedCPU]", float, double) {
    using PrecisionT = TestType;

    std::mt19937 re{1337};

    SECTION("Test invalid arguments without params") {
        const size_t num_qubits = 4;
        StateVectorManagedCPU<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"PauliX", "PauliY"}, {{0}}, {false, false}),
            Catch::Contains("must all be equal")); // invalid wires
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"PauliX", "PauliY"}, {{0}, {1}}, {false}),
            Catch::Contains("must all be equal")); // invalid inverse
    }

    SECTION("applyOperations without params works as expected") {
        const size_t num_qubits = 3;
        StateVectorManagedCPU<PrecisionT> sv1(num_qubits);

        sv1.updateData(createRandomState<PrecisionT>(re, num_qubits));
        StateVectorManagedCPU<PrecisionT> sv2 = sv1;

        sv1.applyOperations({"PauliX", "PauliY"}, {{0}, {1}}, {false, false});

        sv2.applyOperation("PauliX", {0}, false);
        sv2.applyOperation("PauliY", {1}, false);

        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test invalid arguments with params") {
        const size_t num_qubits = 4;
        StateVectorManagedCPU<PrecisionT> sv(num_qubits);
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

    SECTION("applyOperations with params works as expected") {
        const size_t num_qubits = 3;
        StateVectorManagedCPU<PrecisionT> sv1(num_qubits);

        sv1.updateData(createRandomState<PrecisionT>(re, num_qubits));
        StateVectorManagedCPU<PrecisionT> sv2 = sv1;

        sv1.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false},
                            {{0.1}, {0.2}});

        sv2.applyOperation("RX", {0}, false, {0.1});
        sv2.applyOperation("RY", {1}, false, {0.2});

        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }
}
