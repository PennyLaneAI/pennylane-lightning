#include "LinearAlgebra.hpp"
#include "StateVectorDynamicCPU.hpp"
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

TEMPLATE_TEST_CASE("StateVectorDynamicCPU::StateVectorDynamicCPU",
                   "[StateVectorDynamicCPU]", float, double) {
    using PrecisionT = TestType;

    SECTION("StateVectorDynamicCPU") {
        REQUIRE(!std::is_constructible_v<StateVectorDynamicCPU<>>);
    }
    SECTION("StateVectorDynamicCPU<TestType>") {
        REQUIRE(!std::is_constructible_v<StateVectorDynamicCPU<TestType>>);
    }
    SECTION("StateVectorDynamicCPU<TestType> {size_t}") {
        REQUIRE(
            std::is_constructible_v<StateVectorDynamicCPU<TestType>, size_t>);
        const size_t num_qubits = 4;
        StateVectorDynamicCPU<PrecisionT> sv(num_qubits);

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getLength() == 16);
        REQUIRE(sv.getDataVector().size() == 16);
    }
    SECTION("StateVectorDynamicCPU<TestType> {const "
            "StateVectorRawCPU<TestType>&}") {
        REQUIRE(std::is_constructible_v<StateVectorDynamicCPU<TestType>,
                                        const StateVectorRawCPU<TestType> &>);
    }
    SECTION("StateVectorDynamicCPU<TestType> {const "
            "StateVectorDynamicCPU<TestType>&}") {
        REQUIRE(std::is_copy_constructible_v<StateVectorDynamicCPU<TestType>>);
    }
    SECTION(
        "StateVectorDynamicCPU<TestType> {StateVectorDynamicCPU<TestType>&&}") {
        REQUIRE(std::is_move_constructible_v<StateVectorDynamicCPU<TestType>>);
    }
    SECTION("Aligned 256bit statevector") {
        const auto memory_model = CPUMemoryModel::Aligned256;
        StateVectorDynamicCPU<PrecisionT> sv(4, Threading::SingleThread,
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
        StateVectorDynamicCPU<PrecisionT> sv(4, Threading::SingleThread,
                                             memory_model);
        REQUIRE((getMemoryModel(sv.getDataVector().data()) ==
                 CPUMemoryModel::Aligned512));
    }
}

TEMPLATE_TEST_CASE("StateVectorDynamicCPU::applyMatrix with std::vector",
                   "[StateVectorDynamicCPU]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test wrong matrix size") {
        std::vector<std::complex<TestType>> m(7, 0.0);
        const size_t num_qubits = 4;
        StateVectorDynamicCPU<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyMatrix(m, {0, 1}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }

    SECTION("Test wrong number of wires") {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorDynamicCPU<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyMatrix(m, {0}),
            Catch::Contains(
                "The size of matrix does not match with the given"));
    }
}

TEMPLATE_TEST_CASE("StateVectorDynamicCPU::applyMatrix with a pointer",
                   "[StateVectorDynamicCPU]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test wrong matrix") {
        std::vector<std::complex<TestType>> m(8, 0.0);
        const size_t num_qubits = 4;
        StateVectorDynamicCPU<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(sv.applyMatrix(m.data(), {}),
                            Catch::Contains("must be larger than 0"));
    }

    SECTION("Test with different number of wires") {
        std::default_random_engine re{1337};
        const size_t num_qubits = 5;
        for (size_t num_wires = 1; num_wires < num_qubits; num_wires++) {
            StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);
            StateVectorDynamicCPU<PrecisionT> sv2(num_qubits);

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

TEMPLATE_TEST_CASE("StateVectorDynamicCPU::applyOperations",
                   "[StateVectorDynamicCPU]", float, double) {
    using PrecisionT = TestType;

    std::mt19937 re{1337};

    SECTION("Test invalid arguments without params") {
        const size_t num_qubits = 4;
        StateVectorDynamicCPU<PrecisionT> sv(num_qubits);
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"PauliX", "PauliY"}, {{0}}, {false, false}),
            Catch::Contains("must all be equal")); // invalid wires
        REQUIRE_THROWS_WITH(
            sv.applyOperations({"PauliX", "PauliY"}, {{0}, {1}}, {false}),
            Catch::Contains("must all be equal")); // invalid inverse
    }

    SECTION("applyOperations without params works as expected") {
        const size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.updateData(createRandomState<PrecisionT>(re, num_qubits));
        StateVectorDynamicCPU<PrecisionT> sv2 = sv1;

        sv1.applyOperations({"PauliX", "PauliY"}, {{0}, {1}}, {false, false});

        sv2.applyOperation("PauliX", {0}, false);
        sv2.applyOperation("PauliY", {1}, false);

        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test invalid arguments with params") {
        const size_t num_qubits = 4;
        StateVectorDynamicCPU<PrecisionT> sv(num_qubits);
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
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.updateData(createRandomState<PrecisionT>(re, num_qubits));
        StateVectorDynamicCPU<PrecisionT> sv2 = sv1;

        sv1.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false},
                            {{0.1}, {0.2}});

        sv2.applyOperation("RX", {0}, false, {0.1});
        sv2.applyOperation("RY", {1}, false, {0.2});

        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("applyOperation with disabled wire") {
        const size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        using WIRE_STATUS = StateVectorDynamicCPU<PrecisionT>::WIRE_STATUS;

        REQUIRE(sv1.getTotalNumWires() == num_qubits);

        sv1.releaseWire(0);
        REQUIRE(sv1.getWireStatus(0) == WIRE_STATUS::RELEASED);

        REQUIRE(
            (sv1.getNumReleasedWires() == 1 && sv1.getNumActiveWires() == 2));

        sv1.disableWire(0);
        REQUIRE(sv1.getWireStatus(0) == WIRE_STATUS::DISABLED);

        REQUIRE(
            (sv1.getNumReleasedWires() == 0 && sv1.getNumDisabledWires() == 1));

        REQUIRE_THROWS_WITH(
            sv1.applyOperation("PauliX", sv1.mapWiresToOperation({0}), false),
            Catch::Contains("Invalid list of wires"));
    }

    SECTION("applyOperations with released wires") {
        const size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        using WIRE_STATUS = StateVectorDynamicCPU<PrecisionT>::WIRE_STATUS;

        size_t new_idx = sv1.activateWire();
        REQUIRE(sv1.getWireStatus(new_idx) == WIRE_STATUS::ACTIVE);

        sv1.releaseWire(new_idx);
        REQUIRE((sv1.getWireStatus(new_idx) == WIRE_STATUS::RELEASED &&
                 sv1.getWireStatus(0) == WIRE_STATUS::ACTIVE));

        REQUIRE(sv1.getNumQubits() == num_qubits);
        REQUIRE(sv1.getTotalNumWires() == num_qubits + 1);
        REQUIRE((sv1.getNumReleasedWires() == 1 &&
                 sv1.getNumActiveWires() == num_qubits));

        REQUIRE_THROWS_WITH(
            sv1.applyOperations({"PauliX", "PauliY"},
                                sv1.mapWiresToOperations({{new_idx}, {1}}),
                                {false, false}),
            Catch::Contains("Invalid list of wires"));
    }
}

TEMPLATE_TEST_CASE("StateVectorDynamicCPU::activeWires",
                   "[StateVectorDynamicCPU]", float, double) {
    using PrecisionT = TestType;

    SECTION("Test counting wires for a simple state-vector") {
        const size_t num_qubits = 10;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.releaseWire(1);
        sv1.releaseWire(4);
        sv1.releaseWire(6);
        sv1.releaseWire(9);
        sv1.disableWire(6);
        sv1.allocateWire();
        sv1.allocateWire();
        sv1.allocateWire();
        sv1.allocateWire();

        REQUIRE(
            (sv1.getTotalNumWires() == 14 && sv1.getNumActiveWires() == 10));
        REQUIRE(
            (sv1.getNumReleasedWires() == 3 && sv1.getNumDisabledWires() == 1));
    }

    SECTION("Test counting Active wires for a simple state-vector") {
        const size_t num_qubits = 10;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.releaseWire(1);
        sv1.releaseWire(4);
        sv1.releaseWire(6);
        sv1.releaseWire(9);
        sv1.disableWire(6);
        sv1.allocateWire();
        sv1.allocateWire();
        sv1.disableWire(11);
        sv1.allocateWire();
        sv1.allocateWire();

        REQUIRE(
            (sv1.getNumActiveWires(13) == 8 && sv1.getNumActiveWires(0) == 0));
        REQUIRE(
            (sv1.getNumActiveWires(1) == 1 && sv1.getNumActiveWires(2) == 1));
        REQUIRE(
            (sv1.getNumActiveWires(3) == 2 && sv1.getNumActiveWires(4) == 3));
        REQUIRE(
            (sv1.getNumActiveWires(5) == 3 && sv1.getNumActiveWires(6) == 4));
        REQUIRE(
            (sv1.getNumActiveWires(7) == 4 && sv1.getNumActiveWires(8) == 5));
        REQUIRE(
            (sv1.getNumActiveWires(9) == 6 && sv1.getNumActiveWires(10) == 6));
    }
}

TEMPLATE_TEST_CASE("StateVectorDynamicCPU::mapWiresToOperation",
                   "[StateVectorDynamicCPU]", float, double) {
    using PrecisionT = TestType;
    SECTION("Test counting wires for a simple state-vector") {
        const size_t num_qubits = 10;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        std::vector<size_t> expected_wires(num_qubits);
        std::iota(expected_wires.begin(), expected_wires.end(), 0);

        REQUIRE(sv1.mapWiresToOperation(expected_wires) ==
                approx(expected_wires));
    }

    SECTION("Test counting wires for a random state-vector") {
        const size_t num_qubits = 10;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.releaseWire(0);
        sv1.disableWire(8);
        sv1.releaseWire(1);
        sv1.releaseWire(6);
        sv1.releaseWire(2);
        sv1.allocateWire();
        sv1.allocateWire();
        sv1.disableWire(11);
        sv1.releaseWire(10);

        REQUIRE_THROWS_WITH(sv1.mapWiresToOperation({0, 3, 5}),
                            Catch::Contains("Invalid list of wires"));

        REQUIRE_THROWS_WITH(sv1.mapWiresToOperation({4, 3}),
                            Catch::Contains("Invalid list of wires"));

        REQUIRE(sv1.mapWiresToOperation({3, 4, 5}) ==
                approx(std::vector<size_t>{0, 1, 2}));

        REQUIRE(sv1.mapWiresToOperation({3, 5, 7}) ==
                approx(std::vector<size_t>{0, 2, 3}));

        REQUIRE(sv1.mapWiresToOperation({3, 5, 9}) ==
                approx(std::vector<size_t>{0, 2, 4}));
    }
}

TEMPLATE_TEST_CASE("StateVectorDynamicCPU::getSubsystemPurity",
                   "[StateVectorDynamicCPU]", float, double) {
    using PrecisionT = TestType;

    SECTION("Test getSubsystemPurity for a state-vector with RX-RY") {
        const size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false},
                            {{0.1}, {0.2}});

        REQUIRE(sv1.getSubsystemPurity(0) ==
                approx(std::complex<PrecisionT>{1, 0}));
        REQUIRE(sv1.getSubsystemPurity(1) ==
                approx(std::complex<PrecisionT>{1, 0}));
        REQUIRE(sv1.getSubsystemPurity(2) ==
                approx(std::complex<PrecisionT>{1, 0}));
    }

    SECTION("Test checkSubsystemPurity for a state-vector with RX-RY") {
        const size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.applyOperations({"RX", "RY"}, {{0}, {1}}, {false, false},
                            {{0.1}, {0.2}});

        REQUIRE((sv1.checkSubsystemPurity(0) && sv1.checkSubsystemPurity(1)));
        REQUIRE(sv1.checkSubsystemPurity(2));
    }

    SECTION("Test getSubsystemPurity for a state-vector with CNOT-RY") {
        const size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.applyOperations({"CNOT", "RY"}, {{0, 1}, {1}}, {false, false},
                            {{}, {0.2}});

        REQUIRE(sv1.getSubsystemPurity(0) ==
                approx(std::complex<PrecisionT>{1, 0}));
        REQUIRE(sv1.getSubsystemPurity(1) ==
                approx(std::complex<PrecisionT>{1, 0}));
        REQUIRE(sv1.checkSubsystemPurity(2));
    }

    SECTION("Test getSubsystemPurity for a custom state-vector") {
        const size_t num_qubits = 2;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        std::vector<std::complex<PrecisionT>> data{
            {1 / 2, 0}, {1 / 2, 0}, {-1 / 2, 0}, {-1 / 2, 0}};

        sv1.updateData(data);

        REQUIRE(sv1.getSubsystemPurity(0) !=
                approx(std::complex<PrecisionT>{1, 0}));
        REQUIRE(sv1.getSubsystemPurity(1) !=
                approx(std::complex<PrecisionT>{1, 0}));
    }
}

TEMPLATE_TEST_CASE("StateVectorDynamicCPU::allocateWire",
                   "[StateVectorDynamicCPU]", float, double) {
    using PrecisionT = TestType;
    // using WIRE_STATUS = StateVectorDynamicCPU<PrecisionT>::WIRE_STATUS;

    SECTION("Test allocation/deallocation of wires for a state-vector with "
            "num_qubits=1") {
        const size_t num_qubits = 1;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);
        sv1.applyOperation("Hadamard", {0}, false, {});

        StateVectorDynamicCPU<PrecisionT> sv2 = sv1;

        size_t new_idx = sv1.allocateWire();
        sv1.applyOperation("RX", sv1.mapWiresToOperation({new_idx}), false,
                           {0.3});

        sv1.releaseWire(0);
        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test allocation/deallocation of wires for a state-vector with "
            "num_qubits=2") {
        const size_t num_qubits = 2;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);
        sv1.applyOperations({"RX", "CNOT"}, {{0}, {0, 1}}, {false, false},
                            {{0.4}, {}});

        StateVectorDynamicCPU<PrecisionT> sv2 = sv1;

        size_t new_idx = sv1.allocateWire();
        sv1.applyOperation("RX", sv1.mapWiresToOperation({new_idx}), false,
                           {0.3});

        sv1.releaseWire(0);

        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test allocation/deallocation of wires for a state-vector with "
            "num_qubits=3") {
        const size_t num_qubits = 3;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.applyOperations({"RX", "SWAP"}, {{0}, {0, 2}}, {false, false},
                            {{0.4}, {}});

        StateVectorDynamicCPU<PrecisionT> sv2{num_qubits - 1};
        sv2.applyOperations({"RX", "SWAP"}, {{0}, {0, 1}}, {false, false},
                            {{0.4}, {}});

        sv1.releaseWire(1);
        REQUIRE(sv1.getDataVector() == approx(sv2.getDataVector()));
    }

    SECTION("Test allocation/deallocation of wires for a state-vector with "
            "num_qubits=4") {
        const size_t num_qubits = 4;
        StateVectorDynamicCPU<PrecisionT> sv1(num_qubits);

        sv1.applyOperations({"RX", "SWAP", "RY", "Hadamard", "RZ", "CNOT"},
                            {{0}, {1, 2}, {1}, {3}, {2}, {1, 3}},
                            {false, false, false, false, false, false},
                            {{0.4}, {}, {0.6}, {}, {0.8}, {}});

        std::vector<std::complex<TestType>> result{
            {-0.27536, -0.651288},
            {-0.27536, -0.651288},
        };

        auto sv_data = sv1.getDataVector();
        std::cout << "state: ";
        for (const auto &e : sv_data) {
            std::cout << e << ", ";
        }
        std::cout << std::endl;

        sv1.releaseWire(1);
        sv1.releaseWire(2);
        sv1.releaseWire(3);

        sv_data = sv1.getDataVector();
        std::cout << "state: ";
        for (const auto &e : sv_data) {
            std::cout << e << ", ";
        }
        std::cout << std::endl;

        REQUIRE(sv1.getDataVector() == approx(result));
    }
}
