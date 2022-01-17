#include <complex>
#include <numeric>
#include <vector>

#include "StateVectorRaw.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

using namespace Pennylane;

constexpr auto referenceKernel = KernelType::PI;

std::mt19937_64 re{1337};

TEMPLATE_TEST_CASE("StateVectorRaw::StateVectorRaw", "[StateVectorRaw]", float,
                   double) {
    using fp_t = TestType;

    SECTION("StateVectorRaw<TestType> {std::complex<TestType>*, size_t}") {
        const size_t num_qubits = 4;
        auto st_data = create_random_state<fp_t>(re, num_qubits);
        StateVectorRaw<fp_t> sv(st_data.data(), st_data.size());

        REQUIRE(sv.getNumQubits() == 4);
        REQUIRE(sv.getData() == st_data.data());
        REQUIRE(sv.getLength() == 16);
    }
    SECTION("StateVectorRaw<TestType> {std::complex<TestType>*, size_t}") {
        std::vector<std::complex<TestType>> st_data(14, 0.0);
        REQUIRE_THROWS(StateVectorRaw<fp_t>(st_data.data(), st_data.size()));
    }
}

TEMPLATE_TEST_CASE("StateVectorRaw::setData", "[StateVectorRaw]", float,
                   double) {
    using fp_t = TestType;

    auto st_data = create_random_state<fp_t>(re, 4);
    StateVectorRaw<fp_t> sv(st_data.data(), st_data.size());

    auto st_data2 = create_random_state<fp_t>(re, 8);
    sv.setData(st_data2.data(), st_data2.size());

    REQUIRE(sv.getNumQubits() == 8);
    REQUIRE(sv.getData() == st_data2.data());
    REQUIRE(sv.getLength() == (1U << 8));
}

#define PENNYLANE_TEST_STATIC_DISPATCH(GATE_NAME)                              \
    template <typename fp_t, int num_params> struct TestGateOps##GATE_NAME {   \
        static void test() {                                                   \
            static_assert(                                                     \
                (num_params != 2) || (num_params > 3),                         \
                "Unsupported number of parameters for gate " #GATE_NAME ".");  \
        }                                                                      \
    };                                                                         \
    template <typename fp_t> struct TestGateOps##GATE_NAME<fp_t, 0> {          \
        constexpr static GateOperations op = GateOperations::GATE_NAME;        \
        template <KernelType kernel>                                           \
        static std::enable_if_t<                                               \
            array_has_elt(SelectGateOps<fp_t, kernel>::implemented_gates, op), \
            void>                                                              \
        test() {                                                               \
            const size_t num_qubits = 4;                                       \
            auto ini_st = create_random_state<fp_t>(re, num_qubits);           \
            auto num_wires = lookup(Constant::gate_wires, op);                 \
            std::vector<size_t> wires;                                         \
            wires.resize(num_wires);                                           \
            std::iota(wires.begin(), wires.end(), 0);                          \
            auto expected = ini_st;                                            \
            SelectGateOps<fp_t, referenceKernel>::apply##GATE_NAME(            \
                expected.data(), num_qubits, wires, false);                    \
            auto sv_data = ini_st;                                             \
            StateVectorRaw sv(sv_data.data(), sv_data.size());                 \
            sv.apply##GATE_NAME(wires, false);                                 \
            REQUIRE(isApproxEqual(sv_data, expected));                         \
        }                                                                      \
        template <KernelType kernel>                                           \
        static std::enable_if_t<                                               \
            !array_has_elt(SelectGateOps<fp_t, kernel>::implemented_gates,     \
                           op),                                                \
            void>                                                              \
        test() { /* do nothing */                                              \
        }                                                                      \
    };                                                                         \
    template <typename fp_t> struct TestGateOps##GATE_NAME<fp_t, 1> {          \
        constexpr static GateOperations op = GateOperations::GATE_NAME;        \
        constexpr static std::array<fp_t, 1> params = {0.312};                 \
        template <KernelType kernel>                                           \
        static std::enable_if_t<                                               \
            array_has_elt(SelectGateOps<fp_t, kernel>::implemented_gates, op), \
            void>                                                              \
        test() {                                                               \
            const size_t num_qubits = 4;                                       \
            auto ini_st = create_random_state<fp_t>(re, num_qubits);           \
            auto num_wires = lookup(Constant::gate_wires, op);                 \
            std::vector<size_t> wires;                                         \
            wires.resize(num_wires);                                           \
            std::iota(wires.begin(), wires.end(), 0);                          \
            auto expected = ini_st;                                            \
            SelectGateOps<fp_t, referenceKernel>::template apply##GATE_NAME<   \
                fp_t>(expected.data(), num_qubits, wires, false, params[0]);   \
            auto sv_data = ini_st;                                             \
            StateVectorRaw sv(sv_data.data(), sv_data.size());                 \
            sv.apply##GATE_NAME(wires, false, params[0]);                      \
            REQUIRE(isApproxEqual(sv_data, expected));                         \
        }                                                                      \
        template <KernelType kernel>                                           \
        static std::enable_if_t<                                               \
            !array_has_elt(SelectGateOps<fp_t, kernel>::implemented_gates,     \
                           op),                                                \
            void>                                                              \
        test() { /* do nothing */                                              \
        }                                                                      \
    };                                                                         \
    template <typename fp_t> struct TestGateOps##GATE_NAME<fp_t, 3> {          \
        constexpr static GateOperations op = GateOperations::GATE_NAME;        \
        constexpr static std::array<fp_t, 3> params = {0.128, -0.563, 1.414};  \
        template <KernelType kernel>                                           \
        static std::enable_if_t<                                               \
            array_has_elt(SelectGateOps<fp_t, kernel>::implemented_gates, op), \
            void>                                                              \
        test() {                                                               \
            const size_t num_qubits = 4;                                       \
            auto ini_st = create_random_state<fp_t>(re, num_qubits);           \
            auto num_wires = lookup(Constant::gate_wires, op);                 \
            std::vector<size_t> wires;                                         \
            wires.resize(num_wires);                                           \
            std::iota(wires.begin(), wires.end(), 0);                          \
            auto expected = ini_st;                                            \
            SelectGateOps<fp_t, referenceKernel>::template apply##GATE_NAME<   \
                fp_t>(expected.data(), num_qubits, wires, false, params[0],    \
                      params[1], params[2]);                                   \
            auto sv_data = ini_st;                                             \
            StateVectorRaw sv(sv_data.data(), sv_data.size());                 \
            sv.apply##GATE_NAME(wires, false, params[0], params[1],            \
                                params[2]);                                    \
            REQUIRE(isApproxEqual(sv_data, expected));                         \
        }                                                                      \
        template <KernelType kernel>                                           \
        static std::enable_if_t<                                               \
            !array_has_elt(SelectGateOps<fp_t, kernel>::implemented_gates,     \
                           op),                                                \
            void>                                                              \
        test() { /* do nothing */                                              \
        }                                                                      \
    };                                                                         \
    template <typename fp_t, size_t idx>                                       \
    void testStateVectorApply##GATE_NAME##Iter() {                             \
        if constexpr (idx < Constant::available_kernels.size()) {              \
            constexpr auto kernel =                                            \
                std::get<0>(Constant::available_kernels[idx]);                 \
            TestGateOps##GATE_NAME<                                            \
                fp_t,                                                          \
                static_lookup<GateOperations::GATE_NAME>(                      \
                    Constant::gate_num_params)>::template test<kernel>();      \
            testStateVectorApply##GATE_NAME##Iter<fp_t, idx + 1>();            \
        }                                                                      \
    }                                                                          \
    template <typename fp_t> void testStateVectorApply##GATE_NAME() {          \
        testStateVectorApply##GATE_NAME##Iter<fp_t, 0>();                      \
    }                                                                          \
    TEMPLATE_TEST_CASE("StateVectorRaw::" #GATE_NAME, "[StateVectorRaw]",      \
                       float, double) {                                        \
        using fp_t = TestType;                                                 \
        testStateVectorApply##GATE_NAME<fp_t>();                               \
    }

PENNYLANE_TEST_STATIC_DISPATCH(PauliX)
PENNYLANE_TEST_STATIC_DISPATCH(PauliY)
PENNYLANE_TEST_STATIC_DISPATCH(PauliZ)
PENNYLANE_TEST_STATIC_DISPATCH(Hadamard)
PENNYLANE_TEST_STATIC_DISPATCH(S)
PENNYLANE_TEST_STATIC_DISPATCH(T)
PENNYLANE_TEST_STATIC_DISPATCH(PhaseShift)
PENNYLANE_TEST_STATIC_DISPATCH(RX)
PENNYLANE_TEST_STATIC_DISPATCH(RY)
PENNYLANE_TEST_STATIC_DISPATCH(RZ)
PENNYLANE_TEST_STATIC_DISPATCH(Rot)
PENNYLANE_TEST_STATIC_DISPATCH(CNOT)
PENNYLANE_TEST_STATIC_DISPATCH(CZ)
PENNYLANE_TEST_STATIC_DISPATCH(SWAP)
PENNYLANE_TEST_STATIC_DISPATCH(ControlledPhaseShift)
PENNYLANE_TEST_STATIC_DISPATCH(CRX)
PENNYLANE_TEST_STATIC_DISPATCH(CRY)
PENNYLANE_TEST_STATIC_DISPATCH(CRZ)
PENNYLANE_TEST_STATIC_DISPATCH(CRot)
PENNYLANE_TEST_STATIC_DISPATCH(Toffoli)
PENNYLANE_TEST_STATIC_DISPATCH(CSWAP)
PENNYLANE_TEST_STATIC_DISPATCH(GeneratorPhaseShift)
PENNYLANE_TEST_STATIC_DISPATCH(GeneratorCRX)
PENNYLANE_TEST_STATIC_DISPATCH(GeneratorCRY)
PENNYLANE_TEST_STATIC_DISPATCH(GeneratorCRZ)
PENNYLANE_TEST_STATIC_DISPATCH(GeneratorControlledPhaseShift)
