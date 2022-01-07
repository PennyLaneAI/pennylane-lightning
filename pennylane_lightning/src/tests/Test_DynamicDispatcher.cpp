#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "DynamicDispatcher.hpp"
#include "Gates.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;

/**
 * @file This file contains tests for DynamicDispatcher class
 */

#define PENNYLANE_TEST_DYNAMIC_DISPATCH(GATE_NAME)                             \
    template <class fp_t, KernelType kernel, int num_params>                   \
    struct testDispatch##GATE_NAME##ForKernel {                                \
        template <class RandomEngine>                                          \
        static void test(RandomEngine &re, size_t num_qubits) {                \
            static_assert((num_params <= 1) || (num_params == 3),              \
                          "Cannot find the given num_params.");                \
        }                                                                      \
    };                                                                         \
    template <class fp_t, KernelType kernel>                                   \
    struct testDispatch##GATE_NAME##ForKernel<fp_t, kernel, 0> {               \
        template <class RandomEngine>                                          \
        static void test(RandomEngine &re, size_t num_qubits) {                \
            using CFP_t = std::complex<fp_t>;                                  \
            std::vector<CFP_t> ini_st =                                        \
                create_random_state<fp_t>(re, num_qubits);                     \
            std::vector<CFP_t> expected = ini_st;                              \
            std::vector<size_t> wires =                                        \
                createWires(GateOperations::GATE_NAME);                        \
            std::vector<fp_t> params =                                         \
                createParams<fp_t>(GateOperations::GATE_NAME);                 \
            SelectGateOps<fp_t, KernelType::PI>::apply##GATE_NAME(             \
                expected.data(), num_qubits, wires, false);                    \
            auto test_st = ini_st;                                             \
            if constexpr (array_has_elt(                                       \
                              SelectGateOps<fp_t, kernel>::implemented_gates,  \
                              GateOperations::GATE_NAME)) {                    \
                DynamicDispatcher<fp_t>::getInstance().applyOperation(         \
                    kernel, test_st.data(), num_qubits, #GATE_NAME, wires,     \
                    false, params);                                            \
                REQUIRE(isApproxEqual(test_st, expected));                     \
            } else {                                                           \
                REQUIRE_THROWS(                                                \
                    DynamicDispatcher<fp_t>::getInstance().applyOperation(     \
                        kernel, test_st.data(), num_qubits, #GATE_NAME, wires, \
                        false, params));                                       \
            }                                                                  \
        }                                                                      \
    };                                                                         \
    template <class fp_t, KernelType kernel>                                   \
    struct testDispatch##GATE_NAME##ForKernel<fp_t, kernel, 1> {               \
        template <class RandomEngine>                                          \
        static void test(RandomEngine &re, size_t num_qubits) {                \
            using CFP_t = std::complex<fp_t>;                                  \
            std::vector<CFP_t> ini_st =                                        \
                create_random_state<fp_t>(re, num_qubits);                     \
            std::vector<CFP_t> expected = ini_st;                              \
            std::vector<size_t> wires =                                        \
                createWires(GateOperations::GATE_NAME);                        \
            std::vector<fp_t> params =                                         \
                createParams<fp_t>(GateOperations::GATE_NAME);                 \
            SelectGateOps<fp_t, KernelType::PI>::apply##GATE_NAME(             \
                expected.data(), num_qubits, wires, false, params[0]);         \
            auto test_st = ini_st;                                             \
            if constexpr (array_has_elt(                                       \
                              SelectGateOps<fp_t, kernel>::implemented_gates,  \
                              GateOperations::GATE_NAME)) {                    \
                DynamicDispatcher<fp_t>::getInstance().applyOperation(         \
                    kernel, test_st.data(), num_qubits, #GATE_NAME, wires,     \
                    false, params);                                            \
                REQUIRE(isApproxEqual(test_st, expected));                     \
            } else {                                                           \
                REQUIRE_THROWS(                                                \
                    DynamicDispatcher<fp_t>::getInstance().applyOperation(     \
                        kernel, test_st.data(), num_qubits, #GATE_NAME, wires, \
                        false, params));                                       \
            }                                                                  \
        }                                                                      \
    };                                                                         \
    template <class fp_t, KernelType kernel>                                   \
    struct testDispatch##GATE_NAME##ForKernel<fp_t, kernel, 3> {               \
        template <class RandomEngine>                                          \
        static void test(RandomEngine &re, size_t num_qubits) {                \
            using CFP_t = std::complex<fp_t>;                                  \
            std::vector<CFP_t> ini_st =                                        \
                create_random_state<fp_t>(re, num_qubits);                     \
            std::vector<CFP_t> expected = ini_st;                              \
            std::vector<size_t> wires =                                        \
                createWires(GateOperations::GATE_NAME);                        \
            std::vector<fp_t> params =                                         \
                createParams<fp_t>(GateOperations::GATE_NAME);                 \
            SelectGateOps<fp_t, KernelType::PI>::apply##GATE_NAME(             \
                expected.data(), num_qubits, wires, false, params[0],          \
                params[1], params[2]);                                         \
            auto test_st = ini_st;                                             \
            if constexpr (array_has_elt(                                       \
                              SelectGateOps<fp_t, kernel>::implemented_gates,  \
                              GateOperations::GATE_NAME)) {                    \
                DynamicDispatcher<fp_t>::getInstance().applyOperation(         \
                    kernel, test_st.data(), num_qubits, #GATE_NAME, wires,     \
                    false, params);                                            \
                REQUIRE(isApproxEqual(test_st, expected));                     \
            } else {                                                           \
                REQUIRE_THROWS(                                                \
                    DynamicDispatcher<fp_t>::getInstance().applyOperation(     \
                        kernel, test_st.data(), num_qubits, #GATE_NAME, wires, \
                        false, params));                                       \
            }                                                                  \
        }                                                                      \
    };                                                                         \
    template <class fp_t, int idx, class RandomEngine>                         \
    void testDispatch##GATE_NAME##Iter(RandomEngine &&re, size_t num_qubits) { \
        if constexpr (idx < Constant::available_kernels.size()) {              \
            testDispatch##GATE_NAME##ForKernel<                                \
                fp_t, std::get<0>(Constant::available_kernels[idx]),           \
                static_lookup<GateOperations::GATE_NAME>(                      \
                    Constant::gate_num_params)>::test(re, num_qubits);         \
            testDispatch##GATE_NAME##Iter<fp_t, idx + 1>(re, num_qubits);      \
        }                                                                      \
    }                                                                          \
    template <class fp_t, class RandomEngine>                                  \
    void testDispatch##GATE_NAME(RandomEngine &&re, size_t num_qubits) {       \
        testDispatch##GATE_NAME##Iter<fp_t, 0>(re, num_qubits);                \
    }

std::vector<size_t> createWires(GateOperations op) {
    switch (lookup(Constant::gate_wires, op)) {
    case 1:
        return {0};
    case 2:
        return {0, 1};
    case 3:
        return {0, 1, 2};
    default:
        PL_ABORT("The number of wires for a given gate is unset.");
    }
}

template <class fp_t> std::vector<fp_t> createParams(GateOperations op) {
    switch (lookup(Constant::gate_num_params, op)) {
    case 0:
        return {};
    case 1:
        return {0.312};
    case 3:
        return {0.128, -0.563, 1.414};
    default:
        PL_ABORT("The number of wires for a given gate is unset.");
    }
}

PENNYLANE_TEST_DYNAMIC_DISPATCH(PauliX)
PENNYLANE_TEST_DYNAMIC_DISPATCH(PauliY)
PENNYLANE_TEST_DYNAMIC_DISPATCH(PauliZ)
PENNYLANE_TEST_DYNAMIC_DISPATCH(Hadamard)
PENNYLANE_TEST_DYNAMIC_DISPATCH(S)
PENNYLANE_TEST_DYNAMIC_DISPATCH(T)
PENNYLANE_TEST_DYNAMIC_DISPATCH(PhaseShift)
PENNYLANE_TEST_DYNAMIC_DISPATCH(RX)
PENNYLANE_TEST_DYNAMIC_DISPATCH(RY)
PENNYLANE_TEST_DYNAMIC_DISPATCH(RZ)
PENNYLANE_TEST_DYNAMIC_DISPATCH(Rot)
PENNYLANE_TEST_DYNAMIC_DISPATCH(CNOT)
PENNYLANE_TEST_DYNAMIC_DISPATCH(CZ)
PENNYLANE_TEST_DYNAMIC_DISPATCH(SWAP)
PENNYLANE_TEST_DYNAMIC_DISPATCH(ControlledPhaseShift)
PENNYLANE_TEST_DYNAMIC_DISPATCH(CRX)
PENNYLANE_TEST_DYNAMIC_DISPATCH(CRY)
PENNYLANE_TEST_DYNAMIC_DISPATCH(CRZ)
PENNYLANE_TEST_DYNAMIC_DISPATCH(CRot)
PENNYLANE_TEST_DYNAMIC_DISPATCH(Toffoli)
PENNYLANE_TEST_DYNAMIC_DISPATCH(CSWAP)

TEMPLATE_TEST_CASE("DynamicDispatcher::applyOperation", "[DynamicDispatcher]",
                   float, double) {
    std::default_random_engine re{1337};
    testDispatchPauliX<TestType>(re, 4);
    testDispatchPauliY<TestType>(re, 4);
    testDispatchPauliZ<TestType>(re, 4);
    testDispatchHadamard<TestType>(re, 4);
    testDispatchS<TestType>(re, 4);
    testDispatchT<TestType>(re, 4);
    testDispatchPhaseShift<TestType>(re, 4);
    testDispatchRX<TestType>(re, 4);
    testDispatchRY<TestType>(re, 4);
    testDispatchRZ<TestType>(re, 4);
    testDispatchRot<TestType>(re, 4);
    testDispatchCNOT<TestType>(re, 4);
    testDispatchCZ<TestType>(re, 4);
    testDispatchSWAP<TestType>(re, 4);
    testDispatchControlledPhaseShift<TestType>(re, 4);
    testDispatchCRX<TestType>(re, 4);
    testDispatchCRY<TestType>(re, 4);
    testDispatchCRZ<TestType>(re, 4);
    testDispatchCRot<TestType>(re, 4);
    testDispatchToffoli<TestType>(re, 4);
    testDispatchCSWAP<TestType>(re, 4);
}

// DynamicDispatcher::appyMatrix?
