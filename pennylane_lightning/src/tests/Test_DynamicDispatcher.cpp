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

using Pennylane::Internal::callGateOps;
using Pennylane::Internal::GateOpsFuncPtrPairs;

/**
 * @file This file contains tests for DynamicDispatcher class
 */

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

template <class PrecisionT>
std::vector<PrecisionT> createParams(GateOperations op) {
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

template <typename PrecisionT, typename ParamT, KernelType kernel>
struct testDispatchForKernel {
    template <GateOperations gate_op, class RandomEngine>
    static void test(RandomEngine &re, size_t num_qubits) {
        using CFP_t = std::complex<PrecisionT>;
        std::vector<CFP_t> ini_st =
            create_random_state<PrecisionT>(re, num_qubits);
        std::vector<CFP_t> expected = ini_st;

        const auto wires = createWires(gate_op);
        const auto params = createParams<PrecisionT>(gate_op);

        constexpr size_t num_params =
            static_lookup<gate_op>(Constant::gate_num_params);

        auto gate_func = static_lookup<gate_op>(
            GateOpsFuncPtrPairs<PrecisionT, ParamT, kernel, num_params>::value);

        callGateOps(gate_func, expected.data(), num_qubits, wires, false,
                    params);

        const auto gate_name =
            std::string(static_lookup<gate_op>(Constant::gate_names));

        auto test_st = ini_st;
        if constexpr (array_has_elt(
                          SelectGateOps<PrecisionT, kernel>::implemented_gates,
                          gate_op)) {
            DynamicDispatcher<PrecisionT>::getInstance().applyOperation(
                kernel, test_st.data(), num_qubits, gate_name, wires, false,
                params);
            REQUIRE(isApproxEqual(test_st, expected));
        } else {
            REQUIRE_THROWS(
                DynamicDispatcher<PrecisionT>::getInstance().applyOperation(
                    kernel, test_st.data(), num_qubits, gate_name.c_str(),
                    wires, false, params));
        }
    }
};

template <typename PrecisionT, typename ParamT, KernelType kernel, size_t idx,
          class RandomEngine>
constexpr void testAllGatesForKernelIter(RandomEngine &re,
                                         size_t max_num_qubits) {
    if constexpr (idx < static_cast<int>(GateOperations::END) - 1) {
        constexpr auto gate_op = static_cast<GateOperations>(idx);
        for (size_t num_qubits = static_lookup<gate_op>(Constant::gate_wires);
             num_qubits <= max_num_qubits; num_qubits++) {
            testDispatchForKernel<PrecisionT, ParamT,
                                  kernel>::template test<gate_op>(re,
                                                                  num_qubits);
        }

        testAllGatesForKernelIter<PrecisionT, ParamT, kernel, idx + 1>(
            re, max_num_qubits);
    }
}

template <typename PrecisionT, typename ParamT, KernelType kernel,
          class RandomEngine>
void testAllGatesForKernel(RandomEngine &re, size_t max_num_qubits) {
    testAllGatesForKernelIter<PrecisionT, ParamT, kernel, 0>(re,
                                                             max_num_qubits);
}

template <typename PrecisionT, typename ParamT, size_t idx, class RandomEngine>
void testAllKernelsIter(RandomEngine &re, size_t max_num_qubits) {
    if constexpr (idx < Constant::available_kernels.size()) {
        testAllGatesForKernel<PrecisionT, ParamT,
                              std::get<0>(Constant::available_kernels[idx])>(
            re, max_num_qubits);

        testAllKernelsIter<PrecisionT, ParamT, idx + 1, RandomEngine>(
            re, max_num_qubits);
    }
}

template <typename PrecisionT, typename ParamT, class RandomEngine>
void testAllKernels(RandomEngine &re, size_t max_num_qubits) {
    testAllKernelsIter<PrecisionT, ParamT, 0>(re, max_num_qubits);
}

TEMPLATE_TEST_CASE("DynamicDispatcher::applyOperation", "[DynamicDispatcher]",
                   float, double) {
    std::mt19937_64 re{1337};

    constexpr size_t max_num_qubits = 6;

    testAllGatesForKernel<TestType, TestType, KernelType::PI>(re,
                                                              max_num_qubits);
}

// DynamicDispatcher::appyMatrix?
