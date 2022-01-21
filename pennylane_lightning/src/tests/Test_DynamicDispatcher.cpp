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
namespace Constant = Pennylane::Constant;

using Pennylane::Internal::callGateOps;

/**
 * @file This file contains tests for DynamicDispatcher class
 *
 * We just check DynamicDispacther calls the correct functuion by comparing
 * the result from it with that of the direct call.
 */

std::vector<size_t> createWires(GateOperation op) {
    if (array_has_elt(Constant::multi_qubit_gates, op)) {
        // if multi-qubit gates
        return {0, 1, 2};
    }
    switch (lookup(Constant::gate_wires, op)) {
    case 1:
        return {0};
    case 2:
        return {0, 1};
    case 3:
        return {0, 1, 2};
    default:
        PL_ABORT("The number of wires for a given gate is unknown.");
    }
}

template <class PrecisionT>
std::vector<PrecisionT> createParams(GateOperation op) {
    switch (lookup(Constant::gate_num_params, op)) {
    case 0:
        return {};
    case 1:
        return {0.312};
    case 3:
        return {0.128, -0.563, 1.414};
    default:
        PL_ABORT("The number of parameters for a given gate is unknown.");
    }
}

template <typename PrecisionT, typename ParamT, class GateImplementation>
struct testDispatchForKernel {
    template <GateOperation gate_op, class RandomEngine,
              std::enable_if_t<
                  array_has_elt(GateImplementation::implemented_gates, gate_op),
                  bool> = true>
    static void test(RandomEngine &re, size_t num_qubits) {
        using CFP_t = std::complex<PrecisionT>;
        const std::vector<CFP_t> ini_st =
            create_random_state<PrecisionT>(re, num_qubits);
        std::vector<CFP_t> expected = ini_st;

        const auto wires = createWires(gate_op);
        const auto params = createParams<PrecisionT>(gate_op);

        // We first calculate expected directly calling a static member function
        // in the GateImplementation
        auto gate_func =
            GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                  gate_op>::value;
        callGateOps(gate_func, expected.data(), num_qubits, wires, false,
                    params);

        // and compare it to the dynamic dispatcher
        auto test_st = ini_st;
        const auto gate_name =
            std::string(static_lookup<gate_op>(Constant::gate_names));
        DynamicDispatcher<PrecisionT>::getInstance().applyOperation(
            GateImplementation::kernel_id, test_st.data(), num_qubits,
            gate_name, wires, false, params);
        REQUIRE(isApproxEqual(test_st, expected));
    }

    template <
        GateOperation gate_op, class RandomEngine,
        std::enable_if_t<!array_has_elt(GateImplementation::implemented_gates,
                                        gate_op),
                         bool> = true>
    static void test(RandomEngine &re, size_t num_qubits) {
    } // Do nothing if not implemented
};

template <typename PrecisionT, typename ParamT, class GateImplementation,
          size_t idx, class RandomEngine>
constexpr void testAllGatesForKernelIter(RandomEngine &re,
                                         size_t max_num_qubits) {
    if constexpr (idx < static_cast<int>(GateOperation::END) - 1) {
        constexpr auto gate_op = static_cast<GateOperation>(idx);
        for (size_t num_qubits = 3; num_qubits <= max_num_qubits;
             num_qubits++) {
            testDispatchForKernel<PrecisionT, ParamT, GateImplementation>::
                template test<gate_op>(re, num_qubits);
        }

        testAllGatesForKernelIter<PrecisionT, ParamT, GateImplementation,
                                  idx + 1>(re, max_num_qubits);
    }
}

template <typename PrecisionT, typename ParamT, class GateImplementation,
          class RandomEngine>
void testAllGatesForKernel(RandomEngine &re, size_t max_num_qubits) {
    testAllGatesForKernelIter<PrecisionT, ParamT, GateImplementation, 0>(
        re, max_num_qubits);
}

template <typename PrecisionT, typename ParamT, typename TypeList,
          class RandomEngine>
void testAllKernelsIter(RandomEngine &re, size_t max_num_qubits) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        testAllGatesForKernel<PrecisionT, ParamT, typename TypeList::Type>(
            re, max_num_qubits);

        testAllKernelsIter<PrecisionT, ParamT, typename TypeList::Next,
                           RandomEngine>(re, max_num_qubits);
    }
}

template <typename PrecisionT, typename ParamT, class RandomEngine>
void testAllKernels(RandomEngine &re, size_t max_num_qubits) {
    testAllKernelsIter<PrecisionT, ParamT, AvailableKernels>(re,
                                                             max_num_qubits);
}

TEMPLATE_TEST_CASE("DynamicDispatcher::applyOperation", "[DynamicDispatcher]",
                   float, double) {
    std::mt19937_64 re{1337};

    constexpr size_t max_num_qubits = 6;

    testAllKernels<TestType, TestType>(re, max_num_qubits);
}

// DynamicDispatcher::appyMatrix?
