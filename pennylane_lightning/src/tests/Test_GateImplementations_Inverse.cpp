#include "OpToMemberFuncPtr.hpp"
#include "SelectKernel.hpp"
#include "TestHelpers.hpp"
#include "TestKernels.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <complex>
#include <type_traits>
#include <utility>

/**
 * @file
 * We test inverse of each gate operation here. For all gates in
 * implemented_gates, we test wether the state after applying an operation and
 * its inverse is the same as the initial state.
 *
 * Note the we only test generators only when it is included in
 * constexpr member variable implemented_generators.
 */

using namespace Pennylane;
using namespace Pennylane::Gates;

template <typename PrecisionT, typename ParamT, class GateImplementation,
          GateOperation gate_op, class RandomEngine>
void testInverseKernelGate(RandomEngine &re, size_t num_qubits) {
    if constexpr (gate_op != GateOperation::Matrix) {
        constexpr auto gate_name = static_lookup<gate_op>(Constant::gate_names);
        DYNAMIC_SECTION("Test inverse of " << gate_name << " for kernel "
                                           << GateImplementation::name) {
            const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);

            auto st = ini_st;

            const auto func_ptr =
                GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
                                      gate_op>::value;

            const auto wires = createWires(gate_op);
            const auto params = createParams<ParamT>(gate_op);

            callGateOps(func_ptr, st.data(), num_qubits, wires, false, params);
            callGateOps(func_ptr, st.data(), num_qubits, wires, true, params);

            REQUIRE(st == PLApprox(ini_st).margin(1e-7));
        }
    } else {
        static_cast<void>(re);
        static_cast<void>(num_qubits);
    }
}

template <typename PrecisionT, typename ParamT, class GateImplementation,
          size_t gate_idx, class RandomEngine>
void testKernelInversesIter(RandomEngine &re, size_t num_qubits) {
    if constexpr (gate_idx < GateImplementation::implemented_gates.size()) {
        testInverseKernelGate<PrecisionT, ParamT, GateImplementation,
                              GateImplementation::implemented_gates[gate_idx]>(
            re, num_qubits);
        testKernelInversesIter<PrecisionT, ParamT, GateImplementation,
                               gate_idx + 1>(re, num_qubits);
    } else {
        static_cast<void>(re);
        static_cast<void>(num_qubits);
    }
}

template <typename PrecisionT, typename ParamT, class GateImplementation,
          class RandomEngine>
void testKernelInverses(RandomEngine &re, size_t num_qubits) {
    testKernelInversesIter<PrecisionT, ParamT, GateImplementation, 0>(
        re, num_qubits);
}

template <
    typename PrecisionT, typename ParamT, typename TypeList,
    class RandomEngine> //, typename
                        // std::enable_if<std::is_same<TypeList,void>::value>
                        //>
void testKernels(RandomEngine &re, size_t num_qubits) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using GateImplementation = typename TypeList::Type;
        testKernelInverses<PrecisionT, ParamT, GateImplementation>(re,
                                                                   num_qubits);
        testKernels<PrecisionT, ParamT, typename TypeList::Next>(re,
                                                                 num_qubits);
    } else {
        static_cast<void>(re);
        static_cast<void>(num_qubits);
    }
}

TEMPLATE_TEST_CASE("Test inverse of gate implementations",
                   "[GateImplementations_Inverse]", float, double) {
    std::mt19937 re(1337);

    testKernels<TestType, TestType, TestKernels>(re, 5);
}
