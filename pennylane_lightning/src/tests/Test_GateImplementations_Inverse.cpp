#include "OpToMemberFuncPtr.hpp"
#include "SelectGateOps.hpp"
#include "TestHelpers.hpp"
#include "TestKernels.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <complex>
#include <type_traits>
#include <utility>

using namespace Pennylane;

template <typename PrecisionT, typename ParamT, class GateImplementation,
          GateOperation gate_op, class RandomEngine>
void testInverseKernelGate(RandomEngine &re, size_t num_qubits) {
    using TestHelper::Approx;

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

            REQUIRE_THAT(st, Approx(ini_st).margin(1e-7));
        }
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
    }
}

template <typename PrecisionT, typename ParamT, class GateImplementation,
          class RandomEngine>
void testKernelInverses(RandomEngine &re, size_t num_qubits) {
    testKernelInversesIter<PrecisionT, ParamT, GateImplementation, 0>(
        re, num_qubits);
}

template <typename PrecisionT, typename ParamT, typename TypeList,
          class RandomEngine>
void testKernels(RandomEngine &re, size_t num_qubits) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using GateImplementation = typename TypeList::Type;
        testKernelInverses<PrecisionT, ParamT, GateImplementation>(re,
                                                                   num_qubits);
        testKernels<PrecisionT, ParamT, typename TypeList::Next>(re,
                                                                 num_qubits);
    }
}

TEMPLATE_TEST_CASE("Test inverse of gate implementations",
                   "[GateImplementations_Inverse]", float, double) {
    std::mt19937 re(1337);

    testKernels<TestType, TestType, TestKernels>(re, 5);
}
