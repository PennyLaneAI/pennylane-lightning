#include "DynamicDispatcher.hpp"
#include "OpToMemberFuncPtr.hpp"
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

template <typename PrecisionT, class RandomEngine>
void testInverseGateKernel(RandomEngine &re, KernelType kernel,
                           GateOperation gate_op, size_t num_qubits) {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    const auto gate_name = Util::lookup(Constant::gate_names, gate_op);
    const auto kernel_name = dispatcher.getKernelName(kernel);

    DYNAMIC_SECTION("Test inverse of " << gate_name << " for kernel "
                                       << kernel_name) {
        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);

        auto st = ini_st;

        const auto wires = createWires(gate_op, num_qubits);
        const auto params = createParams<PrecisionT>(gate_op);

        dispatcher.applyOperation(kernel, st.data(), num_qubits, gate_op, wires,
                                  false, params);
        dispatcher.applyOperation(kernel, st.data(), num_qubits, gate_op, wires,
                                  true, params);

        REQUIRE(st == approx(ini_st).margin(PrecisionT{1e-7}));
    }
}

template <typename PrecisionT, class RandomEngine>
void testInverseForAllGatesKernel(RandomEngine &re, KernelType kernel,
                                  size_t num_qubits) {
    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    for (const auto gate_op : dispatcher.registeredGatesForKernel(kernel)) {
        testInverseGateKernel<PrecisionT>(re, kernel, gate_op, num_qubits);
    }
}

TEMPLATE_TEST_CASE("Test inverse of gate implementations",
                   "[GateImplementations_Inverse]", float, double) {
    using PrecisionT = TestType;
    std::mt19937 re(1337);

    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();
    for (const auto kernel : dispatcher.registeredKernels()) {
        testInverseForAllGatesKernel<PrecisionT>(re, kernel, 5);
    }
}
