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
#include "OpToMemberFuncPtr.hpp"
#include "SelectKernel.hpp"
#include "Util.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace Pennylane::Gates;
namespace Constant = Pennylane::Gates::Constant;

using Pennylane::Gates::callGateOps;

/**
 * @file This file contains tests for DynamicDispatcher class
 *
 * We just check DynamicDispacther calls the correct functuion by comparing
 * the result from it with that of the direct call.
 */

template <typename PrecisionT, typename ParamT, class GateImplementation,
          GateOperation gate_op, class RandomEngine, class Enable = void>
struct testDispatchForKernel {
    static void test(RandomEngine &re, size_t num_qubits) {
        // Keep source, but allow clang-tidy to pass for unused
        static_cast<void>(re);
        static_cast<void>(num_qubits);
    } // Do nothing if not implemented;
      // This could probably be replaced with an enable_if or SFINAE-like
      // pattern.
};
template <typename PrecisionT, typename ParamT, class GateImplementation,
          GateOperation gate_op, class RandomEngine>
struct testDispatchForKernel<
    PrecisionT, ParamT, GateImplementation, gate_op, RandomEngine,
    std::enable_if_t<Util::array_has_elt(GateImplementation::implemented_gates,
                                         gate_op)>> {
    static void test(RandomEngine &re, size_t num_qubits) {
        const auto ini_st = createRandomState<PrecisionT>(re, num_qubits);
        auto expected = ini_st;

        const auto wires = createWires(gate_op, num_qubits);
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
            std::string(Util::lookup(Constant::gate_names, gate_op));
        DynamicDispatcher<PrecisionT>::getInstance().applyOperation(
            GateImplementation::kernel_id, test_st.data(), num_qubits,
            gate_name, wires, false, params);
        REQUIRE(test_st == expected);
    }
};

constexpr auto calcMinNumWires(GateOperation gate_op) -> size_t {
    if (Util::array_has_elt(Constant::multi_qubit_gates, gate_op)) {
        return size_t{3};
    }
    return Util::lookup(Constant::gate_wires, gate_op);
}

template <typename PrecisionT, typename ParamT, class GateImplementation,
          size_t idx, class RandomEngine>
constexpr void testAllGatesForKernelIter(RandomEngine &re,
                                         size_t max_num_qubits) {
    if constexpr (idx < static_cast<int>(GateOperation::END)) {
        constexpr auto gate_op = static_cast<GateOperation>(idx);
        constexpr auto num_wires = calcMinNumWires(gate_op);
        for (size_t num_qubits = num_wires; num_qubits <= max_num_qubits;
             num_qubits++) {
            testDispatchForKernel<PrecisionT, ParamT, GateImplementation,
                                  gate_op, RandomEngine>::test(re, num_qubits);
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
    } else {
        static_cast<void>(re);
        static_cast<void>(max_num_qubits);
    }
}

template <typename PrecisionT, typename ParamT, class RandomEngine>
void testAllKernels(RandomEngine &re, size_t max_num_qubits) {
    testAllKernelsIter<PrecisionT, ParamT, Pennylane::AvailableKernels>(
        re, max_num_qubits);
}

TEMPLATE_TEST_CASE("DynamicDispatcher::applyOperation", "[DynamicDispatcher]",
                   float, double) {
    using PrecisionT = TestType;
    std::mt19937_64 re{1337};

    constexpr size_t max_num_qubits = 6;

    SECTION("Test all gates are registered for all kernels") {
        testAllKernels<TestType, TestType>(re, max_num_qubits);
    }

    SECTION("Throw an exception for a kernel not registered") {
        const size_t num_qubits = 3;
        auto st = createProductState<PrecisionT>("000");

        auto &dispatcher = DynamicDispatcher<TestType>::getInstance();

        REQUIRE_THROWS_WITH(
            dispatcher.applyOperation(Gates::KernelType::None, st.data(),
                                      num_qubits, "Toffoli", {0, 1, 2}, false),
            Catch::Contains("Cannot find"));

        REQUIRE_THROWS_WITH(dispatcher.applyOperation(
                                Gates::KernelType::None, st.data(), num_qubits,
                                GateOperation::Toffoli, {0, 1, 2}, false),
                            Catch::Contains("Cannot find"));
    }
}

TEMPLATE_TEST_CASE("DynamicDispatcher::applyGenerator", "[DynamicDispatcher]",
                   float, double) {
    using PrecisionT = TestType;
    std::mt19937_64 re{1337};

    // applyGenerator test?

    SECTION("Throw an exception for a kernel not registered") {
        const size_t num_qubits = 3;
        auto st = createProductState<PrecisionT>("000");

        auto &dispatcher = DynamicDispatcher<TestType>::getInstance();

        REQUIRE_THROWS_WITH(dispatcher.applyGenerator(Gates::KernelType::None,
                                                      st.data(), num_qubits,
                                                      "RX", {0, 1, 2}, false),
                            Catch::Contains("Cannot find"));

        REQUIRE_THROWS_WITH(dispatcher.applyGenerator(
                                Gates::KernelType::None, st.data(), num_qubits,
                                GeneratorOperation::RX, {0, 1, 2}, false),
                            Catch::Contains("Cannot find"));
    }
}

TEMPLATE_TEST_CASE("DynamicDispatcher::applyMatrix", "[DynamicDispatcher]",
                   float, double) {
    using PrecisionT = TestType;
    std::mt19937_64 re{1337};

    // applyMatrix test?

    SECTION("Throw an exception for a kernel not registered") {
        const size_t num_qubits = 3;
        auto st = createProductState<PrecisionT>("000");

        auto &dispatcher = DynamicDispatcher<TestType>::getInstance();

        std::vector<std::complex<PrecisionT>> matrix(4, 0.0);

        REQUIRE_THROWS_WITH(dispatcher.applyMatrix(Gates::KernelType::None,
                                                   st.data(), num_qubits,
                                                   matrix.data(), {0}, false),
                            Catch::Contains("is not registered") &&
                                Catch::Contains("SingleQubitOp"));
    }
}
