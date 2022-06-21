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
#include "Macros.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "Util.hpp"

/* Kernels */
#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

#include "TestHelpers.hpp"

using namespace Pennylane;
using namespace Pennylane::Gates;
namespace Constant = Pennylane::Gates::Constant;

using Pennylane::Gates::callGateOps;

/**
 * @file This file contains tests for DynamicDispatcher class
 *
 * We just check DynamicDispatcher calls the correct function by comparing
 * the result from it with that of the direct call.
 */

TEMPLATE_TEST_CASE("Print registered kernels", "[DynamicDispatcher]", float,
                   double) {
    using Pennylane::Util::operator<<;
    const auto &dispatcher = DynamicDispatcher<TestType>::getInstance();
    const auto kernels = dispatcher.registeredKernels();

    std::ostringstream ss;
    ss << "Registered kernels: ";
    for (size_t n = 0; n < kernels.size(); n++) {
        ss << dispatcher.getKernelName(kernels[n]);
        if (n != kernels.size() - 1) {
            ss << ", ";
        }
    }
    WARN(ss.str());
    REQUIRE(true);
}

TEMPLATE_TEST_CASE("DynamicDispatcher::applyOperation", "[DynamicDispatcher]",
                   float, double) {
    using PrecisionT = TestType;

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
