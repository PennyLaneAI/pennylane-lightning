#include "SelectGateOps.hpp"
#include "TestHelpers.hpp"
#include "Util.hpp"

#include <catch2/catch.hpp>

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace Pennylane;

template <class fp_t, KernelType kernel>
std::enable_if_t<SelectGateOps<fp_t, kernel>::kernel_id == kernel, void>
checkKernelId() {
    static_assert(SelectGateOps<fp_t, kernel>::kernel_id == kernel);
}

template <class fp_t, KernelType kernel>
std::enable_if_t<SelectGateOps<fp_t, kernel>::kernel_id != kernel, void>
checkKernelId() {
    static_assert(SelectGateOps<fp_t, kernel>::kernel_id == kernel);
}

template <class fp_t, size_t idx> void checkAllAvailableKernelsIter() {
    if constexpr (idx == Constant::available_kernels.size()) {
        // do nothing
    } else {
        checkKernelId<fp_t, std::get<0>(Constant::available_kernels[idx])>();
        checkAllAvailableKernelsIter<fp_t, idx + 1>();
    }
}

template <class fp_t> void checkAllAvailableKernels() {
    checkAllAvailableKernelsIter<fp_t, 0>();
}

TEMPLATE_TEST_CASE("SelectGateOps", "[SelectGateOps]", float, double) {
    SECTION(
        "Check all available gate implementations have correct kernel ids.") {
        checkAllAvailableKernels<TestType>();
    }

    SECTION("Check all gate operations have default kernels") {
        // TODO: This can be done in compile time...
        std::set<GateOperations> gate_ops_set;
        for (const auto &[gate_op, kernel] : Constant::default_kernel_for_ops) {
            gate_ops_set.emplace(gate_op);
        }
        REQUIRE(gate_ops_set.size() == static_cast<int>(GateOperations::END));
    }
}
