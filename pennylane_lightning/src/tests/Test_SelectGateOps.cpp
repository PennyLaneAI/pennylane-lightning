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

template<class fp_t, KernelType kernel>
std::enable_if_t<SelectGateOps<fp_t, kernel>::kernel_id == kernel, void> checkKernelId() {
    static_assert(SelectGateOps<fp_t, kernel>::kernel_id == kernel);
}

template<class fp_t, KernelType kernel>
std::enable_if_t<SelectGateOps<fp_t, kernel>::kernel_id != kernel, void> checkKernelId() {
    static_assert(SelectGateOps<fp_t, kernel>::kernel_id == kernel);
}

template<class fp_t, size_t idx>
void checkAllAvailableKernelsIter() {
    if constexpr (idx == AVAILABLE_KERNELS.size()) {
        // do nothing
    } else {
        checkKernelId<fp_t, std::get<0>(AVAILABLE_KERNELS[idx])>();
        checkAllAvailableKernelsIter<fp_t, idx+1>();
    }
}

template<class fp_t>
void checkAllAvailableKernels() {
    checkAllAvailableKernelsIter<fp_t, 0>();
}

TEMPLATE_TEST_CASE("SelectGateOps", "[SelectGateOps]", float, double) {
    checkAllAvailableKernels<TestType>();
}
