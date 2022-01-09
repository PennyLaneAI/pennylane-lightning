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

template <typename UnaryPredicate>
constexpr size_t count_elt(int begin, int end, UnaryPredicate p) {
    size_t count = 0;
    for (int i = begin; i < end; i++) {
        if (p(i)) {
            ++count;
        }
    }
    return count;
}

template <size_t num_params, size_t idx> struct CountGatesWithNumParamsIter {
    constexpr static size_t value =
        CountGatesWithNumParamsIter<num_params, idx - 1>::value +
        int(static_lookup<static_cast<GateOperations>(idx)>(
                Constant::gate_num_params) == num_params);
};

template <size_t num_params> struct CountGatesWithNumParamsIter<num_params, 0> {
    constexpr static size_t value = 0;
};

template <size_t num_params> struct CountGatesWithNumParams {
    constexpr static size_t value = CountGatesWithNumParamsIter<
        num_params, static_cast<int>(GateOperations::END) - 1>::value;
};

/*
 * This function will be acceptible in GCC >= 10
 */
template <typename T, size_t size>
constexpr size_t count_unique(const std::array<T, size> &arr) {
    size_t res = 0;

    for (size_t i = 0; i < size; i++) {
        bool counted = false;
        for (size_t j = 0; j < i; j++) {
            if (arr[j] == arr[i]) {
                counted = true;
                break;
            }
        }
        if (!counted) {
            ++res;
        }
    }
    return res;
}

template <class fp_t, KernelType kernel, size_t num_params>
void testGateFuncPtrPair() {
    // TODO: This also can be done in compile time
    std::set<GateOperations> gate_ops_set;
    for (const auto &[gate_op, func] :
         Pennylane::Internal::GateOpsFuncPtrPairs<fp_t, fp_t, kernel,
                                                  num_params>::value) {
        gate_ops_set.emplace(gate_op);
    }

    REQUIRE(gate_ops_set.size() ==
            count_elt(
                static_cast<int>(GateOperations::BEGIN),
                static_cast<int>(GateOperations::END) - 1, /*Besides matrix*/
                [](int v) {
                    return lookup(Constant::gate_num_params,
                                  static_cast<GateOperations>(v)) == num_params;
                }));
}

template <class fp_t, size_t kernel_idx, size_t num_params>
constexpr void testGateFuncPtrPairIter() {
    using Pennylane::Internal::GateOpsFuncPtrPairs;
    if constexpr (kernel_idx < Constant::available_kernels.size()) {
        const auto kernel =
            std::get<0>(Constant::available_kernels[kernel_idx]);

        static_assert(
            count_unique(Util::first_elts_of(
                GateOpsFuncPtrPairs<fp_t, fp_t, kernel, num_params>::value)) ==
            CountGatesWithNumParams<num_params>::value);
        static_assert(
            count_unique(Util::second_elts_of(
                GateOpsFuncPtrPairs<fp_t, fp_t, kernel, num_params>::value)) ==
            CountGatesWithNumParams<num_params>::value);
    }
}

TEMPLATE_TEST_CASE("GateOpsFuncPtrPairs", "[GateOpsFuncPtrPairs]", float,
                   double) {
    testGateFuncPtrPairIter<TestType, 0, 0>();
    testGateFuncPtrPairIter<TestType, 0, 1>();
    // The following line must not be compiled
    // testGateFuncPtrPairIter<TestType, 0, 2>();
    testGateFuncPtrPairIter<TestType, 0, 3>();
    REQUIRE(true);
}
