#pragma once
/**
 * @brief We define test kernels. Note that kernels not registered to
 * AvailableKernels can be also tested by adding them here.
 */
#include "AvailableKernels.hpp"

#include <utility>

using TestKernels =
    Pennylane::Util::TypeList<Pennylane::Gates::GateImplementationsLM,
                              Pennylane::Gates::GateImplementationsPI, void>;

namespace detail {
template <size_t... Is>
constexpr auto
testKernelIdsHelper([[maybe_unused]] std::index_sequence<Is...> indices) {
    return std::array{
        Pennylane::Util::getNthType<TestKernels, Is>::kernel_id...};
}
} // namespace detail

[[maybe_unused]] constexpr static auto test_kernel_ids =
    detail::testKernelIdsHelper(
        std::make_index_sequence<Pennylane::Util::length<TestKernels>()>());
