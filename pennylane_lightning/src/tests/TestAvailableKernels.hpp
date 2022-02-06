#include "AvailableKernels.hpp"
#include "Constant.hpp"
#include "KernelType.hpp"
#include "SelectKernel.hpp"
#include "Util.hpp"

namespace Pennylane::Gates {

// Define some utility function
template <typename TypeList>
constexpr auto is_available_kernel_helper(KernelType kernel) -> bool {
    if (TypeList::Type::kernel_id == kernel) {
        return true;
    }
    return is_available_kernel_helper<typename TypeList::Next>(kernel);
}
template <>
constexpr auto
is_available_kernel_helper<void>([[maybe_unused]] KernelType kernel) -> bool {
    return false;
}
/**
 * @brief Check the given kernel is in AvailableKernels.
 */
constexpr auto is_available_kernel(KernelType kernel) -> bool {
    return is_available_kernel_helper<AvailableKernels>(kernel);
}

template <size_t size>
constexpr auto
check_kernels_are_available(const std::array<KernelType, size> &arr) -> bool {
    // TODO: change to constexpr std::all_of in C++20
    // which is not constexpr in C++17.
    // NOLINTNEXTLINE (readability-use-anyofallof)
    for (const auto &kernel : arr) {
        if (!is_available_kernel(kernel)) {
            return false;
        }
    }
    return true;
}

/*******************************************************************************
 * Check all kernels in kernels_to_pyexport are available
 ******************************************************************************/

constexpr auto check_kernels_to_pyexport() -> bool {
    // TODO: change to constexpr std::any_of in C++20
    // NOLINTNEXTLINE (readability-use-anyofallof)
    for (const auto &kernel : kernels_to_pyexport) {
        if (!is_available_kernel(kernel)) {
            return false;
        }
    }
    return true;
}
static_assert(check_kernels_to_pyexport(),
              "Some of Kernels in Python export is not available.");

/*******************************************************************************
 * Check each element in kernelIdNamesPairs is unique
 ******************************************************************************/

static_assert(Util::count_unique(Util::first_elts_of(kernel_id_name_pairs)) ==
                  Util::length<AvailableKernels>(),
              "Kernel ids must be distinct.");

static_assert(Util::count_unique(Util::second_elts_of(kernel_id_name_pairs)) ==
                  Util::length<AvailableKernels>(),
              "Kernel names must be distinct.");

/*******************************************************************************
 * Check all kernels in default_kernel_for_gates are available
 ******************************************************************************/

static_assert(check_kernels_are_available(
                  Util::second_elts_of(Constant::default_kernel_for_gates)),
              "default_kernel_for_gates contains an unavailable kernel");

/*******************************************************************************
 * Check all kernels in default_kernel_for_generators are available
 ******************************************************************************/

static_assert(check_kernels_are_available(Util::second_elts_of(
                  Constant::default_kernel_for_generators)),
              "default_kernel_for_gates contains an unavailable kernel");
} // namespace Pennylane::Gates
