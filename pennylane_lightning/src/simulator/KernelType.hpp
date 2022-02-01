// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file
 * Defines possible kernel types as enum and define python export.
 */
#pragma once
#include "Error.hpp"
#include "Util.hpp"

#include <array>
#include <string>
#include <string_view>
#include <utility>

namespace Pennylane {
enum class KernelType { PI, LM, Unknown };

namespace Constant {
constexpr std::array available_kernels = {
    std::pair<KernelType, std::string_view>{KernelType::LM, "LM"},
    std::pair<KernelType, std::string_view>{KernelType::PI, "PI"},
};

[[maybe_unused]] constexpr std::array kernels_to_pyexport = {KernelType::PI,
                                                             KernelType::LM};
} // namespace Constant

constexpr auto string_to_kernel(std::string_view str) -> KernelType {
    for (const auto &[k, v] : Constant::available_kernels) {
        if (v == str) {
            return k;
        }
    }
    // TODO: Throw exception or call PL_ABORT in GCC >= 9
    return KernelType::Unknown;
}

/// @cond DEV
/*******************************************************************************
 * The functions below are only used in the compile time to check
 * internal consistency.
 ******************************************************************************/
constexpr auto is_available_kernel(KernelType kernel) -> bool {
    // TODO: change to constexpr std::any_of in C++20
    // NOLINTNEXTLINE (readability-use-anyofallof)
    for (const auto &[avail_kernel, avail_kernel_name] :
         Constant::available_kernels) {
        if (kernel == avail_kernel) {
            return true;
        }
    }
    return false;
}

constexpr auto check_kernels_to_pyexport() -> bool {
    // TODO: change to constexpr std::any_of in C++20
    // NOLINTNEXTLINE (readability-use-anyofallof)
    for (const auto &kernel : Constant::kernels_to_pyexport) {
        if (!is_available_kernel(kernel)) {
            return false;
        }
    }
    return true;
}
static_assert(check_kernels_to_pyexport(),
              "Some of Kernels in Python export is not available.");
/// @endcond
} // namespace Pennylane
