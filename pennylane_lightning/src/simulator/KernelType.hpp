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

#include <array>
#include <string>

namespace Pennylane {
enum class KernelType {PI, LM};

constexpr std::array<KernelType, 2> KERNELS_TO_PYEXPORT = {
    KernelType::PI, KernelType::LM,
};

constexpr auto kernel_to_string(KernelType kernel) -> std::string_view {
    switch(kernel) {
    case KernelType::PI:
        return "PI";
    case KernelType::LM:
        return "LM";
    }
    static_assert(true, "Unreachable code");
    return "";
}

} // namespace Pennylane

