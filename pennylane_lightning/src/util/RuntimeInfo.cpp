// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "RuntimeInfo.hpp"
#include "Macros.hpp"

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#include <cpuid.h>
#elif defined(_MSC_VER) && defined(_WIN64)
#include <array>
#include <intrin.h>
#endif

namespace Pennylane::Util {
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
RuntimeInfo::InternalRuntimeInfo::InternalRuntimeInfo() {
    const auto nids = __get_cpuid_max(0x00, nullptr);

    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;

    if (nids >= 1) {
        eax = 1;
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
        f_1_ecx = ecx;
        f_1_edx = edx;
    }
    if (nids >= 7) { // NOLINT(readability-magic-numbers)
        // NOLINTNEXTLINE(readability-magic-numbers)
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        f_7_ebx = ebx;
        f_7_ecx = ecx;
    }
}
#elif defined(_MSC_VER) && defined(_M_AMD64)
RuntimeInfo::InternalRuntimeInfo::InternalRuntimeInfo() {
    std::array<int, 4> cpui;
    __cpuid(cpui.data(), 0);

    int nids = cpui[0];

    if (nids >= 1) {
        __cpuidex(cpui.data(), 1, 0);
        f_1_ecx = cpui[2];
        f_1_edx = cpui[3];
    }

    if (nids >= 7) {
        __cpuidex(cpui.data(), 7, 0);
        f_7_ebx = cpui[1];
        f_7_ecx = cpui[2];
    }
}
#else
RuntimeInfo::InternalRuntimeInfo::InternalRuntimeInfo(){};
#endif
} // namespace Pennylane::Util
