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
/**
 * @file
 * Define keys to select kernels
 */
#pragma once

#include "Macros.hpp"

#include <cstdint>

#ifdef PL_USE_OMP
#include <omp.h>
#endif

namespace Pennylane {
enum class Threading : uint8_t {
    SingleThread,
    MultiThread,
    END,
    BEGIN = SingleThread,
};

enum class CPUMemoryModel : uint8_t {
    Unaligned,
    Aligned256,
    Aligned512,
    END,
    BEGIN = Unaligned,
};

constexpr uint32_t toDispatchKey(Threading threading,
                                 CPUMemoryModel memory_model) {
    /* Threading is in higher priority */
    return (static_cast<uint32_t>(threading) << 8U) |
           static_cast<uint32_t>(memory_model);
}

inline auto getMemoryModel(const void *ptr) -> CPUMemoryModel {
    if ((reinterpret_cast<uintptr_t>(ptr) % 64) == 0) {
        return CPUMemoryModel::Aligned512;
    }

    if ((reinterpret_cast<uintptr_t>(ptr) % 32) == 0) {
        return CPUMemoryModel::Aligned256;
    }

    return CPUMemoryModel::Unaligned;
}

/**
 * @brief Choose the best threading based on the current context.
 */
inline auto bestThreading() -> Threading {
#ifdef PL_USE_OMP
    if (omp_in_parallel() != 0) {
        // We are already inside of the openmp parallel region (e.g.
        // inside adjoint diff).
        return Threading::SingleThread;
    }
    return Threading::MultiThread;
#endif
    return Threading::SingleThread;
}

constexpr inline auto bestCPUMemoryModel() -> CPUMemoryModel {
    if constexpr (use_avx512f) {
        return CPUMemoryModel::Aligned512;
    } else if (use_avx2) {
        return CPUMemoryModel::Aligned256;
    }
    return CPUMemoryModel::Unaligned;
}

} // namespace Pennylane
