
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
 * Define memory models for CPU
 */
#pragma once
#include "Macros.hpp"

#include <cstdint>
#include <memory>

namespace Pennylane {
enum class CPUMemoryModel : uint8_t {
    Unaligned,
    Aligned256,
    Aligned512,
    END,
    BEGIN = Unaligned,
};

inline auto getMemoryModel(const void *ptr) -> CPUMemoryModel {
    if ((reinterpret_cast<uintptr_t>(ptr) % 64) == 0) {
        return CPUMemoryModel::Aligned512;
    }

    if ((reinterpret_cast<uintptr_t>(ptr) % 32) == 0) {
        return CPUMemoryModel::Aligned256;
    }

    return CPUMemoryModel::Unaligned;
}

constexpr inline auto bestCPUMemoryModel() -> CPUMemoryModel {
    if constexpr (use_avx512f) {
        return CPUMemoryModel::Aligned512;
    } else if (use_avx2) {
        return CPUMemoryModel::Aligned256;
    }
    return CPUMemoryModel::Unaligned;
}

template <class PrecisionT>
constexpr inline auto getAlignment(CPUMemoryModel memory_model) -> size_t {
    switch (memory_model) {
    case CPUMemoryModel::Unaligned:
        return alignof(PrecisionT);
    case CPUMemoryModel::Aligned256:
        return 32U;
    case CPUMemoryModel::Aligned512:
        return 64U;
    default:
        break;
    }
    PL_UNREACHABLE;
}

template <typename T>
auto allocateMemory(CPUMemoryModel memory_model, size_t size)
    // NOLINTNEXTLINE(modernize-avoid-c-arrays,hicpp-avoid-c-arrays)
    -> std::unique_ptr<T[]> {
    switch (memory_model) {
    case CPUMemoryModel::Unaligned:
        // NOLINTNEXTLINE(modernize-avoid-c-arrays,hicpp-avoid-c-arrays)
        return std::unique_ptr<T[]>{new T[size]};
    case CPUMemoryModel::Aligned256:
        // NOLINTNEXTLINE(modernize-avoid-c-arrays,hicpp-avoid-c-arrays)
        return std::unique_ptr<T[]>{new (std::align_val_t(32)) T[size]};
    case CPUMemoryModel::Aligned512:
        // NOLINTNEXTLINE(modernize-avoid-c-arrays,hicpp-avoid-c-arrays)
        return std::unique_ptr<T[]>{new (std::align_val_t(64)) T[size]};
    default:
        break;
    }
    PL_UNREACHABLE;
}
} // namespace Pennylane
