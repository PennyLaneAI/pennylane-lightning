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
#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>

#include "ConstantUtil.hpp"
#include "TypeList.hpp"

/* Apple clang does not support std::aligned_alloc in Mac 10.14 */

namespace Pennylane {
/**
 * @brief Custom aligned allocate function. As appleclang does not support
 * std::aligned_alloc in Mac OS 10.14, we use posix memalign
 */
inline auto alignedAlloc(uint32_t alignment, size_t bytes) -> void* {
#if defined(__clang__) // probably AppleClang
    void* p;
    posix_memalign(&p, alignment, bytes);
    return p;
#elif  defined(_MSC_VER)
    return _aligned_malloc(bytes, alignment);
#else
    return std::aligned_alloc(alignment, bytes);
#endif
}

inline void alignedFree(void* p) {
#if defined(__clang__)
    return free(p);
#elif  defined(_MSC_VER)
    return _aligned_free(p);
#else
    return std::free(p);
#endif
}

template <class T, uint32_t alignment> struct AlignedAllocator {
    static_assert(Util::constIsPerfectPowerOf2(alignment),
                  "Template parameter alignment must be power of 2.");
    using value_type = T;

    AlignedAllocator() = default;

    template <class U> struct rebind {
        using other = AlignedAllocator<U, alignment>;
    };

    template <typename U>
    explicit constexpr AlignedAllocator(
        [[maybe_unused]] const AlignedAllocator<U, alignment> &rhs) noexcept {}

    [[nodiscard]] T *allocate(std::size_t size) {
        if (size == 0) {
            return nullptr;
        }
        void *p = alignedAlloc(alignment, sizeof(T) * size);
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(p);
    }

    void deallocate(T *p, [[maybe_unused]] std::size_t size) noexcept {
        alignedFree(p);
    }

    template <class U> void construct(U *ptr) { ::new ((void *)ptr) U(); }

    template <class U> void destroy(U *ptr) {
        (void)ptr;
        ptr->~U();
    }
};

template <class T, class U, uint32_t alignment>
bool operator==([[maybe_unused]] const AlignedAllocator<T, alignment> &lhs,
                [[maybe_unused]] const AlignedAllocator<U, alignment> &rhs) {
    return true;
}

template <class T, class U, uint32_t alignment>
bool operator!=([[maybe_unused]] const AlignedAllocator<T, alignment> &lhs,
                [[maybe_unused]] const AlignedAllocator<U, alignment> &rhs) {
    return false;
}

/**
 * @brief This function calculate the common multiplier of alignments of all
 * kernels.
 *
 * As all alignment must be a multiple of 2, we just can choose the maximum
 * alignment.
 */
template <typename TypeList> struct commonAlignmentHelper {
    constexpr static uint32_t value =
        std::max(TypeList::Type::packed_bytes,
                 commonAlignmentHelper<typename TypeList::Next>::value);
};
template <> struct commonAlignmentHelper<void> {
    constexpr static uint32_t value = 4U;
};

template <typename TypeList>
[[maybe_unused]] constexpr static size_t common_alignment =
    commonAlignmentHelper<TypeList>::value;

template <class T, uint32_t alignment>
using PLAllocator = std::conditional_t<alignment == 4, std::allocator<T>,
                                       AlignedAllocator<T, alignment>>;

} // namespace Pennylane
