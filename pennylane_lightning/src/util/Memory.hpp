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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>

#include "BitUtil.hpp"
#include "TypeList.hpp"

namespace Pennylane {
/**
 * @brief Custom aligned allocate function. As appleclang does not support
 * std::aligned_alloc in Mac OS 10.14, we use posix_memalign function.
 *
 * Note that alignment must be larger than max_align_t. Otherwise, the behavior
 * is undefined.
 *
 * @param alignment Alignment value we want for the data pointer
 * @param bytes Number of bytes to allocate
 * @return Memory pointer
 */
inline auto alignedAlloc(uint32_t alignment, size_t bytes) -> void * {
#if defined(__clang__)
    /* Apple clang does not support std::aligned_alloc in Mac 10.14.
     * Thus we use Posix function instead. */
    void *p;
    posix_memalign(&p, alignment, bytes);
    return p;
#elif defined(_MSC_VER)
    return _aligned_malloc(bytes, alignment);
#else
    return std::aligned_alloc(alignment, bytes);
#endif
}

/**
 * @brief Free memory allocated by alignedAlloc.
 *
 * @param p Pointer to the memory location allocated by aligendAlloc
 */
inline void alignedFree(void *p) {
#if defined(__clang__)
    return ::free(p); // NOLINT(hicpp-no-malloc)
#elif defined(_MSC_VER)
    return _aligned_free(p);
#else
    return std::free(p);
#endif
}

/**
 * @brief C++ Allocator class for aligned memory.
 *
 * @tparam T Datatype to allocate
 */
template <class T> struct AlignedAllocator {
    uint32_t alignment_;
    using value_type = T;

    /**
     * @brief Constructor of AlignedAllocator class
     *
     * @param alignment Memory alignment we want.
     */
    constexpr explicit AlignedAllocator(uint32_t alignment)
        : alignment_{alignment} {
        // We do not check input now as it doesn't allow the constructor to be
        // a constexpr.
        // TODO: Using exception is allowed in GCC>=10
        // assert(Util::isPerfectPowerOf2(alignment));
    }

    template <class U> struct rebind { using other = AlignedAllocator<U>; };

    template <typename U>
    explicit constexpr AlignedAllocator(
        [[maybe_unused]] const AlignedAllocator<U> &rhs) noexcept
        : alignment_{rhs.alignment_} {}

    /**
     * @brief Allocate memory with for the given number of datatype T
     *
     * @param size The number of T objects for the allocation
     * @return Allocated aligned memory
     */
    [[nodiscard]] T *allocate(std::size_t size) {
        if (size == 0) {
            return nullptr;
        }
        void *p;
        if (alignment_ > alignof(std::max_align_t)) {
            p = alignedAlloc(alignment_, sizeof(T) * size);
        } else {
            // NOLINTNEXTLINE(hicpp-no-malloc)
            p = malloc(sizeof(T) * size);
        }
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        return static_cast<T *>(p);
    }

    /**
     * @brief Deallocate allocated memory
     *
     * @param p Pointer to the allocated data
     * @param size Size of the data we allocated (unused).
     */
    void deallocate(T *p, [[maybe_unused]] std::size_t size) noexcept {
        if (alignment_ > alignof(std::max_align_t)) {
            alignedFree(p);
        } else {
            // NOLINTNEXTLINE(hicpp-no-malloc)
            free(p);
        }
    }

    template <class U> void construct(U *ptr) { ::new ((void *)ptr) U(); }

    template <class U> void destroy(U *ptr) {
        (void)ptr;
        ptr->~U();
    }
};

/**
 * @brief Compare two allocators
 *
 * By [the standard](https://en.cppreference.com/w/cpp/named_req/Allocator),
 * two allocators are equal if the memory allocated by one can be deallocated
 * by the other.
 */
template <class T, class U>
bool operator==([[maybe_unused]] const AlignedAllocator<T> &lhs,
                [[maybe_unused]] const AlignedAllocator<U> &rhs) {
    return lhs.alignment_ == rhs.alignment_;
}

/**
 * @brief Compare two allocators. See `%operator==` above.
 */
template <class T, class U, uint32_t alignment>
bool operator!=([[maybe_unused]] const AlignedAllocator<T> &lhs,
                [[maybe_unused]] const AlignedAllocator<U> &rhs) {
    return lhs.alignment_ != rhs.alignment_;
}

///@cond DEV
template <typename TypeList> struct commonAlignmentHelper {
    constexpr static uint32_t value =
        std::max(TypeList::Type::packed_bytes,
                 commonAlignmentHelper<typename TypeList::Next>::value);
};
template <> struct commonAlignmentHelper<void> {
    constexpr static uint32_t value = 4U;
};
///@endcond

/**
 * @brief This function calculate the common multiplier of alignments of the
 * given kernels in TypeList.
 *
 * As all alignment must be a power of 2, we just can choose the maximum
 * alignment.
 *
 * @tparam TypeList Type list of kernels.
 */
template <typename TypeList>
[[maybe_unused]] constexpr static size_t common_alignment =
    commonAlignmentHelper<TypeList>::value;
} // namespace Pennylane
