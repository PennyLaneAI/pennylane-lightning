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
 * Contains utility functions for processing constants
 */
#pragma once
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <tuple>

#include "Util.hpp"

namespace Pennylane::Util {
/**
 * @brief Lookup key in array of pairs. For a constexpr map-like behavior.
 *
 * @tparam Key Type of keys
 * @tparam Value Type of values
 * @tparam size Size of std::array
 * @param arr Array to lookup
 * @param key Key to find
 */
template <typename Key, typename Value, size_t size>
constexpr auto lookup(const std::array<std::pair<Key, Value>, size> &arr,
                      const Key &key) -> Value {
    for (size_t idx = 0; idx < size; idx++) {
        if (std::get<0>(arr[idx]) == key) {
            return std::get<1>(arr[idx]);
        }
    }
    throw std::range_error("The given key does not exist.");
};

/**
 * @brief Check an array has an element.
 *
 * @tparam U Type of array elements.
 * @tparam size Size of array.
 * @param arr Array to check.
 * @param elt Element to find.
 */
template <typename U, size_t size>
constexpr auto array_has_elt(const std::array<U, size> &arr, const U &elt)
    -> bool {
    for (size_t idx = 0; idx < size; idx++) {
        if (arr[idx] == elt) {
            return true;
        }
    }
    return false;
};

/**
 * @brief Extract first elements from the array of pairs.
 *
 * @tparam T Type of the first elements.
 * @tparam U Type of the second elements.
 * @tparam size Size of the array.
 * @param arr Array to extract.
 */
template <typename T, typename U, size_t size>
constexpr std::array<T, size>
first_elts_of(const std::array<std::pair<T, U>, size> &arr) {
    // TODO: change to std::transform in C++20
    std::array<T, size> res = {
        T{},
    };
    for (size_t i = 0; i < size; i++) {
        res[i] = std::get<0>(arr[i]);
    }
    return res;
}
/**
 * @brief Extract second elements from the array of pairs.
 *
 * @tparam T Type of the first elements.
 * @tparam U Type of the second elements.
 * @tparam size Size of the array.
 * @param arr Array to extract.
 */
template <typename T, typename U, size_t size>
constexpr std::array<U, size>
second_elts_of(const std::array<std::pair<T, U>, size> &arr) {
    // TODO: change to std::transform in C++20
    std::array<U, size> res = {
        U{},
    };
    for (size_t i = 0; i < size; i++) {
        res[i] = std::get<1>(arr[i]);
    }
    return res;
}

/**
 * @brief Count the number of unique elements in the array.
 *
 * Current runtime is O(n^2).
 * TODO: count using sorted array in C++20 using constexpr std::sort.
 *
 * @tparam T Type of array elements
 * @tparam size Size of the array
 * @return size_t
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
            res++;
        }
    }
    return res;
}

/// @cond DEV
namespace Internal {
/**
 * @brief Helper function for prepend_to_tuple
 */
template <class T, class Tuple, std::size_t... I>
constexpr auto
prepend_to_tuple_helper(T &&elt, Tuple &&t,
                        [[maybe_unused]] std::index_sequence<I...> dummy) {
    return std::make_tuple(elt, std::get<I>(std::forward<Tuple>(t))...);
}
} // namespace Internal
/// @endcond

/**
 * @brief Prepend an element to a tuple
 * @tparam T Type of element
 * @tparam Tuple Type of the tuple (usually std::tuple)
 *
 * @param elt Element to prepend
 * @param t Tuple to add an element
 */
template <class T, class Tuple>
constexpr auto prepend_to_tuple(T &&elt, Tuple &&t) {
    return Internal::prepend_to_tuple_helper(
        std::forward<T>(elt), std::forward<Tuple>(t),
        std::make_index_sequence<
            std::tuple_size_v<std::remove_reference_t<Tuple>>>{});
}

/**
 * @brief Transform a tuple to an array
 *
 * This function only works when all elements of the tuple are the same
 * type or convertible to the same type.
 *
 * @tparam T Type of the elements. This type usually needs to be specified.
 * @tparam Tuple Type of the tuple.
 * @param tuple Tuple to transform
 */
template <class Tuple> constexpr auto tuple_to_array(Tuple &&tuple) {
    using T = std::tuple_element_t<0, remove_cvref_t<Tuple>>;
    return std::apply(
        [](auto... n) { return std::array<T, sizeof...(n)>{n...}; },
        std::forward<Tuple>(tuple));
}

/// @cond DEV
namespace Internal {
/**
 * @brief Helper function for prepend_to_tuple
 */
template <class T, class U, size_t size, std::size_t... I>
constexpr auto
reverse_pairs_helper(const std::array<std::pair<T, U>, size> &arr,
                     [[maybe_unused]] std::index_sequence<I...> dummy) {
    return std::array{std::pair{arr[I].second, arr[I].first}...};
}
} // namespace Internal
/// @endcond

/**
 * @brief Swap positions of elements in each pair
 *
 * @tparam T Type of first elements
 * @tparam U Type of second elements
 * @tparam size Size of the array
 */
template <class T, class U, size_t size>
constexpr auto reverse_pairs(const std::array<std::pair<T, U>, size> &arr)
    -> std::array<std::pair<U, T>, size> {
    return Internal::reverse_pairs_helper(arr,
                                          std::make_index_sequence<size>{});
}
} // namespace Pennylane::Util
