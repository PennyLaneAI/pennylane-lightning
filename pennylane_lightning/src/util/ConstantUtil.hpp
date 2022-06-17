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

#include "TypeTraits.hpp"
#include "Util.hpp"

#include <algorithm>
#include <array>
#include <compare>
#include <cstdlib>
#include <stdexcept>
#include <tuple>

#if __has_include(<version>)
#include <version>
#endif

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
}

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
}

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
    std::array<T, size> res = {
        T{},
    };
    std::transform(arr.begin(), arr.end(), res.begin(),
                   [](const auto &elt) { return std::get<0>(elt); });
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
    std::array<U, size> res = {
        U{},
    };
    std::transform(arr.begin(), arr.end(), res.begin(),
                   [](const auto &elt) { return std::get<1>(elt); });
    return res;
}

/**
 * @brief Count the number of unique elements in the array.
 *
 * This is O(n^2) version for all T
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

#if __cpp_lib_three_way_comparison >= 201907L
/**
 * @brief Count the number of unique elements in the array.
 *
 * This is a specialized version for partially ordered type T.
 *
 * @tparam T Type of array elements
 * @tparam size Size of the array
 * @return size_t
 */
template <std::three_way_comparable T, size_t size>
constexpr size_t count_unique(const std::array<T, size> &arr) {
    auto arr_cpd = arr;
    size_t dup_cnt = 0;
    std::sort(std::begin(arr_cpd), std::end(arr_cpd));
    for (size_t i = 0; i < size - 1; i++) {
        if (arr_cpd[i] == arr_cpd[i + 1]) {
            dup_cnt++;
        }
    }
    return size - dup_cnt;
}
#endif

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
    using T = std::tuple_element_t<0, std::remove_cvref_t<Tuple>>;
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
 * @param arr Array to reverse
 * @return reversed array
 */
template <class T, class U, size_t size>
constexpr auto reverse_pairs(const std::array<std::pair<T, U>, size> &arr)
    -> std::array<std::pair<U, T>, size> {
    return Internal::reverse_pairs_helper(arr,
                                          std::make_index_sequence<size>{});
}

/**
 * @brief For lookup from any array of pair whose first elements are
 * GateOperation.
 *
 * As Util::lookup can be used in constexpr context, this function is redundant
 * (by the standard). But GCC 9 still does not accept Util::lookup in constexpr
 * some cases.
 *
 * As we now move to GCC>=10, this function is deprecated.
 */
template <auto op, class T, size_t size>
constexpr auto
static_lookup(const std::array<std::pair<decltype(op), T>, size> &arr) -> T {
    for (size_t idx = 0; idx < size; idx++) {
        if (std::get<0>(arr[idx]) == op) {
            return std::get<1>(arr[idx]);
        }
    }
    return T{};
}
} // namespace Pennylane::Util
