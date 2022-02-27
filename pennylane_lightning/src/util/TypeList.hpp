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
 * @file TypeList.hpp
 * Define type list
 */
#pragma once

#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <utility>

namespace Pennylane::Util {
template <typename T, typename... Ts> struct TypeNode {
    using Type = T;
    using Next = TypeNode<Ts...>;
};
template <typename T> struct TypeNode<T, void> {
    using Type = T;
    using Next = void;
};
template <typename T> struct TypeNode<T> {
    using Type = T;
    using Next = void;
};

/**
 * @brief Define type list
 */
template <typename... Ts> using TypeList = TypeNode<Ts...>;

template <typename TypeList, size_t n> struct getNth {
    using Type = typename getNth<typename TypeList::Next, n - 1>::Type;
};

template <typename TypeList> struct getNth<TypeList, 0> {
    static_assert(!std::is_same_v<typename TypeList::Type, void>,
                  "The given n is larger than the length of the type list.");
    using Type = typename TypeList::Type;
};

/**
 * @brief Alias
 */
template <typename TypeList, size_t n>
using getNthType = typename getNth<TypeList, n>::Type;

template <typename TypeList> constexpr size_t length() {
    if constexpr (std::is_same_v<TypeList, void>) {
        return 0;
    } else {
        return 1 + length<typename TypeList::Next>();
    }
}

template <typename T, typename U> struct PrependToTypeList;

template <typename T, typename... Ts>
struct PrependToTypeList<T, TypeNode<Ts...>> {
    using Type = TypeNode<T, Ts...>;
};
template <typename T> struct PrependToTypeList<T, void> {
    using Type = TypeNode<T, void>;
};

} // namespace Pennylane::Util
