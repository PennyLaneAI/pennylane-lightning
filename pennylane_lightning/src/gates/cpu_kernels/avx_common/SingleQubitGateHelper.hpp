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
 * A helper class for single-qubit gates
 */
#pragma once
#include "BitUtil.hpp"
#include "ConstantUtil.hpp"

#include <cassert>
#include <complex>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace Pennylane::Gates::AVXCommon {
/// @cond DEV
template <class T, class = void>
struct HasInternalWithoutParam : std::false_type {};

template <class T>
struct HasInternalWithoutParam<
    T, std::void_t<decltype(&T::template applyInternal<0>)>> : std::true_type {
};

template <class T, class = void>
struct HasExternalWithoutParam : std::false_type {};

template <class T>
struct HasExternalWithoutParam<T, std::void_t<decltype(&T::applyExternal)>>
    : std::true_type {};

template <class T, class = void>
struct HasInternalWithParam : std::false_type {};

template <class T>
struct HasInternalWithParam<
    T, std::void_t<decltype(&T::template applyInternal<0, double>)>>
    : std::true_type {};

template <class T, class = void>
struct HasExternalWithParam : std::false_type {};

template <class T>
struct HasExternalWithParam<
    T, std::void_t<decltype(&T::template applyExternal<double>)>>
    : std::true_type {};

template <class T>
concept SingleQubitGateWithoutParam =
    HasInternalWithoutParam<T>::value && HasExternalWithoutParam<T>::value;

template <class T>
concept SingleQubitGateWithParam =
    HasInternalWithParam<T>::value && HasExternalWithParam<T>::value;

template <class T>
concept SingleQubitGate =
    SingleQubitGateWithoutParam<T> || SingleQubitGateWithParam<T>;

namespace Internal {
template <SingleQubitGateWithoutParam AVXImpl, size_t... rev_wire>
constexpr auto
InternalFunctions_Iter([[maybe_unused]] std::index_sequence<rev_wire...> dummy)
    -> decltype(auto) {
    return std::array{&AVXImpl::template applyInternal<rev_wire>...};
}

template <SingleQubitGateWithParam AVXImpl, typename ParamT, size_t... rev_wire>
constexpr auto
InternalFunctions_Iter([[maybe_unused]] std::index_sequence<rev_wire...> dummy)
    -> decltype(auto) {
    return std::array{&AVXImpl::template applyInternal<rev_wire, ParamT>...};
}

template <SingleQubitGateWithoutParam AVXImpl>
constexpr auto InternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return InternalFunctions_Iter<AVXImpl>(
        std::make_index_sequence<internal_wires>());
}

template <SingleQubitGateWithParam AVXImpl, typename ParamT>
constexpr auto InternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return InternalFunctions_Iter<AVXImpl, ParamT>(
        std::make_index_sequence<internal_wires>());
}
} // namespace Internal
/// @endcond

template <SingleQubitGateWithoutParam AVXImpl>
class SingleQubitGateWithoutParamHelper {
  public:
    using Precision = typename AVXImpl::Precision;
    using FuncType = void (*)(std::complex<Precision> *, size_t,
                              const std::vector<size_t> &, bool);
    constexpr static size_t packed_size = AVXImpl::packed_size_;

  private:
    FuncType fallback_func_;

  public:
    explicit SingleQubitGateWithoutParamHelper(FuncType fallback_func)
        : fallback_func_{fallback_func} {}

    void operator()(std::complex<Precision> *arr, const size_t num_qubits,
                    const std::vector<size_t> &wires, bool inverse) {
        assert(wires.size() == 1);

        constexpr static size_t internal_wires =
            Util::log2PerfectPower(packed_size / 2);
        constexpr static auto internal_functions =
            Internal::InternalFunctions<AVXImpl>();

        const size_t rev_wire = num_qubits - wires[0] - 1;

        if (Util::exp2(num_qubits) < packed_size / 2) {
            fallback_func_(arr, num_qubits, wires, inverse);
            return;
        }

        if (rev_wire < internal_wires) {
            auto func = internal_functions[rev_wire];
            (*func)(arr, num_qubits, inverse);
            return;
        }

        AVXImpl::applyExternal(arr, num_qubits, rev_wire, inverse);
    }
};

template <SingleQubitGateWithParam AVXImpl, typename ParamT>
class SingleQubitGateWithParamHelper {
  public:
    using Precision = typename AVXImpl::Precision;
    using FuncType = void (*)(std::complex<Precision> *, size_t,
                              const std::vector<size_t> &, bool, ParamT);
    constexpr static size_t packed_size = AVXImpl::packed_size_;

  private:
    FuncType fallback_func_;

  public:
    explicit SingleQubitGateWithParamHelper(FuncType fallback_func)
        : fallback_func_{fallback_func} {}

    void operator()(std::complex<Precision> *arr, const size_t num_qubits,
                    const std::vector<size_t> &wires, bool inverse,
                    ParamT angle) {
        assert(wires.size() == 1);

        constexpr static size_t internal_wires =
            Util::log2PerfectPower(packed_size / 2);
        constexpr static auto internal_functions =
            Internal::InternalFunctions<AVXImpl, ParamT>();

        const size_t rev_wire = num_qubits - wires[0] - 1;

        // When the size of an array is smaller than the AVX type
        if (Util::exp2(num_qubits) < packed_size / 2) {
            fallback_func_(arr, num_qubits, wires, inverse, angle);
            return;
        }

        // The gate applies within a register (packed bytes)
        if (rev_wire < internal_wires) {
            auto func = internal_functions[rev_wire];
            (*func)(arr, num_qubits, inverse, angle);
            return;
        }

        AVXImpl::applyExternal(arr, num_qubits, rev_wire, inverse, angle);
    }
};
} // namespace Pennylane::Gates::AVXCommon
