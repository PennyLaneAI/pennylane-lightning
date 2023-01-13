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
 * A helper class for two-qubit gates
 */
#pragma once
#include "BitUtil.hpp"
#include "ConstantUtil.hpp"
#include "TypeTraits.hpp"

#include <cassert>
#include <complex>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace Pennylane::Gates::AVXCommon {
/// @cond DEV
template <class T, class = void>
struct HasInternalInternalWithoutParam : std::false_type {};

template <class T>
struct HasInternalInternalWithoutParam<
    T, std::void_t<decltype(&T::template applyInternalInternal<0, 0>)>>
    : std::true_type {};

template <class T, class = void>
struct HasInternalExternalWithoutParam : std::false_type {};

template <class T>
struct HasInternalExternalWithoutParam<
    T, std::void_t<decltype(&T::template applyInternalExternal<0>)>>
    : std::true_type {};

template <class T, class = void>
struct HasInternalExternalWithParam : std::false_type {};

template <class T>
struct HasInternalExternalWithParam<
    T, std::void_t<decltype(&T::template applyInternalExternal<0, double>)>>
    : std::true_type {};

template <class T, class = void>
struct HasExternalInternalWithParam : std::false_type {};

template <class T>
struct HasExternalInternalWithParam<
    T, std::void_t<decltype(&T::template applyExternalInternal<0, double>)>>
    : std::true_type {};

template <class T, class = void>
struct HasExternalInternalWithoutParam : std::false_type {};

template <class T>
struct HasExternalInternalWithoutParam<
    T, std::void_t<decltype(&T::template applyExternalInternal<0>)>>
    : std::true_type {};

template <class T, class = void>
struct HasExternalExternalWithParam : std::false_type {};

template <class T>
struct HasExternalExternalWithParam<
    T, std::void_t<decltype(&T::template applyExternalExternal<double>)>>
    : std::true_type {};

template <class T, class = void>
struct HasExternalExternalWithoutParam : std::false_type {};

template <class T>
struct HasExternalExternalWithoutParam<
    T, std::void_t<decltype(&T::applyExternalExternal)>> : std::true_type {};

template <class T, class = void>
struct HasInternalInternalWithParam : std::false_type {};

template <class T>
struct HasInternalInternalWithParam<
    T, std::void_t<decltype(&T::template applyInternalInternal<0, 0, double>)>>
    : std::true_type {};

template <class T>
concept SymmetricTwoQubitGateWithParam =
    T::symmetric && HasInternalInternalWithParam<T>::value &&
    HasInternalExternalWithParam<T>::value &&
    HasExternalExternalWithParam<T>::value;

template <class T>
concept AsymmetricTwoQubitGateWithParam =
    !T::symmetric && HasInternalInternalWithParam<T>::value &&
    HasInternalExternalWithParam<T>::value &&
    HasExternalInternalWithParam<T>::value &&
    HasExternalExternalWithParam<T>::value;

template <class T>
concept SymmetricTwoQubitGateWithoutParam =
    T::symmetric && HasInternalInternalWithoutParam<T>::value &&
    HasInternalExternalWithoutParam<T>::value &&
    HasExternalExternalWithoutParam<T>::value;

template <class T>
concept AsymmetricTwoQubitGateWithoutParam =
    !T::symmetric && HasInternalInternalWithoutParam<T>::value &&
    HasInternalExternalWithoutParam<T>::value &&
    HasExternalInternalWithoutParam<T>::value &&
    HasExternalExternalWithoutParam<T>::value;

template <class T>
concept TwoQubitGateWithParam =
    SymmetricTwoQubitGateWithParam<T> || AsymmetricTwoQubitGateWithParam<T>;

template <class T>
concept TwoQubitGateWithoutParam = SymmetricTwoQubitGateWithoutParam<T> ||
    AsymmetricTwoQubitGateWithoutParam<T>;

namespace Internal {
// InternalInternal
template <SymmetricTwoQubitGateWithoutParam AVXImpl, size_t control,
          size_t... target>
constexpr auto InternalInternalFunctions_IterTargets(
    [[maybe_unused]] std::index_sequence<target...> dummy) {
    return std::array{
        &AVXImpl::template applyInternalInternal<std::min(control, target),
                                                 std::max(control, target)>...};
}

template <AsymmetricTwoQubitGateWithoutParam AVXImpl, size_t control,
          size_t... target>
constexpr auto InternalInternalFunctions_IterTargets(
    [[maybe_unused]] std::index_sequence<target...> dummy) {
    return std::array{
        &AVXImpl::template applyInternalInternal<control, target>...};
}

template <SymmetricTwoQubitGateWithParam AVXImpl, size_t control,
          typename ParamT, size_t... target>
constexpr auto InternalInternalFunctions_IterTargets(
    [[maybe_unused]] std::index_sequence<target...> dummy) {
    return std::array{&AVXImpl::template applyInternalInternal<
        std::min(control, target), std::max(control, target), ParamT>...};
}

template <AsymmetricTwoQubitGateWithParam AVXImpl, size_t control,
          typename ParamT, size_t... target>
constexpr auto InternalInternalFunctions_IterTargets(
    [[maybe_unused]] std::index_sequence<target...> dummy) {
    return std::array{
        &AVXImpl::template applyInternalInternal<control, target, ParamT>...};
}

template <TwoQubitGateWithParam AVXImpl, typename ParamT, size_t... control>
constexpr auto InternalInternalFunctions_Iter(
    [[maybe_unused]] std::index_sequence<control...> dummy) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return Util::tuple_to_array(std::tuple{
        InternalInternalFunctions_IterTargets<AVXImpl, control, ParamT>(
            std::make_index_sequence<internal_wires>())...});
}

template <TwoQubitGateWithoutParam AVXImpl, size_t... control>
constexpr auto InternalInternalFunctions_Iter(
    [[maybe_unused]] std::index_sequence<control...> dummy) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return Util::tuple_to_array(
        std::tuple{InternalInternalFunctions_IterTargets<AVXImpl, control>(
            std::make_index_sequence<internal_wires>())...});
}

template <TwoQubitGateWithParam AVXImpl, typename ParamT>
constexpr auto InternalInternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return InternalInternalFunctions_Iter<AVXImpl, ParamT>(
        std::make_index_sequence<internal_wires>());
}

template <TwoQubitGateWithoutParam AVXImpl>
constexpr auto InternalInternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return InternalInternalFunctions_Iter<AVXImpl>(
        std::make_index_sequence<internal_wires>());
}

// Asymmetric two qubit gate without param begin
template <AsymmetricTwoQubitGateWithoutParam AVXImpl, size_t... targets>
constexpr auto ExternalInternalFunctions_Iter(
    [[maybe_unused]] std::index_sequence<targets...> dummy) -> decltype(auto) {
    return Util::tuple_to_array(
        std::tuple{&AVXImpl::template applyExternalInternal<targets>...});
}

template <AsymmetricTwoQubitGateWithoutParam AVXImpl>
constexpr auto ExternalInternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return ExternalInternalFunctions_Iter<AVXImpl>(
        std::make_index_sequence<internal_wires>());
}
template <AsymmetricTwoQubitGateWithParam AVXImpl, typename ParamT,
          size_t... targets>
constexpr auto ExternalInternalFunctions_Iter(
    [[maybe_unused]] std::index_sequence<targets...> dummy) -> decltype(auto) {
    return Util::tuple_to_array(std::tuple{
        &AVXImpl::template applyExternalInternal<targets, ParamT>...});
}

template <AsymmetricTwoQubitGateWithParam AVXImpl, typename ParamT>
constexpr auto ExternalInternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return ExternalInternalFunctions_Iter<AVXImpl, ParamT>(
        std::make_index_sequence<internal_wires>());
}

template <TwoQubitGateWithoutParam AVXImpl, size_t... controls>
constexpr auto InternalExternalFunctions_Iter(
    [[maybe_unused]] std::index_sequence<controls...> dummy) -> decltype(auto) {
    return std::array{&AVXImpl::template applyInternalExternal<controls>...};
}

template <TwoQubitGateWithoutParam AVXImpl>
constexpr auto InternalExternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return InternalExternalFunctions_Iter<AVXImpl>(
        std::make_index_sequence<internal_wires>());
}

template <TwoQubitGateWithParam AVXImpl, typename ParamT, size_t... controls>
constexpr auto InternalExternalFunctions_Iter(
    [[maybe_unused]] std::index_sequence<controls...> dummy) -> decltype(auto) {
    return std::array{
        &AVXImpl::template applyInternalExternal<controls, ParamT>...};
}

template <TwoQubitGateWithParam AVXImpl, typename ParamT>
constexpr auto InternalExternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return InternalExternalFunctions_Iter<AVXImpl, ParamT>(
        std::make_index_sequence<internal_wires>());
}
// Symmetric two qubit gate without param end
// Symmetric two qubit gate with param begin
template <SymmetricTwoQubitGateWithParam AVXImpl, typename ParamT,
          size_t control, size_t... target>
constexpr auto InternalInternalFunctions_IterTargets(
    [[maybe_unused]] std::index_sequence<target...> dummy) {
    return std::array{&AVXImpl::template applyInternalInternal<
        std::min(control, target), std::max(control, target), ParamT>...};
}

template <SymmetricTwoQubitGateWithParam AVXImpl, typename ParamT,
          size_t... control>
constexpr auto InternalInternalFunctions_Iter(
    [[maybe_unused]] std::index_sequence<control...> dummy) {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return Util::tuple_to_array(std::tuple{
        InternalInternalFunctions_IterTargets<AVXImpl, ParamT, control>(
            std::make_index_sequence<internal_wires>())...});
}

template <SymmetricTwoQubitGateWithParam AVXImpl, typename ParamT>
constexpr auto InternalInternalFunctions() {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return InternalInternalFunctions_Iter<AVXImpl, ParamT>(
        std::make_index_sequence<internal_wires>());
}

template <SymmetricTwoQubitGateWithParam AVXImpl, typename ParamT,
          size_t... controls>
constexpr auto InternalExternalFunctions_Iter(
    [[maybe_unused]] std::index_sequence<controls...> dummy) {
    return std::array{
        &AVXImpl::template applyInternalExternal<controls, ParamT>...};
}

template <SymmetricTwoQubitGateWithParam AVXImpl, typename ParamT>
constexpr auto InternalExternalFunctions() {
    constexpr size_t internal_wires =
        Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return InternalExternalFunctions_Iter<AVXImpl, ParamT>(
        std::make_index_sequence<internal_wires>());
}
// Symmetric two qubit gate with param end
} // namespace Internal
/// @endcond

template <class T, class ParamT> class TwoQubitGateWithParamHelper {
    static_assert(sizeof(T) == -1, "Only specialized template can be used.");
};

template <class AVXImpl>
requires TwoQubitGateWithoutParam<AVXImpl>
class TwoQubitGateWithoutParamHelper {
  public:
    using Precision = typename AVXImpl::Precision;
    using ReturnType = typename Util::FuncReturn<
        decltype(AVXImpl::applyExternalExternal)>::Type;
    using FuncType = ReturnType (*)(std::complex<Precision> *, size_t,
                                    const std::vector<size_t> &, bool);
    constexpr static size_t packed_size = AVXImpl::packed_size_;

  private:
    FuncType fallback_func_;

  public:
    explicit TwoQubitGateWithoutParamHelper(FuncType fallback_func)
        : fallback_func_{fallback_func} {}

    auto operator()(std::complex<Precision> *arr, const size_t num_qubits,
                    const std::vector<size_t> &wires, bool inverse) const
        -> ReturnType requires SymmetricTwoQubitGateWithoutParam<AVXImpl> {
        PL_ASSERT(wires.size() == 2);

        constexpr static size_t internal_wires =
            Util::log2PerfectPower(packed_size / 2);
        constexpr static auto internal_internal_functions =
            Internal::InternalInternalFunctions<AVXImpl>();

        constexpr static auto internal_external_functions =
            Internal::InternalExternalFunctions<AVXImpl>();

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1;

        if (Util::exp2(num_qubits) < packed_size / 2) {
            return fallback_func_(arr, num_qubits, wires, inverse);
        }

        if ((rev_wire0 < internal_wires) && (rev_wire1 < internal_wires)) {
            auto func = internal_internal_functions[rev_wire0][rev_wire1];
            return (*func)(arr, num_qubits, inverse);
        }

        const auto min_rev_wire = std::min(rev_wire0, rev_wire1);
        const auto max_rev_wire = std::max(rev_wire0, rev_wire1);

        if (min_rev_wire < internal_wires) {
            return (*internal_external_functions[min_rev_wire])(
                arr, num_qubits, max_rev_wire, inverse);
        }

        return AVXImpl::applyExternalExternal(arr, num_qubits, rev_wire0,
                                              rev_wire1, inverse);
    }

    auto operator()(std::complex<Precision> *arr, const size_t num_qubits,
                    const std::vector<size_t> &wires, bool inverse) const
        -> ReturnType requires AsymmetricTwoQubitGateWithoutParam<AVXImpl> {
        PL_ASSERT(wires.size() == 2);

        constexpr static size_t internal_wires =
            Util::log2PerfectPower(packed_size / 2);
        constexpr static auto internal_internal_functions =
            Internal::InternalInternalFunctions<AVXImpl>();

        constexpr static auto internal_external_functions =
            Internal::InternalExternalFunctions<AVXImpl>();

        constexpr static auto external_internal_functions =
            Internal::ExternalInternalFunctions<AVXImpl>();

        const size_t target = num_qubits - wires[1] - 1;
        const size_t control = num_qubits - wires[0] - 1;

        if (Util::exp2(num_qubits) < packed_size / 2) {
            return fallback_func_(arr, num_qubits, wires, inverse);
        }

        if ((control < internal_wires) && (target < internal_wires)) {
            auto func = internal_internal_functions[control][target];
            return (*func)(arr, num_qubits, inverse);
        }

        if (control < internal_wires) {
            return (*internal_external_functions[control])(arr, num_qubits,
                                                           target, inverse);
        }

        if (target < internal_wires) {
            return (*external_internal_functions[target])(arr, num_qubits,
                                                          control, inverse);
        }

        return AVXImpl::applyExternalExternal(arr, num_qubits, control, target,
                                              inverse);
    }
};

template <class AVXImpl, class ParamT>
requires TwoQubitGateWithParam<AVXImpl>
class TwoQubitGateWithParamHelper<AVXImpl, ParamT> {
  public:
    using Precision = typename AVXImpl::Precision;
    using ReturnType = typename Util::FuncReturn<
        decltype(AVXImpl::template applyExternalExternal<Precision>)>::Type;
    using FuncType = ReturnType (*)(std::complex<Precision> *, size_t,
                                    const std::vector<size_t> &, bool, ParamT);
    constexpr static size_t packed_size = AVXImpl::packed_size_;

  private:
    FuncType fallback_func_;

  public:
    explicit TwoQubitGateWithParamHelper(FuncType fallback_func)
        : fallback_func_{fallback_func} {}

    auto operator()(std::complex<Precision> *arr, const size_t num_qubits,
                    const std::vector<size_t> &wires, bool inverse,
                    ParamT angle) const
        -> ReturnType requires SymmetricTwoQubitGateWithParam<AVXImpl> {
        PL_ASSERT(wires.size() == 2);

        constexpr static size_t internal_wires =
            Util::log2PerfectPower(packed_size / 2);
        constexpr static auto internal_internal_functions =
            Internal::InternalInternalFunctions<AVXImpl, ParamT>();

        constexpr static auto internal_external_functions =
            Internal::InternalExternalFunctions<AVXImpl, ParamT>();

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1;

        if (Util::exp2(num_qubits) < packed_size / 2) {
            return fallback_func_(arr, num_qubits, wires, inverse, angle);
        }

        if (rev_wire0 < internal_wires && rev_wire1 < internal_wires) {
            auto func = internal_internal_functions[rev_wire0][rev_wire1];
            return (*func)(arr, num_qubits, inverse, angle);
        }

        const auto min_rev_wire = std::min(rev_wire0, rev_wire1);
        const auto max_rev_wire = std::max(rev_wire0, rev_wire1);

        if (min_rev_wire < internal_wires) {
            return (*internal_external_functions[min_rev_wire])(
                arr, num_qubits, max_rev_wire, inverse, angle);
        }
        return AVXImpl::applyExternalExternal(arr, num_qubits, rev_wire0,
                                              rev_wire1, inverse, angle);
    }

    auto operator()(std::complex<Precision> *arr, const size_t num_qubits,
                    const std::vector<size_t> &wires, bool inverse,
                    ParamT angle) const
        -> ReturnType requires AsymmetricTwoQubitGateWithParam<AVXImpl> {
        PL_ASSERT(wires.size() == 2);

        constexpr static size_t internal_wires =
            Util::log2PerfectPower(packed_size / 2);
        constexpr static auto internal_internal_functions =
            Internal::InternalInternalFunctions<AVXImpl, ParamT>();

        constexpr static auto internal_external_functions =
            Internal::InternalExternalFunctions<AVXImpl, ParamT>();

        constexpr static auto external_internal_functions =
            Internal::ExternalInternalFunctions<AVXImpl, ParamT>();

        const size_t target = num_qubits - wires[1] - 1;
        const size_t control = num_qubits - wires[0] - 1;

        if (Util::exp2(num_qubits) < packed_size / 2) {
            return fallback_func_(arr, num_qubits, wires, inverse, angle);
        }

        if ((control < internal_wires) && (target < internal_wires)) {
            auto func = internal_internal_functions[control][target];
            return (*func)(arr, num_qubits, inverse, angle);
        }

        if (control < internal_wires) {
            return (*internal_external_functions[control])(
                arr, num_qubits, target, inverse, angle);
        }

        if (target < internal_wires) {
            return (*external_internal_functions[target])(
                arr, num_qubits, control, inverse, angle);
        }

        return AVXImpl::applyExternalExternal(arr, num_qubits, control, target,
                                              inverse, angle);
    }
};
} // namespace Pennylane::Gates::AVXCommon
