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
#include "ConstantUtil.hpp"
#include "BitUtil.hpp"

#include <cassert>
#include <cstdlib>
#include <complex>
#include <vector>
#include <type_traits>

namespace Pennylane::Gates::AVX {

template <class T, class = void>
struct HasInternalInternalWithoutParam : std::false_type {};

template <class T>
struct HasInternalInternalWithoutParam<T, std::void_t<decltype(&T::template applyInternalInternal<0, 0>)>> : std::true_type {};

template <class T, class = void>
struct HasInternalExternalWithoutParam : std::false_type {};

template <class T>
struct HasInternalExternalWithoutParam<T, std::void_t<decltype(&T::template applyInternalExternal<0>)>> : std::true_type {};

template <class T, class = void>
struct HasInternalExternalWithParam : std::false_type {};

template <class T>
struct HasInternalExternalWithParam<T, std::void_t<decltype(&T::template applyInternalExternal<0, double>)>> : std::true_type {};

template <class T, class = void>
struct HasInternalInternalWithParam : std::false_type {};

template <class T>
struct HasInternalInternalWithParam<T, std::void_t<decltype(&T::template applyInternalInternal<0, 0, double>)>> : std::true_type {};


template <class T>
concept SymmetricTwoQubitGateWithParam = T::symmetric && HasInternalInternalWithParam<T>::value && HasInternalExternalWithParam<T>::value;

template <class T>
concept SymmetricTwoQubitGateWithoutParam = T::symmetric && HasInternalInternalWithoutParam<T>::value && HasInternalExternalWithoutParam<T>::value;


namespace Internal {
template <SymmetricTwoQubitGateWithoutParam AVXImpl, size_t control, size_t... target>
constexpr auto InternalInternalFunctions_IterTargets([[maybe_unused]] std::index_sequence<target...> dummy) {
    return std::array{
            &AVXImpl::template applyInternalInternal<std::min(control, target), 
                                                     std::max(control, target)>...
    };
}

template <SymmetricTwoQubitGateWithoutParam AVXImpl, size_t... control>
constexpr auto InternalInternalFunctions_Iter([[maybe_unused]] std::index_sequence<control...> dummy) {
    constexpr size_t internal_wires = Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return Util::tuple_to_array(std::tuple{
            InternalInternalFunctions_IterTargets<AVXImpl, control>(
                    std::make_index_sequence<internal_wires>())...
        });
}

template <SymmetricTwoQubitGateWithoutParam AVXImpl>
constexpr auto InternalInternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires = Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return InternalInternalFunctions_Iter<AVXImpl>(std::make_index_sequence<internal_wires>());
}

template <SymmetricTwoQubitGateWithoutParam AVXImpl, size_t... controls>
constexpr auto InternalExternalFunctions_Iter([[maybe_unused]] std::index_sequence<controls...> dummy) -> decltype(auto) {
    return std::array{
        &AVXImpl::template applyInternalExternal<controls>...
    };
}

template <SymmetricTwoQubitGateWithoutParam AVXImpl>
constexpr auto InternalExternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires = Util::log2PerfectPower(AVXImpl::packed_size_ / 2);
    return InternalExternalFunctions_Iter<AVXImpl>(std::make_index_sequence<internal_wires>());
}

template <typename AVXImpl, size_t... targets>
constexpr auto ExternalInternalFunctions_Iter([[maybe_unused]] std::index_sequence<targets...> dummy) -> decltype(auto) {
    return Util::tuple_to_array(std::tuple{
        &AVXImpl::template applyExternalInternal<targets>...
    });
}

template <typename PrecisionT, size_t packed_size>
constexpr auto ExternalInternalFunctions() -> decltype(auto) {
    constexpr size_t internal_wires = Util::log2PerfectPower(packed_size / 2);
    return ExternalInternalFunctions_Iter<PrecisionT, packed_size>(std::make_index_sequence<internal_wires>());
}
} // namespace Internal

template <SymmetricTwoQubitGateWithoutParam AVXImpl>
class TwoQubitGateHelper {
  public:
    using Precision = typename AVXImpl::Precision;
    using FuncType = void (*)(std::complex<Precision>*, size_t, const std::vector<size_t>&, bool);
    constexpr static size_t packed_size = AVXImpl::packed_size_;
  private:
    FuncType fallback_func_;
  public:

    explicit TwoQubitGateHelper(FuncType fallback_func) : fallback_func_{fallback_func} {
    }

    void operator()(std::complex<Precision>* arr, const size_t num_qubits,
            const std::vector<size_t>& wires, bool inverse) const {
        assert(wires.size() == 2);

        constexpr static size_t internal_wires = Util::log2PerfectPower(packed_size / 2);
        constexpr static auto internal_internal_functions =
            Internal::InternalInternalFunctions<AVXImpl>();

        constexpr static auto internal_external_functions =
            Internal::InternalExternalFunctions<AVXImpl>();

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1;

        if (Util::exp2(num_qubits) < packed_size / 2) {
            fallback_func_(arr, num_qubits, wires, inverse);
            return ;
        }
        
        if (rev_wire0 < internal_wires && rev_wire1 < internal_wires) {
            auto func = internal_internal_functions[rev_wire0][rev_wire1];
            (*func)(arr, num_qubits, inverse);
            return ;
        }
        
        const auto min_rev_wire = std::min(rev_wire0, rev_wire1);
        const auto max_rev_wire = std::max(rev_wire0, rev_wire1);

        if (min_rev_wire < internal_wires) {
            (*internal_external_functions[min_rev_wire])(arr, num_qubits, max_rev_wire, inverse);
            return ;
        }

        AVXImpl::applyExternalExternal(arr, num_qubits, rev_wire0, rev_wire1, inverse);
    }
};
} // namespace Pennylane::Gates::AVX
