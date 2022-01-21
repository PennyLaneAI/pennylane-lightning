// Copyright 2021 Xanadu Quantum Technologies Inc.

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
 * @file SelectGateOps.hpp
 * Defines a template class for choosing a Gate operations
 */
#pragma once
#include "AvailableKernels.hpp"
#include "Constant.hpp"
#include "GateOperation.hpp"
#include "KernelType.hpp"
#include "Macros.hpp"
#include "TypeList.hpp"

#include <array>
#include <functional>
#include <variant>

namespace Pennylane {
/**
 * @brief For lookup from any array of pair whose first elements are
 * GateOperations.
 *
 * As Util::lookup can be used in constexpr context, this function is redundant
 * (by the standard). But GCC 9 still does not accept Util::lookup in constexpr
 * some cases.
 */
template <GateOperation op, class T, size_t size>
constexpr auto
static_lookup(const std::array<std::pair<GateOperation, T>, size> &arr) -> T {
    for (size_t idx = 0; idx < size; idx++) {
        if (std::get<0>(arr[idx]) == op) {
            return std::get<1>(arr[idx]);
        }
    }
    return T{};
}

template <GeneratorOperation op, class T, size_t size>
constexpr auto
static_lookup(const std::array<std::pair<GeneratorOperation, T>, size> &arr)
    -> T {
    for (size_t idx = 0; idx < size; idx++) {
        if (std::get<0>(arr[idx]) == op) {
            return std::get<1>(arr[idx]);
        }
    }
    return T{};
}

/// @cond DEV
namespace Internal {
template <typename TypeList> struct KernelIdNamePairsHelper {
    constexpr static std::tuple value = Util::prepend_to_tuple(
        std::pair{TypeList::Type::kernel_id, TypeList::Type::name},
        KernelIdNamePairsHelper<typename TypeList::Next>::value);
};

template <> struct KernelIdNamePairsHelper<void> {
    constexpr static std::tuple value = std::tuple{};
};

} // namespace Internal
/// @endcond

/**
 * @brief Array of kernel_id and name pairs
 */
constexpr static auto kernelIdNamePairs = Util::tuple_to_array(
    Internal::KernelIdNamePairsHelper<AvailableKernels>::value);

/**
 * @brief Return kernel_id for the given kernel_name
 *
 * @param kernel_name Name of kernel to search
 */
constexpr auto string_to_kernel(const std::string_view kernel_name)
    -> KernelType {
    return Util::lookup(Util::reverse_pairs(kernelIdNamePairs), kernel_name);
}

/// @cond DEV
namespace Internal {
template <class PrecisionT, KernelType kernel, class TypeList>
struct SelectGateOpsHelper {
    using Type = std::conditional_t<
        TypeList::Type::kernel_id == kernel, typename TypeList::Type,
        typename SelectGateOpsHelper<PrecisionT, kernel,
                                     typename TypeList::Next>::Type>;
};
template <class PrecisionT, KernelType kernel>
struct SelectGateOpsHelper<PrecisionT, kernel, void> {
    using Type = void;
};
} // namespace Internal
/// @endcond

/**
 * @brief This class chooses a gate implementation at the compile time.
 *
 * When one adds another gate implementation, one needs to add a key
 * in KernelType and assign it to SelectGateOps by template specialization.
 *
 * Even though it is impossible to convert this into a constexpr function,
 * one may convert GateOpsFuncPtrPairs into constexpr functions with
 * kernel as a parameter (instead of a template prameter).
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data.
 * @tparam kernel Kernel to select
 */
template <class PrecisionT, KernelType kernel>
using SelectGateOps =
    typename Internal::SelectGateOpsHelper<PrecisionT, kernel,
                                           AvailableKernels>::Type;

} // namespace Pennylane

namespace Pennylane::Internal {
/**
 * @brief Gate operation pointer type for a statevector. See all specialized
 * types.
 */
template <class SVType, class ParamT, size_t num_params> struct GateMemFuncPtr {
    static_assert(num_params < 2 || num_params == 3,
                  "The given num_params is not supported.");
};
/**
 * @brief Function pointer type for a gate operation without parameters.
 */
template <class SVType, class ParamT> struct GateMemFuncPtr<SVType, ParamT, 0> {
    using Type = void (SVType::*)(const std::vector<size_t> &, bool);
};
/**
 * @brief Function pointer type for a gate operation with a single parameter.
 */
template <class SVType, class ParamT> struct GateMemFuncPtr<SVType, ParamT, 1> {
    using Type = void (SVType::*)(const std::vector<size_t> &, bool, ParamT);
};
/**
 * @brief Function pointer type for a gate operation with three parameters.
 */
template <class SVType, class ParamT> struct GateMemFuncPtr<SVType, ParamT, 3> {
    using Type = void (SVType::*)(const std::vector<size_t> &, bool, ParamT,
                                  ParamT, ParamT);
};

template <class SVType, class ParamT, size_t num_params>
using GateMemFuncPtrT =
    typename GateMemFuncPtr<SVType, ParamT, num_params>::Type;

/**
 * @brief Gate operation pointer type. See all specialized types.
 */
template <class PrecisionT, class ParamT, size_t num_params>
struct GateFuncPtr {
    static_assert(num_params < 2 || num_params == 3,
                  "The given num_params is not supported.");
};

/**
 * @brief Pointer type for a gate operation without parameters.
 */
template <class PrecisionT, class ParamT>
struct GateFuncPtr<PrecisionT, ParamT, 0> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &, bool);
};
/**
 * @brief Pointer type for a gate operation with a single parameter
 */
template <class PrecisionT, class ParamT>
struct GateFuncPtr<PrecisionT, ParamT, 1> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &, bool, ParamT);
};
/**
 * @brief Pointer type for a gate operation with three paramters
 */
template <class PrecisionT, class ParamT>
struct GateFuncPtr<PrecisionT, ParamT, 3> {
    using Type = void (*)(std::complex<PrecisionT> *, size_t,
                          const std::vector<size_t> &, bool, ParamT, ParamT,
                          ParamT);
};

/**
 * @brief Pointer type for a generator operation
 */
template <class PrecisionT> struct GeneratorFuncPtr {
    using Type = PrecisionT (*)(std::complex<PrecisionT> *, size_t,
                                const std::vector<size_t> &, bool);
};

/**
 * @brief Convinient type alias for GateFuncPtr. See GateFuncPtr for details.
 */
template <class PrecisionT, class ParamT, size_t num_params>
using GateFuncPtrT = typename GateFuncPtr<PrecisionT, ParamT, num_params>::Type;

template <class PrecisionT>
using GeneratorFuncPtrT = typename GeneratorFuncPtr<PrecisionT>::Type;

/**
 * @defgroup Call gate operation with provided arguments
 *
 * @tparam PrecisionT Floating point type for the state-vector.
 * @tparam ParamT Floating point type for the gate paramters.
 * @param func Function pointer for the gate operation.
 * @param num_qubits The number of qubits of the state-vector.
 * @param wires Wires the gate applies to.
 * @param inverse If true, we apply the inverse of the gate.
 * @param params The list of gate paramters.
 */
/// @{
/**
 * @brief Overload for a gate operation without parameters
 */
template <class PrecisionT, class ParamT>
inline void callGateOps(GateFuncPtrT<PrecisionT, ParamT, 0> func,
                        std::complex<PrecisionT> *data, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        [[maybe_unused]] const std::vector<ParamT> &params) {
    assert(params.empty());
    func(data, num_qubits, wires, inverse);
}

/**
 * @brief Overload for a gate operation for a single paramter
 */
template <class PrecisionT, class ParamT>
inline void callGateOps(GateFuncPtrT<PrecisionT, ParamT, 1> func,
                        std::complex<PrecisionT> *data, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<ParamT> &params) {
    assert(params.size() == 1);
    func(data, num_qubits, wires, inverse, params[0]);
}

/**
 * @brief Overload for a gate operation for three paramters
 */
template <class PrecisionT, class ParamT>
inline void callGateOps(GateFuncPtrT<PrecisionT, ParamT, 3> func,
                        std::complex<PrecisionT> *data, size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        const std::vector<ParamT> &params) {
    assert(params.size() == 3);
    func(data, num_qubits, wires, inverse, params[0], params[1], params[2]);
}
/// @}
/**
 * @brief Call a generator operation.
 *
 * @tparam PrecisionT Floating point type for the state-vector.
 * @return Scaling factor
 */
template <class PrecisionT>
inline PrecisionT callGeneratorOps(GeneratorFuncPtrT<PrecisionT> func,
                                   std::complex<PrecisionT> *data,
                                   size_t num_qubits,
                                   const std::vector<size_t> &wires, bool adj) {
    return func(data, num_qubits, wires, adj);
}

/// @cond DEV
template <class OperatorImplementation> struct ImplementedGates {
    constexpr static auto value = OperatorImplementation::implemented_gates;
};
template <class OperatorImplementation> struct ImplementedGenerators {
    constexpr static auto value =
        OperatorImplementation::implemented_generators;
};

template <class TypeList, class ValueType, template <class> class ValueClass>
auto ValueForKernelHelper([[maybe_unused]] KernelType kernel) {
    if constexpr (std::is_same_v<TypeList, void>) {
        return std::vector<ValueType>{};
    } else {
        if (TypeList::Type::kernel_id == kernel) {
            return std::vector<ValueType>(
                std::cbegin(ValueClass<typename TypeList::Type>::value),
                std::cend(ValueClass<typename TypeList::Type>::value));
        }
        return ValueForKernelHelper<typename TypeList::Next, ValueType,
                                    ValueClass>(kernel);
    }
}
/// @endcond

/**
 * @brief Return implemented_gates constexpr member variables for a given kernel
 *
 * This function interfaces the runtime variable kernel with the constant time
 * variable implemented_gates
 *
 * TODO: Change to constexpr function in C++20
 */
inline auto implementedGatesForKernel(KernelType kernel) {
    return ValueForKernelHelper<AvailableKernels, GateOperation,
                                ImplementedGates>(kernel);
}
/**
 * @brief Return implemented_generators constexpr member variables for a given
 * kernel
 *
 * This function interfaces the runtime variable kernel with the constant time
 * variable implemented_gates
 *
 * TODO: Change to constexpr function in C++20
 */
inline auto implementedGeneratorsForKernel(KernelType kernel) {
    return ValueForKernelHelper<AvailableKernels, GeneratorOperation,
                                ImplementedGenerators>(kernel);
}
} // namespace Pennylane::Internal

/**
 * @brief A hash function for GateOperations type
 */
template <> struct std::hash<Pennylane::GateOperation> {
    size_t operator()(Pennylane::GateOperation gate_operation) const {
        return std::hash<uint32_t>()(static_cast<uint32_t>(gate_operation));
    }
};
