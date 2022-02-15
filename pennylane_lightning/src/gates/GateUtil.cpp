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
#include "GateUtil.hpp"
#include "AvailableKernels.hpp"

#include "Util.hpp"

namespace Pennylane::Gates {

auto getIndicesAfterExclusion(const std::vector<size_t> &indicesToExclude,
                              size_t num_qubits) -> std::vector<size_t> {
    std::set<size_t> indices;
    for (size_t i = 0; i < num_qubits; i++) {
        indices.emplace(i);
    }
    for (const size_t &excludedIndex : indicesToExclude) {
        indices.erase(excludedIndex);
    }
    return {indices.begin(), indices.end()};
}

auto generateBitPatterns(const std::vector<size_t> &qubitIndices,
                         size_t num_qubits) -> std::vector<size_t> {
    std::vector<size_t> indices;
    indices.reserve(Util::exp2(qubitIndices.size()));
    indices.emplace_back(0);

    for (auto index_it = qubitIndices.rbegin(); index_it != qubitIndices.rend();
         index_it++) {
        const size_t value = Util::maxDecimalForQubit(*index_it, num_qubits);
        const size_t currentSize = indices.size();
        for (size_t j = 0; j < currentSize; j++) {
            indices.emplace_back(indices[j] + value);
        }
    }
    return indices;
}
} // namespace Pennylane::Gates

/// @cond DEV
namespace {
template <class OperatorImplementation> struct ImplementedGates {
    constexpr static auto value = OperatorImplementation::implemented_gates;
};
template <class OperatorImplementation> struct ImplementedGenerators {
    constexpr static auto value =
        OperatorImplementation::implemented_generators;
};

template <class TypeList, class ValueType, template <class> class ValueClass>
auto ValueForKernelHelper(
    [[maybe_unused]] Pennylane::Gates::KernelType kernel) {
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
} // namespace
/// @endcond

namespace Pennylane::Gates {
auto implementedGatesForKernel(KernelType kernel)
    -> std::vector<GateOperation> {
    return ValueForKernelHelper<AvailableKernels, GateOperation,
                                ImplementedGates>(kernel);
}
auto implementedGeneratorsForKernel(KernelType kernel)
    -> std::vector<GeneratorOperation> {
    return ValueForKernelHelper<AvailableKernels, GeneratorOperation,
                                ImplementedGenerators>(kernel);
}
} // namespace Pennylane::Gates
