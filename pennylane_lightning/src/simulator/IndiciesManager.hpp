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
 * @file
 * Defines the class representation for quantum state vectors.
 */

#pragma once

#include <cstdlib>
#include <set>
#include <vector>

#include "Util.hpp"

/**
 * This namespace defines helper functions that calculate indices appears in implementations
 * of gate operations. However, as using bit operations within a for loop is usually better
 * performing, consider to use other options before using functions below.
 */
namespace Pennylane::IndicesHelper {

/**
 * @brief Get indices of statevector data not participating in application
 * operation.
 *
 * @param indicesToExclude Indices to exclude from this call.
 * @param num_qubits Total number of qubits for statevector.
 * @return std::vector<size_t>
 */
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
/**
 * @brief Generate indices for applying operations.
 *
 * This method will return the statevector indices participating in the
 * application of a gate to a given set of qubits.
 *
 * @param qubitIndices Indices of the qubits to apply operations.
 * @param num_qubits Number of qubits in register.
 * @return std::vector<size_t>
 */
auto generateBitPatterns(const std::vector<size_t> &qubitIndices,
                                size_t num_qubits) -> std::vector<size_t> {
    std::vector<size_t> indices;
    indices.reserve(Util::exp2(qubitIndices.size()));
    indices.emplace_back(0);

    for (auto index_it = qubitIndices.rbegin();
         index_it != qubitIndices.rend(); index_it++) {
        const size_t value =
            Util::maxDecimalForQubit(*index_it, num_qubits);
        const size_t currentSize = indices.size();
        for (size_t j = 0; j < currentSize; j++) {
            indices.emplace_back(indices[j] + value);
        }
    }
    return indices;
}
} // namespace Pennylane::IndicesHelper
