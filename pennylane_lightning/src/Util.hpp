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
 * Contains uncategorised utility functions.
 */
#pragma once

#include <memory>
#include <assert.h>

namespace Pennylane {

    /**
     * Produces the decimal values for all possible bit patterns determined by a set of indices, taking other indices to be fixed at 0.
     * The qubit indices are taken to be big-endian, i.e. qubit 0 is the most significant bit.
     *
     * For instance, in a circuit with 5 qubits:
     * [0, 1] -> 00000, 01000, 10000, 11000 -> 0, 8, 16, 24
     *
     * The order of the indices determines the order in which bit patterns are generated, e.g.
     * [1, 0] -> 00000, 10000, 01000, 11000 -> 0, 16, 8, 24
     *
     * i.e. the qubit indices are evaluted from last-to-first.
     *
     * @param qubitIndices indices of qubits that comprise the bit pattern
     * @param qubits number of qubits
     * @return decimal value corresponding to all possible bit patterns for the given indices
     */
    std::vector<size_t> generateBitPatterns(const std::vector<unsigned int>& qubitIndices, const unsigned int qubits);

    /**
     * Produces the list of qubit indices that excludes a given set of indices.
     *
     * @param excludedIndices indices to exclude (must be in the range [0, qubits-1])
     * @param qubits number of qubits
     * @return Set difference of [0, ..., qubits-1] and excludedIndices, in ascending order
     */
    std::vector<unsigned int> getIndicesAfterExclusion(const std::vector<unsigned int>& indicesToExclude, const unsigned int qubits);

    /**
     * Calculates 2^n for some integer n > 0 using bitshifts.
     * 
     * @param n the exponent
     * @return value of 2^n
     */
    inline size_t exp2(const unsigned int& n) {
        return (size_t)1 << n;
    }

    /**
     * Calculates the decimal value for a qubit, assuming a big-endian convention.
     * 
     * @param qubitIndex the index of the qubit in the range [0, qubits)
     * @param qubits the number of qubits in the circuit
     * @return decimal value for the qubit at specified index
     */
    inline size_t maxDecimalForQubit(const unsigned int qubitIndex, const unsigned int qubits) {
        assert(qubitIndex < qubits);
        return exp2(qubits - qubitIndex - 1);
    }

}

// Helper similar to std::make_unique from c++14
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
