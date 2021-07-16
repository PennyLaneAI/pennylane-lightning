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
#include "Apply.hpp"

#include <set>

#include "Gates.hpp"
#include "StateVector.hpp"
#include "Util.hpp"

using std::set;
using std::string;
using std::unique_ptr;
using std::vector;

vector<unsigned int> Pennylane::getIndicesAfterExclusion(
    const vector<unsigned int> &indicesToExclude, const unsigned int qubits) {
    set<unsigned int> indices;
    for (unsigned int i = 0; i < qubits; i++) {
        indices.insert(indices.end(), i);
    }
    for (const unsigned int &excludedIndex : indicesToExclude) {
        indices.erase(excludedIndex);
    }
    return vector<unsigned int>(indices.begin(), indices.end());
}

vector<size_t>
Pennylane::generateBitPatterns(const vector<unsigned int> &qubitIndices,
                               const unsigned int qubits) {
    vector<size_t> indices;
    indices.reserve(exp2(qubitIndices.size()));
    indices.push_back(0);
    for (int i = qubitIndices.size() - 1; i >= 0; i--) {
        size_t value = maxDecimalForQubit(qubitIndices[i], qubits);
        size_t currentSize = indices.size();
        for (size_t j = 0; j < currentSize; j++) {
            indices.push_back(indices[j] + value);
        }
    }
    return indices;
}

// Explicitly instantiate template functions from header

template void Pennylane::constructAndApplyOperation<double>(
    StateVector<double> &state, const std::string &opLabel,
    const std::vector<unsigned int> &opWires,
    const std::vector<double> &opParams, bool inverse,
    const unsigned int qubits);

template void Pennylane::constructAndApplyOperation<float>(
    StateVector<float> &state, const std::string &opLabel,
    const std::vector<unsigned int> &opWires,
    const std::vector<double> &opParams, bool inverse,
    const unsigned int qubits);

template void Pennylane::applyGateGenerator<double>(
    StateVector<double> &state, unique_ptr<AbstractGate> gate,
    const vector<unsigned int> &opWires, const unsigned int qubits);

template void Pennylane::applyGateGenerator<float>(
    StateVector<float> &state, unique_ptr<AbstractGate> gate,
    const vector<unsigned int> &opWires, const unsigned int qubits);

template void
Pennylane::apply<double>(StateVector<double> &state, const vector<string> &ops,
                         const vector<vector<unsigned int>> &wires,
                         const vector<vector<double>> &params,
                         const vector<bool> &inverse,
                         const unsigned int qubits);

template void Pennylane::apply<float>(StateVector<float> &state,
                                      const vector<string> &ops,
                                      const vector<vector<unsigned int>> &wires,
                                      const vector<vector<double>> &params,
                                      const vector<bool> &inverse,
                                      const unsigned int qubits);