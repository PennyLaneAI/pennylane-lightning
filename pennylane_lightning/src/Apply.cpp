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
#include "Gates.hpp"
#include "StateVector.hpp"
#include "Util.hpp"

using std::string;
using std::unique_ptr;
using std::vector;

void Pennylane::constructAndApplyOperation(
    StateVector& state,
    const string& opLabel,
    const vector<unsigned int>& opWires,
    const vector<double>& opParams,
    const unsigned int qubits
) {
    unique_ptr<AbstractGate> gate = constructGate(opLabel, opParams);
    if (gate->numQubits != opWires.size())
        throw std::invalid_argument(string("The gate of type ") + opLabel + " requires " + std::to_string(gate->numQubits) + " wires, but " + std::to_string(opWires.size()) + " were supplied");

    const vector<CplxType>& matrix = gate->asMatrix();
    
    vector<size_t> internalIndices = generateBitPatterns(opWires, qubits);

    vector<unsigned int> externalWires = getIndicesAfterExclusion(opWires, qubits);
    vector<size_t> externalIndices = generateBitPatterns(externalWires, qubits);

    vector<CplxType> inputVector(internalIndices.size());
    for (const size_t& externalIndex : externalIndices) {
        CplxType* shiftedStatePtr = state.arr + externalIndex;

        // Gather
        size_t pos = 0;
        for (const size_t& internalIndex : internalIndices) {
            inputVector[pos] = shiftedStatePtr[internalIndex];
            pos++;
        }

        // Apply + scatter
        for (size_t i = 0; i < internalIndices.size(); i++) {
            size_t internalIndex = internalIndices[i];
            shiftedStatePtr[internalIndex] = 0;
            size_t baseIndex = i * internalIndices.size();
            for (size_t j = 0; j < internalIndices.size(); j++) {
                shiftedStatePtr[internalIndex] += matrix[baseIndex + j] * inputVector[j];
            }
        }
    }
}

void Pennylane::apply(
    pybind11::array_t<CplxType>& stateNumpyArray,
    vector<string> ops,
    vector<vector<unsigned int>> wires,
    vector<vector<double>> params,
    const unsigned int qubits
) { 
    if (qubits <= 0)
        throw std::invalid_argument("Must specify one or more qubits");

    size_t expectedLength = exp2(qubits);
    StateVector state = StateVector::create(&stateNumpyArray);
    if (state.length != expectedLength)
        throw std::invalid_argument(string("Input state vector length (") + std::to_string(state.length) + ") does not match the given number of qubits " + std::to_string(qubits));

    size_t numOperations = ops.size();
    if (numOperations != wires.size() || numOperations != params.size())
        throw std::invalid_argument("Invalid arguments: number of operations, wires, and parameters must all be equal");

    for (int i = 0; i < numOperations; i++) {
        constructAndApplyOperation(state, ops[i], wires[i], params[i], qubits);
    }
}


PYBIND11_MODULE(lightning_qubit_ops, m)
{
    m.doc() = "lightning.qubit apply() method";
    m.def("apply", Pennylane::apply, "lightning.qubit apply() method");
}
