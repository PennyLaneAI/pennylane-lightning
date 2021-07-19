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
 * Contains the main `apply()` function for applying a set of operations to a
 * multiqubit statevector.
 *
 * Also includes PyBind boilerplate for interfacing with Python.
 */
#pragma once

#include <string>
#include <vector>

//#include "Gates.hpp"
#include "StateVector.hpp"
#include "typedefs.hpp"

namespace Pennylane {

/**
 * Produces the list of qubit indices that excludes a given set of indices.
 *
 * @param excludedIndices indices to exclude (must be in the range [0,
 * qubits-1])
 * @param qubits number of qubits
 * @return Set difference of [0, ..., qubits-1] and excludedIndices, in
 * ascending order
 */
std::vector<size_t>
getIndicesAfterExclusion(const std::vector<size_t> &indicesToExclude,
                         const size_t qubits);

/**
 * Produces the decimal values for all possible bit patterns determined by a set
 * of indices, taking other indices to be fixed at 0. The qubit indices are
 * taken to be big-endian, i.e. qubit 0 is the most significant bit.
 *
 * For instance, in a circuit with 5 qubits:
 * [0, 1] -> 00000, 01000, 10000, 11000 -> 0, 8, 16, 24
 *
 * The order of the indices determines the order in which bit patterns are
 * generated, e.g. [1, 0] -> 00000, 10000, 01000, 11000 -> 0, 16, 8, 24
 *
 * i.e. the qubit indices are evaluted from last-to-first.
 *
 * @param qubitIndices indices of qubits that comprise the bit pattern
 * @param qubits number of qubits
 * @return decimal value corresponding to all possible bit patterns for the
 * given indices
 */
std::vector<size_t> generateBitPatterns(const std::vector<size_t> &qubitIndices,
                                        const size_t qubits);

/**
 * Constructs the gate defined by the supplied parameters and applies it to the
 * state vector.
 *
 * @param state state vector to which to apply the operation
 * @param opLabel unique string corresponding to a gate type
 * @param opWires index of qubits on which the gate acts
 * @param opParams defines the gate parameterisation (may be zero-length for
 * some gates)
 * @param inverse boolean indicating whether to apply the gate or its inverse
 * @param qubits number of qubits
 */
template <class Precision = double>
void constructAndApplyOperation(StateVector<Precision> &state,
                                const std::string &opLabel,
                                const std::vector<size_t> &opWires,
                                const std::vector<double> &opParams,
                                bool inverse, const size_t qubits) {
    // unique_ptr<AbstractGate> gate = constructGate(opLabel, opParams);
    if (gate->numQubits != opWires.size())
        throw std::invalid_argument(
            string("The gate of type ") + opLabel + " requires " +
            std::to_string(gate->numQubits) + " wires, but " +
            std::to_string(opWires.size()) + " were supplied");

    vector<size_t> internalIndices = generateBitPatterns(opWires, qubits);

    vector<size_t> externalWires = getIndicesAfterExclusion(opWires, qubits);
    vector<size_t> externalIndices = generateBitPatterns(externalWires, qubits);

    gate->applyKernel(state, internalIndices, externalIndices, inverse);
}

/**
 * Applies the generator of the gate to the state vector.
 *
 * @param state state vector to which to apply the operation
 * @param gate unique pointer to the gate whose generator is to be applied
 * @param opWires index of qubits on which the operation acts
 * @param qubits number of qubits
 */
template <class Precision = double>
void applyGateGenerator(StateVector<Precision> &state,
                        std::unique_ptr<AbstractGate> gate,
                        const std::vector<size_t> &opWires,
                        const size_t qubits) {
    vector<size_t> internalIndices = generateBitPatterns(opWires, qubits);

    vector<size_t> externalWires = getIndicesAfterExclusion(opWires, qubits);
    vector<size_t> externalIndices = generateBitPatterns(externalWires, qubits);

    gate->applyGenerator(state, internalIndices, externalIndices);
}

/**
 * Applies specified operations onto an input state of an arbitrary number of
 * qubits.
 *
 * @param state the multiqubit statevector, modified in place
 * @param ops list of unique string names corresponding to gate types, in the
 * order they should be applied
 * @param wires list of wires on which each gate acts
 * @param params list of parameters that defines the gate parameterisation
 * @param inverse list of booleans indicating whether a given gate or its
 * inverse should be applied
 * @param qubits number of qubits
 */
template <class Precision = double>
void apply(StateVector<Precision> &state, const std::vector<std::string> &ops,
           const std::vector<std::vector<size_t>> &wires,
           const std::vector<std::vector<double>> &params,
           const std::vector<bool> &inverse, const size_t qubits) {
    if (qubits <= 0)
        throw std::invalid_argument("Must specify one or more qubits");

    size_t expectedLength = exp2(qubits);
    if (state.length != expectedLength)
        throw std::invalid_argument(
            string("Input state vector length (") +
            std::to_string(state.length) +
            ") does not match the given number of qubits " +
            std::to_string(qubits));

    size_t numOperations = ops.size();
    if (numOperations != wires.size() || numOperations != params.size())
        throw std::invalid_argument("Invalid arguments: number of operations, "
                                    "wires, and parameters must all be equal");

    for (int i = 0; i < numOperations; i++) {
        constructAndApplyOperation(state, ops[i], wires[i], params[i],
                                   inverse[i], qubits);
    }
}

} // namespace Pennylane
