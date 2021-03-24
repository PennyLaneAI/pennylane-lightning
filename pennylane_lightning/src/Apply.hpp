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
 * Contains the main `apply()` function for applying a set of operations to a multiqubit
 * statevector.
 *
 * Also includes PyBind boilerplate for interfacing with Python.
 */
#pragma once

#include <string>
#include <vector>

#include "Gates.hpp"
#include "StateVector.hpp"
#include "typedefs.hpp"

namespace Pennylane {

    /**
     * Produces the list of qubit indices that excludes a given set of indices.
     * 
     * @param excludedIndices indices to exclude (must be in the range [0, qubits-1])
     * @param qubits number of qubits
     * @return Set difference of [0, ..., qubits-1] and excludedIndices, in ascending order
     */
    std::vector<unsigned int> getIndicesAfterExclusion(const std::vector<unsigned int>& indicesToExclude, const unsigned int qubits);

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
     * Constructs the gate defined by the supplied parameters and applies it to the state vector.
     * 
     * @param state state vector to which to apply the operation
     * @param opLabel unique string corresponding to a gate type
     * @param opWires index of qubits on which the gate acts
     * @param opParams defines the gate parameterisation (may be zero-length for some gates)
     * @param inverse boolean indicating whether to apply the gate or its inverse
     * @param qubits number of qubits
     */
    void constructAndApplyOperation(
        StateVector& state,
        const std::string& opLabel,
        const std::vector<unsigned int>& opWires,
        const std::vector<double>& opParams,
        bool inverse,
        const unsigned int qubits
    );
    
    /**
     * Applies the generator of the gate to the state vector.
     * 
     * @param state state vector to which to apply the operation
     * @param gate unique pointer to the gate whose generator is to be applied
     * @param opWires index of qubits on which the operation acts
     * @param qubits number of qubits
     */
    void applyGateGenerator(
        StateVector& state,
        std::unique_ptr<AbstractGate> gate,
        const std::vector<unsigned int>& opWires,
        const unsigned int qubits
    );

    /**
     * Applies specified operations onto an input state of an arbitrary number of qubits.
     *
     * @param state the multiqubit statevector, modified in place
     * @param ops list of unique string names corresponding to gate types, in the order they should be applied
     * @param wires list of wires on which each gate acts
     * @param params list of parameters that defines the gate parameterisation
     * @param inverse list of booleans indicating whether a given gate or its inverse should be applied
     * @param qubits number of qubits
     */
    void apply(
        StateVector& state,
        const std::vector<std::string>& ops,
        const std::vector<std::vector<unsigned int>>& wires,
        const std::vector<std::vector<double>>& params,
        const std::vector<bool>& inverse,
        const unsigned int qubits
    );

}
