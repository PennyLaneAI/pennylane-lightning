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

#include "pybind11/complex.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "StateVector.hpp"
#include "typedefs.hpp"

namespace Pennylane {

    /**
     * Applies specified operations onto an input state of an arbitrary number of qubits.
     *
     * @param state the multiqubit statevector as a numpy array; modified in place
     * @param ops list of unique string names corresponding to gate types, in the order they should be applied
     * @param wires list of wires on which each gate acts
     * @param params list of parameters that defines the gate parameterisation
     */
    void apply(
        pybind11::array_t<CplxType>& stateNumpyArray,
        std::vector<std::string> ops,
        std::vector<std::vector<unsigned int>> wires,
        std::vector<std::vector<double>> params,
        const unsigned int qubits
    );

}
