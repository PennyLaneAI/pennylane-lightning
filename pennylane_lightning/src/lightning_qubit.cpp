// Copyright 2020 Xanadu Quantum Technologies Inc.

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
#include "pybind11/stl.h"
#include "pybind11/eigen.h"
#include "lightning_qubit.hpp"

/**
* Applies specified operations onto an input state of an arbitrary number of qubits.
*
* Note that only up to 50 qubits are currently supported. This limitation is due to the Eigen
* Tensor library not supporting dynamically ranked tensors.
*
* @param state the multiqubit statevector
* @param ops a vector of operation names in the order they should be applied
* @param wires a vector of wires corresponding to the operations specified in ops
* @param params a vector of parameters corresponding to the operations specified in ops
* @return the transformed statevector
*/
VectorXcd apply (
    Ref<VectorXcd> state,
    vector<string> ops,
    vector<vector<int>> wires,
    vector<vector<float>> params,
    const int qubits
) {
    if (qubits <= 0)
        throw std::invalid_argument("Must specify one or more qubits");

    switch (qubits) {
    case 1: return QubitOperations<1>::apply(state, ops, wires, params);
    case 2: return QubitOperations<2>::apply(state, ops, wires, params);
    case 3: return QubitOperations<3>::apply(state, ops, wires, params);
    case 4: return QubitOperations<4>::apply(state, ops, wires, params);
    case 5: return QubitOperations<5>::apply(state, ops, wires, params);
    case 6: return QubitOperations<6>::apply(state, ops, wires, params);
    case 7: return QubitOperations<7>::apply(state, ops, wires, params);
    case 8: return QubitOperations<8>::apply(state, ops, wires, params);
    case 9: return QubitOperations<9>::apply(state, ops, wires, params);
    case 10: return QubitOperations<10>::apply(state, ops, wires, params);
    case 11: return QubitOperations<11>::apply(state, ops, wires, params);
    case 12: return QubitOperations<12>::apply(state, ops, wires, params);
    case 13: return QubitOperations<13>::apply(state, ops, wires, params);
    case 14: return QubitOperations<14>::apply(state, ops, wires, params);
    case 15: return QubitOperations<15>::apply(state, ops, wires, params);
    case 16: return QubitOperations<16>::apply(state, ops, wires, params);
    case 17: return QubitOperations<17>::apply(state, ops, wires, params);
    case 18: return QubitOperations<18>::apply(state, ops, wires, params);
    case 19: return QubitOperations<19>::apply(state, ops, wires, params);
    case 20: return QubitOperations<20>::apply(state, ops, wires, params);
    case 21: return QubitOperations<21>::apply(state, ops, wires, params);
    case 22: return QubitOperations<22>::apply(state, ops, wires, params);
    case 23: return QubitOperations<23>::apply(state, ops, wires, params);
    case 24: return QubitOperations<24>::apply(state, ops, wires, params);
    case 25: return QubitOperations<25>::apply(state, ops, wires, params);
    case 26: return QubitOperations<26>::apply(state, ops, wires, params);
    case 27: return QubitOperations<27>::apply(state, ops, wires, params);
    case 28: return QubitOperations<28>::apply(state, ops, wires, params);
    case 29: return QubitOperations<29>::apply(state, ops, wires, params);
    case 30: return QubitOperations<30>::apply(state, ops, wires, params);
    case 31: return QubitOperations<31>::apply(state, ops, wires, params);
    case 32: return QubitOperations<32>::apply(state, ops, wires, params);
    case 33: return QubitOperations<33>::apply(state, ops, wires, params);
    case 34: return QubitOperations<34>::apply(state, ops, wires, params);
    case 35: return QubitOperations<35>::apply(state, ops, wires, params);
    case 36: return QubitOperations<36>::apply(state, ops, wires, params);
    case 37: return QubitOperations<37>::apply(state, ops, wires, params);
    case 38: return QubitOperations<38>::apply(state, ops, wires, params);
    case 39: return QubitOperations<39>::apply(state, ops, wires, params);
    case 40: return QubitOperations<40>::apply(state, ops, wires, params);
    case 41: return QubitOperations<41>::apply(state, ops, wires, params);
    case 42: return QubitOperations<42>::apply(state, ops, wires, params);
    case 43: return QubitOperations<43>::apply(state, ops, wires, params);
    case 44: return QubitOperations<44>::apply(state, ops, wires, params);
    case 45: return QubitOperations<45>::apply(state, ops, wires, params);
    case 46: return QubitOperations<46>::apply(state, ops, wires, params);
    case 47: return QubitOperations<47>::apply(state, ops, wires, params);
    case 48: return QubitOperations<48>::apply(state, ops, wires, params);
    case 49: return QubitOperations<49>::apply(state, ops, wires, params);
    case 50: return QubitOperations<50>::apply(state, ops, wires, params);
    default: throw std::invalid_argument("No support for > 50 qubits");
    }
}


PYBIND11_MODULE(lightning_qubit_ops, m)
{
    m.doc() = "lightning.qubit apply() method using Eigen";
    m.def("apply", apply, "lightning.qubit apply() method");
}
