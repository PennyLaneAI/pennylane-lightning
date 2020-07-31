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

// Declare tensor shape for state
using State_1q = Tensor<complex<double>, 1>;
using State_2q = Tensor<complex<double>, 2>;
using State_3q = Tensor<complex<double>, 3>;
using State_4q = Tensor<complex<double>, 4>;
using State_5q = Tensor<complex<double>, 5>;
using State_6q = Tensor<complex<double>, 6>;
using State_7q = Tensor<complex<double>, 7>;
using State_8q = Tensor<complex<double>, 8>;
using State_9q = Tensor<complex<double>, 9>;
using State_10q = Tensor<complex<double>, 10>;
using State_11q = Tensor<complex<double>, 11>;
using State_12q = Tensor<complex<double>, 12>;
using State_13q = Tensor<complex<double>, 13>;
using State_14q = Tensor<complex<double>, 14>;
using State_15q = Tensor<complex<double>, 15>;
using State_16q = Tensor<complex<double>, 16>;


/**
* Applies specified operations onto an input state of an arbitrary number of qubits.
*
* Note that only up to 16 qubits are currently supported. This limitation is due to the Eigen
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
    VectorXcd evolved_state;
    switch (qubits) {
    case 1:
        evolved_state = apply_ops_1q (state, ops, params);
        break;
    case 2:
        evolved_state = apply_ops_2q (state, ops, wires, params);
        break;
    case 3:
        evolved_state = apply_ops <State_3q> (state, ops, wires, params, 2, 2, 2);
        break;
    case 4:
        evolved_state = apply_ops <State_4q> (state, ops, wires, params, 2, 2, 2, 2);
        break;
    case 5:
        evolved_state = apply_ops <State_5q> (state, ops, wires, params, 2, 2, 2, 2, 2);
        break;
    case 6:
        evolved_state = apply_ops <State_6q> (state, ops, wires, params, 2, 2, 2, 2, 2, 2);
        break;
    case 7:
        evolved_state = apply_ops <State_7q> (state, ops, wires, params, 2, 2, 2, 2, 2, 2, 2);
        break;
    case 8:
        evolved_state = apply_ops <State_8q> (state, ops, wires, params, 2, 2, 2, 2, 2, 2, 2,
                                              2);
        break;
    case 9:
        evolved_state = apply_ops <State_9q> (state, ops, wires, params,
                                              2, 2, 2, 2, 2, 2, 2, 2, 2);
        break;
    case 10:
        evolved_state = apply_ops <State_10q> (state, ops, wires, params,
                                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        break;
    case 11:
        evolved_state = apply_ops <State_11q> (state, ops, wires, params,
                                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        break;
    case 12:
        evolved_state = apply_ops <State_12q> (state, ops, wires, params,
                                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        break;
    case 13:
        evolved_state = apply_ops <State_13q> (state, ops, wires, params,
                                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        break;
    case 14:
        evolved_state = apply_ops <State_14q> (state, ops, wires, params,
                                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        break;
    case 15:
        evolved_state = apply_ops <State_15q> (state, ops, wires, params,
                                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        break;
    case 16:
        evolved_state = apply_ops <State_16q> (state, ops, wires, params,
                                               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        break;
    }
    return evolved_state;
}


PYBIND11_MODULE(lightning_qubit_ops, m)
{
    m.doc() = "lightning.qubit apply() method using Eigen";
    m.def("apply", apply, "lightning.qubit apply() method");
}
