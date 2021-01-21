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
 * \rst
 * Handles implementation of the ``apply()`` function for a range of different qubit numbers.
 * \endrst
 */
#pragma once

#define _USE_MATH_DEFINES
#include "operations.hpp"

using Eigen::VectorXcd;
using Eigen::MatrixXcd;
using Eigen::Ref;
using Eigen::TensorMap;
using Eigen::Map;
using std::vector;
using std::complex;
using std::string;
using std::find;

const double SQRT2INV = 0.7071067811865475;

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
);

/**
* Calculate the qubit-labelled indices of the state tensor after a contraction by a gate.
*
* For example, consider a 4-qubit state tensor T_{0123}. If we apply a gate to qubits 1 and 2, the
* resulting tensor will be T'_{1203}. If we then apply a gate to qubits 1 and 3, the resulting
* tensor will be T''_{1320}.
*
* @param wires the wires acted upon by the gate
* @param tensor_indices the qubit-labelled indices of the state tensor before contraction
* @return the resultant indices
*/
vector<int> calculate_tensor_indices(const vector<int> &wires, const vector<int> &tensor_indices);

/**
* Returns the tensor representation of a one-qubit gate given its name and parameters.
*
* @param string the name of the gate
* @param params the parameters of the gate
* @return the gate as a tensor
*/
Gate_Xq<1> get_gate_1q(const string &gate_name, const vector<float> &params);

/**
* Returns the tensor representation of a two-qubit gate given its name and parameters.
*
* @param string the name of the gate
* @param params the parameters of the gate
* @return the gate as a tensor
*/
Gate_Xq<2> get_gate_2q(const string &gate_name, const vector<float> &params);

/**
* Returns the tensor representation of a three-qubit gate given its name and parameters.
*
* @param gate_name the name of the gate
* @return the gate as a tensor
*/
Gate_Xq<3> get_gate_3q(const string &gate_name);

/**
* Calculate the positions of qubits in the state tensor.
*
* For example, consider a 4-qubit state tensor T_{0123}. If we apply a gate to qubits 1 and 2, the
* resulting tensor will be T'_{1203} and the qubit positions will be 2013. If we then apply a gate
* to qubits 1 and 3, the resulting tensor will be T''_{1320} and the qubit positions will be 3021.
*
* Implemented using argsort as given in https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes.
*
* @param tensor_indices the wire indices of a contracted tensor, calculated using calculate_tensor_indices()
* @return the resultant indices
*/
vector<int> calculate_qubit_positions(const vector<int> &tensor_indices);

/**
* Contract a one-qubit gate onto a state tensor.
*
* @param state the state tensor
* @param op_string the name of the operation
* @param indices the indices corresponding to the operation
* @param p the parameters used in the operation
* @return the resultant state tensor
*/
template <class State>
State contract_1q_op(
    const State &state, const string &op_string, const vector<int> &indices, const vector<float> &p)
{
    Gate_Xq<1> op_1q = get_gate_1q(op_string, p);
    Pairs_Xq<1> pairs_1q = {Pairs(1, indices[0])};
    return op_1q.contract(state, pairs_1q);
}

/**
* Contract a two-qubit gate onto a state tensor.
*
* @param state the state tensor
* @param op_string the name of the operation
* @param indices the indices corresponding to the operation
* @param p the parameters used in the operation
* @return the resultant state tensor
*/
template <class State>
State contract_2q_op(
    const State &state, const string &op_string, const vector<int> &indices, const vector<float> &p)
{
    Gate_Xq<2> op_2q = get_gate_2q(op_string, p);
    Pairs_Xq<2> pairs_2q = {Pairs(2, indices[0]), Pairs(3, indices[1])};
    return op_2q.contract(state, pairs_2q);
}

/**
* Contract a three-qubit gate onto a state tensor.
*
* @param state the state tensor
* @param op_string the name of the operation
* @param indices the indices corresponding to the operation
* @param p the parameters used in the operation
* @return the resultant state tensor
*/
template <class State>
State contract_3q_op(const State &state, const string &op_string, const vector<int> &indices) {
    Gate_Xq<3> op_3q = get_gate_3q(op_string);
    Pairs_Xq<3> pairs_3q = {Pairs(3, indices[0]), Pairs(4, indices[1]), Pairs(5, indices[2])};
    return op_3q.contract(state, pairs_3q);
}

/**
* Applies specified operations onto an input state of three or more qubits.
*
* This function converts the statevector into a tensor and then loops through the input operations,
* finds the corresponding tensor, and contracts it onto the state tensor according to the gate wires
* and parameters.
*
* The resulting state tensor is then converted back to a vector. This is to ensure interoperability using pybind11.
*
* @param state the multiqubit statevector
* @param ops a vector of operation names in the order they should be applied
* @param wires a vector of wires corresponding to the operations specified in ops
* @param params a vector of parameters corresponding to the operations specified in ops
* @return the transformed statevector
*/
template <class State, typename... Shape>
VectorXcd apply_ops(
    Ref<VectorXcd> state,
    const vector<string> & ops,
    const vector<vector<int>> & wires,
    const vector<vector<float>> &params,
    Shape... shape
) {
    State evolved_tensor = TensorMap<State>(state.data(), shape...);
    const int qubits = log2(evolved_tensor.size());

    vector<int> tensor_indices(qubits);
    std::iota(std::begin(tensor_indices), std::end(tensor_indices), 0);
    vector<int> qubit_positions(qubits);
    std::iota(std::begin(qubit_positions), std::end(qubit_positions), 0);

    int num_ops = ops.size();

    for (int i = 0; i < num_ops; i++) {
        // Load operation string and corresponding wires and parameters
        string op_string = ops[i];
        vector<int> w = wires[i];
        int num_wires = w.size();
        vector<float> p = params[i];
        State tensor_contracted;

        vector<int> wires_to_contract(w.size());
        for (int j = 0; j < num_wires; j++) {
            wires_to_contract[j] = qubit_positions[w[j]];
        }
        tensor_indices = calculate_tensor_indices(w, tensor_indices);
        qubit_positions = calculate_qubit_positions(tensor_indices);

        if (w.size() == 1) {
            tensor_contracted = contract_1q_op<State> (evolved_tensor, op_string, wires_to_contract, p);
        }
        else if (w.size() == 2) {
            tensor_contracted = contract_2q_op<State> (evolved_tensor, op_string, wires_to_contract, p);
        }
        else if (w.size() == 3) {
            tensor_contracted = contract_3q_op<State> (evolved_tensor, op_string, wires_to_contract);
        }
        evolved_tensor = tensor_contracted;
    }
    State shuffled_evolved_tensor = evolved_tensor.shuffle(qubit_positions);

    return Map<VectorXcd> (shuffled_evolved_tensor.data(), shuffled_evolved_tensor.size(), 1);
}

/**
* Applies specified operations onto an input state of one qubit.
*
* Uses matrix-vector multiplication.
*
* @param state the multiqubit statevector
* @param ops a vector of operation names in the order they should be applied
* @param wires a vector of wires corresponding to the operations specified in ops
* @param params a vector of parameters corresponding to the operations specified in ops
* @return the transformed statevector
*/
VectorXcd apply_ops_1q(
    Ref<VectorXcd> state,
    vector<string> ops,
    vector<vector<float>> params
);

/**
* Applies specified operations onto an input state of two qubits.
*
* Implemented similarly to apply_ops() but is restricted to one-qubit and two-qubit gates.
*
* @param state the multiqubit statevector
* @param ops a vector of operation names in the order they should be applied
* @param wires a vector of wires corresponding to the operations specified in ops
* @param params a vector of parameters corresponding to the operations specified in ops
* @return the transformed statevector
*/
VectorXcd apply_ops_2q(
    Ref<VectorXcd> state,
    const vector<string>& ops,
    const vector<vector<int>>& wires,
    const vector<vector<float>>& params
);

// Template classes for a generic interface for qubit operations

/**
* Main recursive template to generate multi-qubit operations
* 
* @tparam Dim the number of qubits (i.e. tensor rank)
* @tparam ValueIdx index to be decremented recursively until 0 to generate the dimensions of the tensor
*/
template<int Dim, int ValueIdx>
class QubitOperationsGenerator
{
public:
    template<typename... Shape>
    static inline VectorXcd apply(
        Ref<VectorXcd> state,
        const vector<string>& ops,
        const vector<vector<int>>& wires,
        const vector<vector<float>>& params,
        Shape... shape)
    {
        return QubitOperationsGenerator<Dim, ValueIdx - 1>::apply(state, ops, wires, params, 2, shape...);
    }
};

/**
* Terminal specialised template for general multi-qubit operations
* 
* @tparam Dim the number of qubits (i.e. tensor rank)
*/
template<int Dim>
class QubitOperationsGenerator<Dim, 0>
{
public:
    template<typename... Shape>
    static inline VectorXcd apply(
        Ref<VectorXcd> state,
        const vector<string>& ops,
        const vector<vector<int>>& wires,
        const vector<vector<float>>& params,
        Shape... shape)
    {
        return apply_ops<State_Xq<Dim>>(state, ops, wires, params, shape...);
    }
};

/**
* Terminal specialised template for single qubit operations
* 
* @tparam ValueIdx ignored, but required to specialised the main recursive template
*/
template<int ValueIdx>
class QubitOperationsGenerator<1, ValueIdx>
{
public:
    template<typename... Shape>
    static inline VectorXcd apply(
        Ref<VectorXcd> state,
        const vector<string>& ops,
        const vector<vector<int>>& wires,
        const vector<vector<float>>& params,
        Shape... shape)
    {
        return apply_ops_1q(state, ops, params);
    }
};

/**
* Terminal specialised template for two qubit operations
* 
* @tparam ValueIdx ignored, but required to specialised the main recursive template
*/
template<int ValueIdx>
class QubitOperationsGenerator<2, ValueIdx>
{
public:
    template<typename... Shape>
    static inline VectorXcd apply(
        Ref<VectorXcd> state,
        const vector<string>& ops,
        const vector<vector<int>>& wires,
        const vector<vector<float>>& params,
        Shape... shape)
    {
        return apply_ops_2q(state, ops, wires, params);
    }
};

/**
* Generic interface that invokes the generator to generate the desired multi-qubit operation
* 
* @tparam Dim the number of qubits (i.e. tensor rank)
*/
template<int Dim>
class QubitOperations
{
public:
    template<typename... Shape>
    static inline VectorXcd apply(
        Ref<VectorXcd> state,
        const vector<string>& ops,
        const vector<vector<int>>& wires,
        const vector<vector<float>>& params)
    {
        return QubitOperationsGenerator<Dim, Dim>::apply(state, ops, wires, params);
    }
};
