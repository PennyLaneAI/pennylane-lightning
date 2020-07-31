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
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include "operations.hpp"

using Eigen::Tensor;
using Eigen::IndexPair;
using Eigen::VectorXcd;
using Eigen::Ref;
using Eigen::TensorMap;
using Eigen::Map;
using std::array;
using std::vector;
using std::complex;
using std::string;
using std::find;

// Declare tensor shape for 1, 2, and 3-qubit gates
using Gate_1q = Tensor<complex<double>, 2>;
using Gate_2q = Tensor<complex<double>, 4>;

// Declare pairings for tensor contraction
using Pairs = IndexPair<int>;
using Pairs_1q = array<IndexPair<int>, 1>;
using Pairs_2q = array<IndexPair<int>, 2>;
using Pairs_3q = array<IndexPair<int>, 3>;

const double SQRT2INV = 0.7071067811865475;

/**
* Calulate the wire indices resulting from the tensor contraction of a gate onto a state tensor.
* Performs the calculation in-place on the wires vector.
*
* @param wires the wires acted upon by the gate
* @param qubits the number of qubits in the system
*/
void calculate_indices(vector<int> &wires, const int &qubits) {
    for (int j = 0; j < qubits; j++) {
        if (count(wires.begin(), wires.end(), j) == 0) {
        wires.push_back(j);
        }
    }
}

/**
* Returns the tensor representation of a one-qubit gate given its name and parameters.
*
* @param string the name of the gate
* @param params the parameters of the gate
* @return the gate as a tensor
*/
Gate_1q get_gate_1q(const string &gate_name, const vector<float> &params) {
    Gate_1q op;

    if (params.empty()){
        pfunc_1q f = OneQubitOps.at(gate_name);
        op = (*f)();
    }
    else if (params.size() == 1){
        pfunc_1q_one_param f = OneQubitOpsOneParam.at(gate_name);
        op = (*f)(params[0]);
    }
    else if (params.size() == 3){
        pfunc_1q_three_params f = OneQubitOpsThreeParams.at(gate_name);
        op = (*f)(params[0], params[1], params[2]);
    }
    return op;
}

/**
* Returns the tensor representation of a two-qubit gate given its name and parameters.
*
* @param string the name of the gate
* @param params the parameters of the gate
* @return the gate as a tensor
*/
Gate_2q get_gate_2q(const string &gate_name, const vector<float> &params) {
    Gate_2q op;

    if (params.empty()) {
        pfunc_2q f = TwoQubitOps.at(gate_name);
        op = (*f)();
    }
    else if (params.size() == 1){
        pfunc_2q_one_param f = TwoQubitOpsOneParam.at(gate_name);
        op = (*f)(params[0]);
    }
    else if (params.size() == 3){
        pfunc_2q_three_params f = TwoQubitOpsThreeParams.at(gate_name);
        op = (*f)(params[0], params[1], params[2]);
    }
    return op;
}

/**
* Returns the tensor representation of a three-qubit gate given its name and parameters.
*
* @param gate_name the name of the gate
* @return the gate as a tensor
*/
Gate_3q get_gate_3q(const string &gate_name) {
    Gate_3q op;
    pfunc_3q f = ThreeQubitOps.at(gate_name);
    op = (*f)();
    return op;
}

/**
* Calulate the index shuffling required to make the state tensor be wire-ordered.
*
* Implemented using argsort as given in https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes.
*
* @param indices the wire indices of a contracted tensor, calculated using calculate_indices()
* @return the resultant indices
*/
vector<int> shuffle_indices(const vector<int> &indices) {

  vector<int> idx(indices.size());
  std::iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(), [&indices](size_t i1, size_t i2) {return indices[i1] < indices[i2];});
  return idx;
}

/**
* Contract a one-qubit gate onto a state tensor.
*
* @param state the state tensor
* @param op_string the name of the operation
* @param w the wires corresponding to the operation
* @param p the parameters used in the operation
* @return the resultant state tensor
*/
template <class State>
State contract_1q_op(
    const State &state, const string &op_string, const vector<int> &w, const vector<float> &p)
{
    Gate_1q op_1q = get_gate_1q(op_string, p);
    Pairs_1q pairs_1q = {Pairs(1, w[0])};
    return op_1q.contract(state, pairs_1q);
}

/**
* Contract a two-qubit gate onto a state tensor.
*
* @param state the state tensor
* @param op_string the name of the operation
* @param w the wires corresponding to the operation
* @param p the parameters used in the operation
* @return the resultant state tensor
*/
template <class State>
State contract_2q_op(
    const State &state, const string &op_string, const vector<int> &w, const vector<float> &p)
{
    Gate_2q op_2q = get_gate_2q(op_string, p);
    Pairs_2q pairs_2q = {Pairs(2, w[0]), Pairs(3, w[1])};
    return op_2q.contract(state, pairs_2q);
}

/**
* Contract a three-qubit gate onto a state tensor.
*
* @param state the state tensor
* @param op_string the name of the operation
* @param w the wires corresponding to the operation
* @param p the parameters used in the operation
* @return the resultant state tensor
*/
template <class State>
State contract_3q_op(const State &state, const string &op_string, const vector<int> &w) {
    Gate_3q op_3q = get_gate_3q(op_string);
    Pairs_3q pairs_3q = {Pairs(3, w[0]), Pairs(4, w[1]), Pairs(5, w[2])};
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

    for (long unsigned int i = 0; i < ops.size(); i++) {
        // Load operation string and corresponding wires and parameters
        string op_string = ops[i];
        vector<int> w = wires[i];
        vector<float> p = params[i];
        State tensor_contracted;

        if (w.size() == 1) {
            tensor_contracted = contract_1q_op<State> (evolved_tensor, op_string, w, p);
        }
        else if (w.size() == 2) {
            tensor_contracted = contract_2q_op<State> (evolved_tensor, op_string, w, p);
        }
       else if (w.size() == 3) {
            tensor_contracted = contract_3q_op<State> (evolved_tensor, op_string, w);
        }

        // Updates w such that it is the calculated indices
        calculate_indices(w, qubits);
        auto inv_perm = shuffle_indices(w);
        evolved_tensor = tensor_contracted.shuffle(inv_perm);
    }

    return Map<VectorXcd> (evolved_tensor.data(), state.size(), 1);
}

/**
* Applies specified operations onto an input state of one qubit.
*
* Implemented similarly to apply_ops() but is restricted to single-qubit gates.
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
    vector<vector<int>> wires,
    vector<vector<float>> params
    ) {
    State_1q evolved_tensor = TensorMap<State_1q>(state.data(), 2);
    const int qubits = log2(evolved_tensor.size());

    for (long unsigned int i = 0; i < ops.size(); i++) {
        // Load operation string and corresponding wires and parameters
        string op_string = ops[i];
        vector<int> w = wires[i];
        vector<float> p = params[i];

        auto tensor_contracted = contract_1q_op<State_1q> (evolved_tensor, op_string, w, p);

        // Updates w such that it is the calculated permutation
        calculate_indices(w, qubits);
        auto inv_perm = shuffle_indices(w);

        evolved_tensor = tensor_contracted.shuffle(inv_perm);
    }

    return Map<VectorXcd> (evolved_tensor.data(), state.size(), 1);
}

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
    ) {
    State_2q evolved_tensor = TensorMap<State_2q>(state.data(), 2, 2);
    const int qubits = log2(evolved_tensor.size());

    for (long unsigned int i = 0; i < ops.size(); i++) {
        // Load operation string and corresponding wires and parameters
        string op_string = ops[i];
        vector<int> w = wires[i];
        vector<float> p = params[i];
        State_2q tensor_contracted;

        if (w.size() == 1) {
            tensor_contracted = contract_1q_op<State_2q> (evolved_tensor, op_string, w, p);
        }
        else if (w.size() == 2) {
            tensor_contracted = contract_2q_op<State_2q> (evolved_tensor, op_string, w, p);
        }

        // Updates w such that it is the calculated permutation
        calculate_indices(w, qubits);
        auto inv_perm = shuffle_indices(w);

        evolved_tensor = tensor_contracted.shuffle(inv_perm);
    }

    return Map<VectorXcd> (evolved_tensor.data(), state.size(), 1);
}
