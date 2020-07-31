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
using Eigen::MatrixXcd;
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
vector<int> calculate_tensor_indices(const vector<int> &wires, const vector<int> &tensor_indices) {
    vector<int> new_tensor_indices = wires;
    int n_indices = tensor_indices.size();
    for (int j = 0; j < n_indices; j++) {
        if (count(wires.begin(), wires.end(), tensor_indices[j]) == 0) {
            new_tensor_indices.push_back(tensor_indices[j]);
        }
    }
    return new_tensor_indices;
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

    if (params.empty()) {
        pfunc_1q f = OneQubitOps.at(gate_name);
        op = (*f)();
    }
    else if (params.size() == 1) {
        pfunc_1q_one_param f = OneQubitOpsOneParam.at(gate_name);
        op = (*f)(params[0]);
    }
    else if (params.size() == 3) {
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
    else if (params.size() == 1) {
        pfunc_2q_one_param f = TwoQubitOpsOneParam.at(gate_name);
        op = (*f)(params[0]);
    }
    else if (params.size() == 3) {
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
vector<int> calculate_qubit_positions(const vector<int> &tensor_indices) {
    vector<int> idx(tensor_indices.size());
    std::iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [&tensor_indices](size_t i1, size_t i2) {
        return tensor_indices[i1] < tensor_indices[i2];
    });
    return idx;
}

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
    Gate_1q op_1q = get_gate_1q(op_string, p);
    Pairs_1q pairs_1q = {Pairs(1, indices[0])};
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
    Gate_2q op_2q = get_gate_2q(op_string, p);
    Pairs_2q pairs_2q = {Pairs(2, indices[0]), Pairs(3, indices[1])};
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
    Gate_3q op_3q = get_gate_3q(op_string);
    Pairs_3q pairs_3q = {Pairs(3, indices[0]), Pairs(4, indices[1]), Pairs(5, indices[2])};
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
) {
    VectorXcd evolved_state = state;

    int num_ops = ops.size();

    for (int i = 0; i < num_ops; i++) {
        // Load operation string and corresponding wires and parameters
        string op_string = ops[i];
        vector<float> p = params[i];

        Gate_1q gate = get_gate_1q(op_string, p);
        MatrixXcd gate_matrix = Map<MatrixXcd> (gate.data(), 2, 2);

        evolved_state = gate_matrix * evolved_state;
    }

    return evolved_state;
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
        State_2q tensor_contracted;

        vector<int> wires_to_contract(w.size());
        for (int j = 0; j < num_wires; j++) {
            wires_to_contract[j] = qubit_positions[w[j]];
        }
        tensor_indices = calculate_tensor_indices(w, tensor_indices);
        qubit_positions = calculate_qubit_positions(tensor_indices);

        if (w.size() == 1) {
            tensor_contracted = contract_1q_op<State_2q> (evolved_tensor, op_string, wires_to_contract, p);
        }
        else if (w.size() == 2) {
            tensor_contracted = contract_2q_op<State_2q> (evolved_tensor, op_string, wires_to_contract, p);
        }
        evolved_tensor = tensor_contracted;
    }
    State_2q shuffled_evolved_tensor = evolved_tensor.shuffle(qubit_positions);

    return Map<VectorXcd> (shuffled_evolved_tensor.data(), shuffled_evolved_tensor.size(), 1);
}
