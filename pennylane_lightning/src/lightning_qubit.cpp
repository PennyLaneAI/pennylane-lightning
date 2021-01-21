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
 * Contains the main `apply()` and auxiliary functions for applying a set of
 * operations to a multiqubit statevector.
 */
#include "lightning_qubit.hpp"

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

VectorXcd marginal_probs (
    Ref<VectorXcd> probs,
    vector<int> wires,
    const int qubits
) {
    switch (qubits) {
    case 1: return QubitOperations<1>::marginal_probs(probs, wires);
    case 2: return QubitOperations<2>::marginal_probs(probs, wires);
    case 3: return QubitOperations<3>::marginal_probs(probs, wires);
    case 4: return QubitOperations<4>::marginal_probs(probs, wires);
    case 5: return QubitOperations<5>::marginal_probs(probs, wires);
    }
}

VectorXcd all_probs (
    Ref<VectorXcd> state
) {
    return (state * state.conjugate().transpose()).diagonal();
}

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

Gate_Xq<1> get_gate_1q(const string &gate_name, const vector<float> &params) {
    Gate_Xq<1> op;

    if (params.empty()) {
        pfunc_Xq<1> f = OneQubitOps.at(gate_name);
        op = (*f)();
    }
    else if (params.size() == 1) {
        pfunc_Xq_one_param<1> f = OneQubitOpsOneParam.at(gate_name);
        op = (*f)(params[0]);
    }
    else if (params.size() == 3) {
        pfunc_Xq_three_params<1> f = OneQubitOpsThreeParams.at(gate_name);
        op = (*f)(params[0], params[1], params[2]);
    }
    return op;
}

Gate_Xq<2> get_gate_2q(const string &gate_name, const vector<float> &params) {
    Gate_Xq<2> op;

    if (params.empty()) {
        pfunc_Xq<2> f = TwoQubitOps.at(gate_name);
        op = (*f)();
    }
    else if (params.size() == 1) {
        pfunc_Xq_one_param<2> f = TwoQubitOpsOneParam.at(gate_name);
        op = (*f)(params[0]);
    }
    else if (params.size() == 3) {
        pfunc_Xq_three_params<2> f = TwoQubitOpsThreeParams.at(gate_name);
        op = (*f)(params[0], params[1], params[2]);
    }
    return op;
}

Gate_Xq<3> get_gate_3q(const string &gate_name) {
    Gate_Xq<3> op;
    pfunc_Xq<3> f = ThreeQubitOps.at(gate_name);
    op = (*f)();
    return op;
}

vector<int> calculate_qubit_positions(const vector<int> &tensor_indices) {
    vector<int> idx(tensor_indices.size());
    std::iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [&tensor_indices](size_t i1, size_t i2) {
        return tensor_indices[i1] < tensor_indices[i2];
    });
    return idx;
}

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

        Gate_Xq<1> gate = get_gate_1q(op_string, p);
        MatrixXcd gate_matrix = Map<MatrixXcd> (gate.data(), 2, 2);

        evolved_state = gate_matrix * evolved_state;
    }

    return evolved_state;
}

VectorXcd apply_ops_2q(
    Ref<VectorXcd> state,
    const vector<string>& ops,
    const vector<vector<int>>& wires,
    const vector<vector<float>>& params
) {
    State_Xq<2> evolved_tensor = TensorMap<State_Xq<2>>(state.data(), 2, 2);
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
        State_Xq<2> tensor_contracted;

        vector<int> wires_to_contract(w.size());
        for (int j = 0; j < num_wires; j++) {
            wires_to_contract[j] = qubit_positions[w[j]];
        }
        tensor_indices = calculate_tensor_indices(w, tensor_indices);
        qubit_positions = calculate_qubit_positions(tensor_indices);

        if (w.size() == 1) {
            tensor_contracted = contract_1q_op<State_Xq<2>> (evolved_tensor, op_string, wires_to_contract, p);
        }
        else if (w.size() == 2) {
            tensor_contracted = contract_2q_op<State_Xq<2>> (evolved_tensor, op_string, wires_to_contract, p);
        }
        evolved_tensor = tensor_contracted;
    }
    State_Xq<2> shuffled_evolved_tensor = evolved_tensor.shuffle(qubit_positions);

    return Map<VectorXcd> (shuffled_evolved_tensor.data(), shuffled_evolved_tensor.size(), 1);
}
