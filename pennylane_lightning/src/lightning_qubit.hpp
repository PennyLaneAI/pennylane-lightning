#pragma once

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

// Declare tensor shape for state
using State_2q = Tensor<complex<double>, 2>;

// Declare tensor shape for 1, 2, and 3-qubit gates
using Gate_1q = Tensor<complex<double>, 2>;
using Gate_2q = Tensor<complex<double>, 4>;

// Declare pairings for tensor contraction
using Pairs = IndexPair<int>;
using Pairs_1q = array<IndexPair<int>, 1>;
using Pairs_2q = array<IndexPair<int>, 2>;

const double SQRT2INV = 0.7071067811865475;

vector<int> calc_perm(vector<int> perm, int qubits) {
    for (int j = 0; j < qubits; j++) {
        if (count(perm.begin(), perm.end(), j) == 0) {
        perm.push_back(j);
        }
    }
    return perm;
}


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


Gate_2q get_gate_2q(const string &gate_name, const vector<float> &params) {
    Gate_2q op;

    if (params.empty()) {
        pfunc_2q f = TwoQubitOps.at(gate_name);
        op = (*f)();
    }
    return op;
}


VectorXcd apply_2q(
    Ref<VectorXcd> state,
    vector<string> ops,
    vector<vector<int>> wires,
    vector<vector<float>> params
    ) {
    const int qubits = 2;
    State_2q state_tensor = TensorMap<State_2q>(state.data(), 2, 2);
    State_2q evolved_tensor = state_tensor;

    for (int i = 0; i < ops.size(); i++) {
        // Load operation string and corresponding wires and parameters
        string op_string = ops[i];
        vector<int> w = wires[i];
        vector<float> p = params[i];
        State_2q tensor_contracted;

        if (w.size() == 1) {
            Gate_1q op_1q = get_gate_1q(op_string, p);
            Pairs_1q pairs_1q = {Pairs(1, w[0])};
            tensor_contracted = op_1q.contract(evolved_tensor, pairs_1q);
        }
        if (w.size() == 2) {
            Gate_2q op_2q = get_gate_2q(op_string, p);
            Pairs_2q pairs_2q = {Pairs(2, w[0]), Pairs(3, w[1])};
            tensor_contracted = op_2q.contract(evolved_tensor, pairs_2q);
        }

        auto perm = calc_perm(w, qubits);
        evolved_tensor = tensor_contracted.shuffle(perm);
    }

    auto out_state = Map<VectorXcd> (evolved_tensor.data(), 4, 1);
    return out_state;
}
