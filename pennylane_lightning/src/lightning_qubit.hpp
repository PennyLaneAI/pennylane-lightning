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

void calc_perm(vector<int> & perm, const int &qubits) {
    for (int j = 0; j < qubits; j++) {
        if (count(perm.begin(), perm.end(), j) == 0) {
        perm.push_back(j);
        }
    }
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

Gate_3q get_gate_3q(const string &gate_name, const vector<float> &params) {
    Gate_3q op;

    if (params.empty()) {
        pfunc_3q f = ThreeQubitOps.at(gate_name);
        op = (*f)();
    }
    return op;
}

// https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
vector<int> argsort(const vector<int> &v) {

  vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

template <class State>
State contract_1q_op(const State &state, string op_string, vector<int> w, vector<float> p) {
    Gate_1q op_1q = get_gate_1q(op_string, p);
    Pairs_1q pairs_1q = {Pairs(1, w[0])};
    return op_1q.contract(state, pairs_1q);
}

template <class State>
State contract_2q_op(const State & state, string op_string, vector<int> w, vector<float> p) {
    Gate_2q op_2q = get_gate_2q(op_string, p);
    Pairs_2q pairs_2q = {Pairs(2, w[0]), Pairs(3, w[1])};
    return op_2q.contract(state, pairs_2q);
}

template <class State>
State contract_3q_op(const State &state, string op_string, vector<int> w, vector<float> p) {
    Gate_3q op_3q = get_gate_3q(op_string, p);
    Pairs_3q pairs_3q = {Pairs(3, w[0]), Pairs(4, w[1]), Pairs(5, w[2])};
    return op_3q.contract(state, pairs_3q);
}


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

        if (w.size() == 1) {
            evolved_tensor = contract_1q_op<State> (evolved_tensor, op_string, w, p);
        }
        else if (w.size() == 2) {
            evolved_tensor = contract_2q_op<State> (evolved_tensor, op_string, w, p);
        }
       else if (w.size() == 3) {
            evolved_tensor = contract_3q_op<State> (evolved_tensor, op_string, w, p);
        }

        // Updates w such that it is the calculated permutation
        calc_perm(w, qubits);
        auto inv_perm = argsort(w);
        evolved_tensor = evolved_tensor.shuffle(inv_perm);
    }

    return Map<VectorXcd> (evolved_tensor.data(), state.size(), 1);
}

VectorXcd apply_ops_1q(
    Ref<VectorXcd> state,
    vector<string> ops,
    vector<vector<int>> wires,
    vector<vector<float>> params
    ) {
    State_1q state_tensor = TensorMap<State_1q>(state.data(), 2);
    State_1q evolved_tensor = state_tensor;

    for (long unsigned int i = 0; i < ops.size(); i++) {
        // Load operation string and corresponding wires and parameters
        string op_string = ops[i];
        vector<int> w = wires[i];
        vector<float> p = params[i];
        State_1q tensor_contracted;

        tensor_contracted = contract_1q_op<State_1q> (evolved_tensor, op_string, w, p);

        const int qubits = log2(tensor_contracted.size());

        // Updates w such that it is the calculated permutation
        calc_perm(w, qubits);
        auto inv_perm = argsort(w);
        evolved_tensor = tensor_contracted.shuffle(inv_perm);
    }

    auto out_state = Map<VectorXcd> (evolved_tensor.data(), state.size(), 1);
    return out_state;
}

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

        if (w.size() == 1) {
            evolved_tensor = contract_1q_op<State_2q> (evolved_tensor, op_string, w, p);
        }
        else if (w.size() == 2) {
            evolved_tensor = contract_2q_op<State_2q> (evolved_tensor, op_string, w, p);
        }

        // Updates w such that it is the calculated permutation
        calc_perm(w, qubits);
        auto inv_perm = argsort(w);
        evolved_tensor = evolved_tensor.shuffle(inv_perm);
    }

    return Map<VectorXcd> (evolved_tensor.data(), state.size(), 1);
}
