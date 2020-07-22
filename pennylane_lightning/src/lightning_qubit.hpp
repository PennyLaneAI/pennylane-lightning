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


vector<int> calc_perm(vector<int> w, int qubits) {
    vector<int> perm = w;
    for (int j = 0; j < qubits; j++) {
        if (count(perm.begin(), perm.end(), j) == 0) {
        perm.push_back(j);
        }
    }
    return perm;
}


vector<int> cal_inv_perm(vector<int> perm) {
    vector<int> inv_perm;
    for (int j = perm.size() - 1; j >= 0; j--) {
        int arg = find(perm.begin(), perm.end(), j)[0];
        inv_perm.push_back(arg);
    }
    return inv_perm;
}


Gate_1q get_gate_1q(string gate_name, vector<float> params) {
    Gate_1q op;
    if (gate_name == "Identity") {
        op = Identity();
    }
    if (gate_name == "PauliX") {
        op = X();
    }
    if (gate_name == "PauliY") {
        op = Y();
    }
    if (gate_name == "PauliZ") {
        op = Z();
    }
    if (gate_name == "Hadamard") {
        op = H();
    }
    if (gate_name == "S") {
        op = S();
    }
    if (gate_name == "T") {
        op = T();
    }
    if (gate_name == "RX") {
        op = RX(params[0]);
    }
    if (gate_name == "RY") {
        op = RY(params[0]);
    }
    if (gate_name == "RZ") {
        op = RZ(params[0]);
    }
    return op;
}


Gate_2q get_gate_2q(string gate_name, vector<float> params) {
    Gate_2q op;
    if (gate_name == "CNOT") {
        op = CNOT();
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
        Gate_1q op_1q;
        Gate_2q op_2q;

        Pairs_1q pairs_1q = {Pairs(1, w[0])};
        Pairs_2q pairs_2q = {Pairs(2, w[0]), Pairs(3, w[1])};

        if (w.size() == 1) {
            op_1q = get_gate_1q(op_string, p);
            tensor_contracted = op_1q.contract(evolved_tensor, pairs_1q);
        }
        if (w.size() == 2) {
            op_2q = get_gate_2q(op_string, p);
            tensor_contracted = op_2q.contract(evolved_tensor, pairs_2q);
        }

        auto perm = calc_perm(w, qubits);
        auto inv_perm = cal_inv_perm(perm); // apparently we don't need this
        evolved_tensor = tensor_contracted.shuffle(perm);
    }

    auto out_state = Map<VectorXcd> (evolved_tensor.data(), 4, 1);
    return out_state;
}
