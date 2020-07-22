#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"
#include "operations.hpp"

using Eigen::Tensor;
using Eigen::RowMajor;
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
using State_2q = Tensor<complex<double>, 2, RowMajor>;

// Declare tensor shape for 1, 2, and 3-qubit gates
using Gate_1q = Tensor<complex<double>, 2, RowMajor>;
using Gate_2q = Tensor<complex<double>, 4, RowMajor>;
using Gate_3q = Tensor<complex<double>, 6, RowMajor>;

// Declare pairings for tensor contraction
using Pairs = IndexPair<int>;
using Pairs_1q = array<IndexPair<int>, 1>;
using Pairs_2q = array<IndexPair<int>, 2>;
using Pairs_3q = array<IndexPair<int>, 3>;

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

//        if (w.size() == 1) {
//            pairs = {Pairs(1, w[0])};
//        }
//        if (w.size() == 2) {
//            pairs = {Pairs(2, w[0]), Pairs(3, w[1])};
//        }
//        if (w.size() == 3) {
//            pairs = {Pairs(3, w[0]), Pairs(4, w[1]), Pairs(5, w[2])};
//        }

        // Load and apply operation
        if (op_string == "Identity") {
            Gate_1q op = Identity();
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);
            auto perm = calc_perm(w, qubits);
            auto inv_perm = cal_inv_perm(perm);
            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
        if (op_string == "PauliX") {
            Gate_1q op = X();
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);
            auto perm = calc_perm(w, qubits);
            auto inv_perm = cal_inv_perm(perm);
            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
        if (op_string == "PauliY") {
            Gate_1q op = Y();
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);
            auto perm = calc_perm(w, qubits);
            auto inv_perm = cal_inv_perm(perm);
            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
        if (op_string == "PauliZ") {
            Gate_1q op = Z();
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);
            auto perm = calc_perm(w, qubits);
            auto inv_perm = cal_inv_perm(perm);
            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
        if (op_string == "Hadamard") {
            Gate_1q op = H();
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);
            auto perm = calc_perm(w, qubits);
            auto inv_perm = cal_inv_perm(perm);
            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
        if (op_string == "S") {
            Gate_1q op = S();
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);
            auto perm = calc_perm(w, qubits);
            auto inv_perm = cal_inv_perm(perm);
            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
        if (op_string == "T") {
            Gate_1q op = T();
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);
            auto perm = calc_perm(w, qubits);
            auto inv_perm = cal_inv_perm(perm);
            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
        if (op_string == "RX") {
            Gate_1q op = RX(p[0]);
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);
            auto perm = calc_perm(w, qubits);
            auto inv_perm = cal_inv_perm(perm);
            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
        if (op_string == "RY") {
            Gate_1q op = RY(p[0]);
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);
            auto perm = calc_perm(w, qubits);
            auto inv_perm = cal_inv_perm(perm);
            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
        if (op_string == "RZ") {
            Gate_1q op = RZ(p[0]);
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);
            auto perm = calc_perm(w, qubits);
            auto inv_perm = cal_inv_perm(perm);
            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
    }

    auto out_state = Map<VectorXcd> (evolved_tensor.data(), 4, 1);
    return out_state;
}
