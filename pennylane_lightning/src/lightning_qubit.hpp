#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"

using namespace Eigen;
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
using Pairs_1q = Eigen::array<IndexPair<int>, 1>;

const double ROOT2 = 0.7071067811865475;


Gate_1q Hadamard() {
    Gate_1q hadamard(2, 2);
    hadamard(0, 0) = ROOT2;
    hadamard(0, 1) = ROOT2;
    hadamard(1, 0) = ROOT2;
    hadamard(1, 1) = -ROOT2;
    return hadamard;
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

        // Load and apply operation
        if (op_string == "Hadamard") {
            Gate_1q op = Hadamard();
            Pairs_1q pairs = {Pairs(1, w[0])};
            State_2q tensor_contracted = op.contract(state_tensor, pairs);

            // Calculate permutation
            vector<int> perm = w;
            for (int j = 0; j < qubits; j++) {
                if (count(perm.begin(), perm.end(), j) == 0) {
                perm.push_back(j);
                }
            }

            // Calculate inverse permutation
            vector<int> inv_perm;
            for (int j = perm.size() - 1; j >= 0; j--) {
                int arg = find(perm.begin(), perm.end(), j)[0];
                inv_perm.push_back(arg);
            }

            evolved_tensor = tensor_contracted.shuffle(inv_perm);
        }
    }

    auto out_state = Map<VectorXcd> (evolved_tensor.data(), 4, 1);
    return out_state;
}
