#include "pybind11/stl.h"
#include "pybind11/eigen.h"
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"

using namespace Eigen;
using namespace std;

// Declare tensor shape for state
using State_3q = Tensor<complex<double>, 3>;

// Declare tensor shape for 1, 2, and 3-qubit gates
using Gate_1q = Tensor<complex<double>, 2>;
using Gate_2q = Tensor<complex<double>, 4>;
using Gate_3q = Tensor<complex<double>, 6>;

const double ROOT2 = 0.7071067811865475;


Gate_1q Hadamard() {
    Gate_1q hadamard(2, 2);
    hadamard(0, 0) = ROOT2;
    hadamard(0, 1) = -ROOT2;
    hadamard(1, 0) = ROOT2;
    hadamard(1, 1) = ROOT2;
    return hadamard;
}


Gate_2q CNOT() {
    Gate_2q cnot(2, 2, 2, 2);
    cnot(0, 0, 0, 0) = 1;
    cnot(0, 0, 0, 1) = 0;
    cnot(0, 0, 1, 0) = 0;
    cnot(0, 0, 1, 1) = 0;
    cnot(0, 1, 0, 0) = 0;
    cnot(0, 1, 0, 1) = 1;
    cnot(0, 1, 1, 0) = 0;
    cnot(0, 1, 1, 1) = 0;
    cnot(1, 0, 0, 0) = 0;
    cnot(1, 0, 0, 1) = 0;
    cnot(1, 0, 1, 0) = 0;
    cnot(1, 0, 1, 1) = 1;
    cnot(1, 1, 0, 0) = 0;
    cnot(1, 1, 0, 1) = 0;
    cnot(1, 1, 1, 0) = 0;
    cnot(1, 1, 1, 1) = 1;
    return cnot;
}


VectorXcd apply_3q(
    Ref<VectorXcd> state,
    vector<string> ops,
    vector<vector<int>> wires,
    vector<vector<float>> params
    ) {
    State_3q state_tensor = TensorMap<State_3q>(state.data(), 2, 2, 2);
    State_3q evolved_tensor = state_tensor;

    for (int i = 0; i < ops.size(); i++) {
        // Load operation string and corresponding wires and parameters
        string op_string = ops[i];
        vector<int> w = wires[i];
        vector<float> p = params[i];

        // Load and apply operation
        if (op_string == "Hadamard") {
            Gate_1q op = Hadamard();
            Eigen::array<IndexPair<int>, 1> pairs = {IndexPair<int>(1, w[0])};
            evolved_tensor = op.contract(state_tensor, pairs);
            // We probably need to shuffle back
        }
        if (op_string == "CNOT") {
            Gate_2q op = CNOT();
            Eigen::array<IndexPair<int>, 2> pairs = {IndexPair<int>(2, w[0]), IndexPair<int>(3,
            w[1])};
            evolved_tensor = op.contract(state_tensor, pairs);
        }
    }


    auto out_state = Map<VectorXcd> (evolved_tensor.data(), 8, 1);
    return out_state;
}


PYBIND11_MODULE(lightning_qubit_ops, m)
{
    m.doc() = "lightning.qubit apply() method using Eigen";
    m.def("apply_3q", apply_3q, "lightning.qubit apply() method");
}
