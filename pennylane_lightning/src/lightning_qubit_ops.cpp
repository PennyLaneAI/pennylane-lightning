#include "pybind11/stl.h"
#include "pybind11/eigen.h"
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;


VectorXcd apply(
    Ref<VectorXcd> state,
    vector<string> ops,
    vector<vector<int>> wires,
    vector<vector<float>> params
    ){
    return state;
}


PYBIND11_MODULE(lightning_qubit_ops, m)
{
    m.doc() = "lightning.qubit apply() method using Eigen";
    m.def("apply", apply, "lightning.qubit apply() method");
}
