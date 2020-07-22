#include "pybind11/stl.h"
#include "pybind11/eigen.h"
#include "lightning_qubit.hpp"


PYBIND11_MODULE(lightning_qubit_ops, m)
{
    m.doc() = "lightning.qubit apply() method using Eigen";
    m.def("apply_2q", apply_2q, "lightning.qubit 2-qubit apply() method");
}
