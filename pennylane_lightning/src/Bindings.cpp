// Copyright 2021 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "StateVector.hpp"
#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

using Pennylane::StateVector;
using std::complex;
using std::string;
using std::vector;

namespace py = pybind11;

template <class T = double>
static StateVector<T> create(const py::array_t<complex<T>> *numpyArray) {
    py::buffer_info numpyArrayInfo = numpyArray->request();

    if (numpyArrayInfo.ndim != 1)
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    if (numpyArrayInfo.itemsize != sizeof(complex<T>))
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    complex<T> *data_ptr = static_cast<complex<T> *>(numpyArrayInfo.ptr);
    return StateVector<T>(
        {data_ptr, static_cast<size_t>(numpyArrayInfo.shape[0])});
}

template <class T = double>
void apply(py::array_t<complex<T>> &stateNumpyArray, const vector<string> &ops,
           const vector<vector<size_t>> &wires, const vector<bool> &inverse,
           const vector<vector<T>> &params) {
    auto state = create<T>(&stateNumpyArray);
    state.applyOperations(ops, wires, inverse, params);
}

template <class fp_t> class StateVecBinder : public StateVector<fp_t> {
  public:
    explicit StateVecBinder(const py::array_t<complex<fp_t>> &stateNumpyArray)
        : StateVector<fp_t>(
              static_cast<complex<fp_t> *>(stateNumpyArray.request().ptr),
              static_cast<size_t>(stateNumpyArray.request().shape[0])) {}

    void apply(const vector<string> &ops, const vector<vector<size_t>> &wires,
               const vector<bool> &inverse,
               const vector<vector<fp_t>> &params) {
        this->applyOperations(ops, wires, inverse, params);
    }
    void apply(const vector<string> &ops, const vector<vector<size_t>> &wires,
               const vector<bool> &inverse) {
        this->applyOperations(ops, wires, inverse);
    }
    /**
     * @brief Directly apply a given matrix to the specified wires.
     *
     * @param matrix
     * @param wires
     */
    void applyMatrixWires(const std::vector<std::complex<fp_t>> &matrix,
                          const vector<size_t> &wires) {
        this->applyOperation(matrix, wires, false);
    }
};

template <class PrecisionT> void lightning_class_bindings(py::module &m) {
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);
    const std::string class_name = "StateVectorC" + bitsize;
    py::class_<StateVecBinder<PrecisionT>>(m, class_name.c_str())
        .def(py::init<
             py::array_t<complex<PrecisionT>,
                         py::array::c_style | py::array::forcecast> &>())
        .def("apply",
             py::overload_cast<
                 const vector<string> &, const vector<vector<size_t>> &,
                 const vector<bool> &, const vector<vector<PrecisionT>> &>(
                 &StateVecBinder<PrecisionT>::apply))
        .def("apply", py::overload_cast<const vector<string> &,
                                        const vector<vector<size_t>> &,
                                        const vector<bool> &>(
                          &StateVecBinder<PrecisionT>::apply))
        .def("applyMatrix", &StateVecBinder<PrecisionT>::applyMatrixWires)
        .def("PauliX", &StateVecBinder<PrecisionT>::applyPauliX)
        .def("PauliY", &StateVecBinder<PrecisionT>::applyPauliY)
        .def("PauliZ", &StateVecBinder<PrecisionT>::applyPauliZ)
        .def("Hadamard", &StateVecBinder<PrecisionT>::applyHadamard)
        .def("S", &StateVecBinder<PrecisionT>::applyS)
        .def("T", &StateVecBinder<PrecisionT>::applyT)
        .def("CNOT", &StateVecBinder<PrecisionT>::applyCNOT)
        .def("SWAP", &StateVecBinder<PrecisionT>::applySWAP)
        .def("CSWAP", &StateVecBinder<PrecisionT>::applyCSWAP)
        .def("Toffoli", &StateVecBinder<PrecisionT>::applyToffoli)
        .def("CZ", &StateVecBinder<PrecisionT>::applyCZ)
        .def("PhaseShift",
             &StateVecBinder<PrecisionT>::template applyPhaseShift<float>)
        .def("PhaseShift",
             &StateVecBinder<PrecisionT>::template applyPhaseShift<double>)
        .def("ControlledPhaseShift",
             &StateVecBinder<PrecisionT>::template applyControlledPhaseShift<
                 float>)
        .def("ControlledPhaseShift",
             &StateVecBinder<PrecisionT>::template applyControlledPhaseShift<
                 double>)
        .def("RX", &StateVecBinder<PrecisionT>::template applyRX<float>)
        .def("RX", &StateVecBinder<PrecisionT>::template applyRX<double>)
        .def("RY", &StateVecBinder<PrecisionT>::template applyRY<float>)
        .def("RY", &StateVecBinder<PrecisionT>::template applyRY<double>)
        .def("RZ", &StateVecBinder<PrecisionT>::template applyRZ<float>)
        .def("RZ", &StateVecBinder<PrecisionT>::template applyRZ<double>)
        .def("Rot", &StateVecBinder<PrecisionT>::template applyRot<float>)
        .def("Rot", &StateVecBinder<PrecisionT>::template applyRot<double>)
        .def("CRX", &StateVecBinder<PrecisionT>::template applyCRX<float>)
        .def("CRX", &StateVecBinder<PrecisionT>::template applyCRX<double>)
        .def("CRY", &StateVecBinder<PrecisionT>::template applyCRY<float>)
        .def("CRY", &StateVecBinder<PrecisionT>::template applyCRY<double>)
        .def("CRZ", &StateVecBinder<PrecisionT>::template applyCRZ<float>)
        .def("CRZ", &StateVecBinder<PrecisionT>::template applyCRZ<double>)
        .def("CRot", &StateVecBinder<PrecisionT>::template applyCRot<float>)
        .def("CRot", &StateVecBinder<PrecisionT>::template applyCRot<double>);
}

PYBIND11_MODULE(lightning_qubit_ops, m) {
    m.doc() = "lightning.qubit apply() method";
    m.def("apply", apply<double>, "lightning.qubit apply() method");
    m.def("apply", apply<float>, "lightning.qubit apply() method");

    lightning_class_bindings<float>(m);
    lightning_class_bindings<double>(m);
}
