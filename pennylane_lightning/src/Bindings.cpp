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

#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "StateVector.hpp"

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

/**
 * @brief Binding class for exposing C++ methods to Python
 *
 * @tparam fp_t Floating point precision type.
 */
template <class fp_t> class StateVecBinder : public StateVector<fp_t> {
  private:
    /**
     * @brief Internal utility struct to track indices of application.
     *
     */
    struct GateIndices {
        const std::vector<size_t> internal;
        const std::vector<size_t> external;
        GateIndices(const std::vector<size_t> &wires, size_t num_qubits)
            : internal{std::move(
                  StateVector<fp_t>::generateBitPatterns(wires, num_qubits))},
              external{std::move(StateVector<fp_t>::generateBitPatterns(
                  StateVector<fp_t>::getIndicesAfterExclusion(wires,
                                                              num_qubits),
                  num_qubits))} {}
    };

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

    template <class Param_t = fp_t>
    void applyPauliX(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyPauliX(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void applyPauliY(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyPauliY(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void applyPauliZ(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyPauliZ(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void
    applyHadamard(const std::vector<size_t> &wires, bool inverse,
                  [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyHadamard(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void applyS(const std::vector<size_t> &wires, bool inverse,
                [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyS(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void applyT(const std::vector<size_t> &wires, bool inverse,
                [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyT(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void applyCNOT(const std::vector<size_t> &wires, bool inverse,
                   [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyCNOT(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void applySWAP(const std::vector<size_t> &wires, bool inverse,
                   [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applySWAP(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void applyCZ(const std::vector<size_t> &wires, bool inverse,
                 [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyCZ(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void applyCSWAP(const std::vector<size_t> &wires, bool inverse,
                    [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyCSWAP(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void applyToffoli(const std::vector<size_t> &wires, bool inverse,
                      [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyToffoli(idx.internal, idx.external, inverse);
    }
    template <class Param_t = fp_t>
    void applyPhaseShift(const std::vector<size_t> &wires, bool inverse,
                         const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyPhaseShift<Param_t>(
            idx.internal, idx.external, inverse, params[0]);
    }
    template <class Param_t = fp_t>
    void applyControlledPhaseShift(const std::vector<size_t> &wires,
                                   bool inverse,
                                   const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyControlledPhaseShift<Param_t>(
            idx.internal, idx.external, inverse, params[0]);
    }
    template <class Param_t = fp_t>
    void applyRX(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyRX<Param_t>(idx.internal, idx.external,
                                                     inverse, params[0]);
    }
    template <class Param_t = fp_t>
    void applyRY(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyRY<Param_t>(idx.internal, idx.external,
                                                     inverse, params[0]);
    }
    template <class Param_t = fp_t>
    void applyRZ(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyRZ<Param_t>(idx.internal, idx.external,
                                                     inverse, params[0]);
    }
    template <class Param_t = fp_t>
    void applyCRX(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyCRX<Param_t>(
            idx.internal, idx.external, inverse, params[0]);
    }
    template <class Param_t = fp_t>
    void applyCRY(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyCRY<Param_t>(
            idx.internal, idx.external, inverse, params[0]);
    }
    template <class Param_t = fp_t>
    void applyCRZ(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyCRZ<Param_t>(
            idx.internal, idx.external, inverse, params[0]);
    }
    template <class Param_t = fp_t>
    void applyRot(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyRot<Param_t>(
            idx.internal, idx.external, inverse, params[0], params[1],
            params[2]);
    }

    template <class Param_t = fp_t>
    void applyCRot(const std::vector<size_t> &wires, bool inverse,
                   const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyCRot<Param_t>(
            idx.internal, idx.external, inverse, params[0], params[1],
            params[2]);
    }

    void apply(const vector<string> &ops, const vector<vector<size_t>> &wires,
               const vector<bool> &inverse) {
        this->applyOperations(ops, wires, inverse);
    }

    /**
     * @brief Directly apply a given matrix to the specified wires. Matrix data
     * in 1D row-major format.
     *
     * @param matrix
     * @param wires
     */
    void applyMatrixWires(const std::vector<std::complex<fp_t>> &matrix,
                          const vector<size_t> &wires, bool inverse = false) {
        this->applyOperation(matrix, wires, inverse);
    }

    /**
     * @brief Directly apply a given matrix to the specified wires. Data in 1/2D
     * numpy complex array format.
     *
     * @param matrix
     * @param wires
     * @param inverse
     */
    void applyMatrixWires(
        const py::array_t<complex<fp_t>,
                          py::array::c_style | py::array::forcecast> &matrix,
        const vector<size_t> &wires, bool inverse = false) {
        const vector<size_t> internalIndices = this->generateBitPatterns(wires);
        const vector<size_t> externalWires =
            this->getIndicesAfterExclusion(wires);
        const vector<size_t> externalIndices =
            this->generateBitPatterns(externalWires);
        this->applyMatrix(static_cast<complex<fp_t> *>(matrix.request().ptr),
                          internalIndices, externalIndices, inverse);
    }
};

template <class PrecisionT, class Param_t>
void lightning_class_bindings(py::module &m) {
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);
    const std::string class_name = "StateVectorC" + bitsize;
    py::class_<StateVecBinder<PrecisionT>>(m, class_name.c_str())
        .def(py::init<
             py::array_t<complex<PrecisionT>,
                         py::array::c_style | py::array::forcecast> &>())
        .def("PauliX",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applyPauliX<Param_t>),
             "Apply the PauliX gate.")

        .def("PauliY",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applyPauliY<Param_t>),
             "Apply the PauliY gate.")

        .def("PauliZ",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applyPauliZ<Param_t>),
             "Apply the PauliZ gate.")

        .def("Hadamard",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applyHadamard<Param_t>),
             "Apply the Hadamard gate.")

        .def("S",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applyS<Param_t>),
             "Apply the S gate.")

        .def("T",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applyT<Param_t>),
             "Apply the T gate.")

        .def("CNOT",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applyCNOT<Param_t>),
             "Apply the CNOT gate.")

        .def("SWAP",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applySWAP<Param_t>),
             "Apply the SWAP gate.")

        .def("CSWAP",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applyCSWAP<Param_t>),
             "Apply the CSWAP gate.")

        .def("Toffoli",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applyToffoli<Param_t>),
             "Apply the Toffoli gate.")

        .def("CZ",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t>>(
                 &StateVecBinder<PrecisionT>::template applyCZ<Param_t>),
             "Apply the CZ gate.")

        .def(
            "PhaseShift",
            py::overload_cast<const std::vector<size_t> &, bool,
                              const std::vector<Param_t> &>(
                &StateVecBinder<PrecisionT>::template applyPhaseShift<Param_t>),
            "Apply the PhaseShift gate.")

        .def("apply",
             py::overload_cast<
                 const vector<string> &, const vector<vector<size_t>> &,
                 const vector<bool> &, const vector<vector<PrecisionT>> &>(
                 &StateVecBinder<PrecisionT>::apply))

        .def("apply", py::overload_cast<const vector<string> &,
                                        const vector<vector<size_t>> &,
                                        const vector<bool> &>(
                          &StateVecBinder<PrecisionT>::apply))
        .def("applyMatrix",
             py::overload_cast<const std::vector<std::complex<PrecisionT>> &,
                               const vector<size_t> &, bool>(
                 &StateVecBinder<PrecisionT>::applyMatrixWires))
        .def("applyMatrix",
             py::overload_cast<
                 const py::array_t<complex<PrecisionT>,
                                   py::array::c_style | py::array::forcecast> &,
                 const vector<size_t> &, bool>(
                 &StateVecBinder<PrecisionT>::applyMatrixWires))

        .def("ControlledPhaseShift",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t> &>(
                 &StateVecBinder<
                     PrecisionT>::template applyControlledPhaseShift<Param_t>),
             "Apply the ControlledPhaseShift gate.")

        .def("RX",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t> &>(
                 &StateVecBinder<PrecisionT>::template applyRX<Param_t>),
             "Apply the RX gate.")

        .def("RY",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t> &>(
                 &StateVecBinder<PrecisionT>::template applyRY<Param_t>),
             "Apply the RY gate.")

        .def("RZ",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t> &>(
                 &StateVecBinder<PrecisionT>::template applyRZ<Param_t>),
             "Apply the RZ gate.")

        .def("Rot",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t> &>(
                 &StateVecBinder<PrecisionT>::template applyRot<Param_t>),
             "Apply the Rot gate.")

        .def("CRX",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t> &>(
                 &StateVecBinder<PrecisionT>::template applyCRX<Param_t>),
             "Apply the CRX gate.")

        .def("CRY",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t> &>(
                 &StateVecBinder<PrecisionT>::template applyCRY<Param_t>),
             "Apply the CRY gate.")

        .def("CRZ",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t> &>(
                 &StateVecBinder<PrecisionT>::template applyCRZ<Param_t>),
             "Apply the CRZ gate.")

        .def("CRot",
             py::overload_cast<const std::vector<size_t> &, bool,
                               const std::vector<Param_t> &>(
                 &StateVecBinder<PrecisionT>::template applyCRot<Param_t>),
             "Apply the CRot gate.");
}

PYBIND11_MODULE(lightning_qubit_ops, m) {
    m.doc() = "lightning.qubit apply() method";
    m.def("apply", apply<double>, "lightning.qubit apply() method");
    m.def("apply", apply<float>, "lightning.qubit apply() method");

    lightning_class_bindings<float, float>(m);
    lightning_class_bindings<double, double>(m);
}
