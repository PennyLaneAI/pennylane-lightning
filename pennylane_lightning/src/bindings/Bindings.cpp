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

#include <set>
#include <tuple>
#include <vector>

#include "AdjointDiff.hpp"
#include "StateVector.hpp"
#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

/// @cond DEV
namespace {
using namespace Pennylane::Algorithms;
using Pennylane::StateVector;
using std::complex;
using std::set;
using std::string;
using std::vector;

namespace py = pybind11;

/**
 * @brief Create a `%StateVector` object from a 1D numpy complex data array.
 *
 * @tparam fp_t Precision data type
 * @param numpyArray Numpy data array.
 * @return StateVector<fp_t> `%StateVector` object.
 */
template <class fp_t = double>
static auto create(const py::array_t<complex<fp_t>> *numpyArray)
    -> StateVector<fp_t> {
    py::buffer_info numpyArrayInfo = numpyArray->request();

    if (numpyArrayInfo.ndim != 1) {
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    }
    if (numpyArrayInfo.itemsize != sizeof(complex<fp_t>)) {
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    }
    auto *data_ptr = static_cast<complex<fp_t> *>(numpyArrayInfo.ptr);
    return StateVector<fp_t>(
        {data_ptr, static_cast<size_t>(numpyArrayInfo.shape[0])});
}

/**
 * @brief Apply given list of operations to Numpy data array using C++
 * `%StateVector` class.
 *
 * @tparam fp_t Precision data type
 * @param stateNumpyArray Complex numpy data array representing statevector.
 * @param ops Operations to apply to the statevector using the C++ backend.
 * @param wires Wires on which to apply each operation from `ops`.
 * @param inverse Indicate whether a given operation is an inverse.
 * @param params Parameters for each given operation in `ops`.
 */
template <class fp_t = double>
void apply(py::array_t<complex<fp_t>> &stateNumpyArray,
           const vector<string> &ops, const vector<vector<size_t>> &wires,
           const vector<bool> &inverse, const vector<vector<fp_t>> &params) {
    auto state = create<fp_t>(&stateNumpyArray);
    state.applyOperations(ops, wires, inverse, params);
}

/**
 * @brief Binding class for exposing C++ methods to Python.
 *
 * @tparam fp_t Floating point precision type.
 */
template <class fp_t = double> class StateVecBinder : public StateVector<fp_t> {
  private:
    /**
     * @brief Internal utility struct to track data indices of application for
     * operations.
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
    /**
     * @brief Construct a binding class inheriting from `%StateVector`.
     *
     * @param stateNumpyArray Complex numpy statevector data array.
     */
    explicit StateVecBinder(const py::array_t<complex<fp_t>> &stateNumpyArray)
        : StateVector<fp_t>(
              static_cast<complex<fp_t> *>(stateNumpyArray.request().ptr),
              static_cast<size_t>(stateNumpyArray.request().shape[0])) {}

    /**
     * @brief Apply the given operations to the statevector data array.
     *
     * @param ops Operations to apply to the statevector.
     * @param wires Wires on which to apply each operation from `ops`.
     * @param inverse Indicate whether a given operation is an inverse.
     * @param params Parameters for each given operation in `ops`.
     */
    void apply(const vector<string> &ops, const vector<vector<size_t>> &wires,
               const vector<bool> &inverse,
               const vector<vector<fp_t>> &params) {
        this->applyOperations(ops, wires, inverse, params);
    }

    /**
     * @brief Apply the given operations to the statevector data array.
     *
     * @param ops Operations to apply to the statevector.
     * @param wires Wires on which to apply each operation from `ops`.
     * @param inverse Indicate whether a given operation is an inverse.
     */
    void apply(const vector<string> &ops, const vector<vector<size_t>> &wires,
               const vector<bool> &inverse) {
        this->applyOperations(ops, wires, inverse);
    }

    /**
     * @brief Apply PauliX gate to the given wires.
     *
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyPauliX(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyPauliX(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply PauliY gate to the given wires.
     *
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyPauliY(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyPauliY(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply PauliZ gate to the given wires.
     *
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyPauliZ(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyPauliZ(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply Hadamard gate to the given wires.
     *
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void
    applyHadamard(const std::vector<size_t> &wires, bool inverse,
                  [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyHadamard(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply S gate to the given wires.
     *
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyS(const std::vector<size_t> &wires, bool inverse,
                [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyS(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply T gate to the given wires.
     *
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyT(const std::vector<size_t> &wires, bool inverse,
                [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyT(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply CNOT (CX) gate to the given wires.
     *
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyCNOT(const std::vector<size_t> &wires, bool inverse,
                   [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyCNOT(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply SWAP gate to the given wires.
     *
     * @param wires Wires to apply operation. First and second indices for
     * target wires.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applySWAP(const std::vector<size_t> &wires, bool inverse,
                   [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applySWAP(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply CZ gate to the given wires.
     *
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyCZ(const std::vector<size_t> &wires, bool inverse,
                 [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyCZ(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply CSWAP gate to the given wires.
     *
     * @param wires Wires to apply operation. First index for control wire,
     * second and third indices for target wires.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyCSWAP(const std::vector<size_t> &wires, bool inverse,
                    [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyCSWAP(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply Toffoli (CCX) gate to the given wires.
     *
     * @param wires Wires to apply operation. First index and second indices for
     * control wires, third index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyToffoli(const std::vector<size_t> &wires, bool inverse,
                      [[maybe_unused]] const std::vector<Param_t> params = {}) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::applyToffoli(idx.internal, idx.external, inverse);
    }
    /**
     * @brief Apply Phase-shift (\f$\textrm{diag}(1, \exp(i\theta))\f$) gate to
     * the given wires.
     *
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyPhaseShift(const std::vector<size_t> &wires, bool inverse,
                         const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyPhaseShift<Param_t>(
            idx.internal, idx.external, inverse, params[0]);
    }
    /**
     * @brief Apply controlled phase-shift
     * (\f$\textrm{diag}(1,1,1,\exp(i\theta))\f$) gate to the given wires.
     *
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <class Param_t = fp_t>
    void applyControlledPhaseShift(const std::vector<size_t> &wires,
                                   bool inverse,
                                   const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyControlledPhaseShift<Param_t>(
            idx.internal, idx.external, inverse, params[0]);
    }

    /**
     * @brief Apply RX (\f$exp(-i\theta\sigma_x/2)\f$) gate to the given wires.
     *
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <class Param_t = fp_t>
    void applyRX(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyRX<Param_t>(idx.internal, idx.external,
                                                     inverse, params[0]);
    }
    /**
     * @brief Apply RY (\f$exp(-i\theta\sigma_y/2)\f$) gate to the given wires.
     *
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <class Param_t = fp_t>
    void applyRY(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyRY<Param_t>(idx.internal, idx.external,
                                                     inverse, params[0]);
    }
    /**
     * @brief Apply RZ (\f$exp(-i\theta\sigma_z/2)\f$) gate to the given wires.
     *
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <class Param_t = fp_t>
    void applyRZ(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyRZ<Param_t>(idx.internal, idx.external,
                                                     inverse, params[0]);
    }
    /**
     * @brief Apply controlled RX gate to the given wires.
     *
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <class Param_t = fp_t>
    void applyCRX(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyCRX<Param_t>(
            idx.internal, idx.external, inverse, params[0]);
    }
    /**
     * @brief Apply controlled RY gate to the given wires.
     *
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <class Param_t = fp_t>
    void applyCRY(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyCRY<Param_t>(
            idx.internal, idx.external, inverse, params[0]);
    }
    /**
     * @brief Apply controlled RZ gate to the given wires.
     *
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <class Param_t = fp_t>
    void applyCRZ(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyCRZ<Param_t>(
            idx.internal, idx.external, inverse, params[0]);
    }
    /**
     * @brief Apply Rot gate to the given wires.
     *
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameters for given gate. Requires 3 values.
     */
    template <class Param_t = fp_t>
    void applyRot(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyRot<Param_t>(
            idx.internal, idx.external, inverse, params[0], params[1],
            params[2]);
    }
    /**
     * @brief Apply controlled Rot gate to the given wires.
     *
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameters for given gate. Requires 3 values.
     */
    template <class Param_t = fp_t>
    void applyCRot(const std::vector<size_t> &wires, bool inverse,
                   const std::vector<Param_t> &params) {
        const GateIndices idx(wires, this->getNumQubits());
        StateVector<fp_t>::template applyCRot<Param_t>(
            idx.internal, idx.external, inverse, params[0], params[1],
            params[2]);
    }

    /**
     * @brief Directly apply a given matrix to the specified wires. Matrix data
     * in 1D row-major format.
     *
     * @param matrix Matrix data to apply.
     * @param wires Wires to apply matrix.
     */
    void applyMatrixWires(const std::vector<std::complex<fp_t>> &matrix,
                          const vector<size_t> &wires, bool inverse = false) {
        this->applyOperation(matrix, wires, inverse);
    }

    /**
     * @brief Directly apply a given matrix to the specified wires. Data in 1/2D
     * numpy complex array format.
     *
     * @param matrix Numpy complex data representing matrix to apply.
     * @param wires Wires to apply given matrix.
     * @param inverse Indicate whether to take adjoint.
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

/**
 * @brief Templated class to build all required precisions for Python module.
 *
 * @tparam PrecisionT Precision of the statevector data.
 * @tparam Param_t Precision of the parameter data.
 * @param m Pybind11 module.
 */
template <class PrecisionT, class Param_t>
void lightning_class_bindings(py::module &m) {
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);
    std::string class_name = "StateVectorC" + bitsize;

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

    //***********************************************************************//
    //                              Observable
    //***********************************************************************//

    class_name = "ObsStructC" + bitsize;
    using np_arr_c = py::array_t<std::complex<Param_t>,
                                 py::array::c_style | py::array::forcecast>;
    using np_arr_r =
        py::array_t<Param_t, py::array::c_style | py::array::forcecast>;

    using obs_data_var = std::variant<std::monostate, np_arr_r, np_arr_c>;
    py::class_<ObsDatum<PrecisionT>>(m, class_name.c_str())
        .def(py::init([](const std::vector<std::string> &names,
                         const std::vector<obs_data_var> &params,
                         const std::vector<std::vector<size_t>> &wires) {
            std::vector<typename ObsDatum<PrecisionT>::param_var_t> conv_params(
                params.size());
            for (size_t p_idx = 0; p_idx < params.size(); p_idx++) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        if constexpr (std::is_same_v<p_t, np_arr_c>) {
                            auto buffer = param.request();
                            auto ptr = static_cast<std::complex<Param_t> *>(
                                buffer.ptr);
                            if (buffer.size) {
                                conv_params[p_idx] =
                                    std::vector<std::complex<Param_t>>{
                                        ptr, ptr + buffer.size};
                            }
                        } else if constexpr (std::is_same_v<p_t, np_arr_r>) {
                            auto buffer = param.request();

                            auto *ptr = static_cast<Param_t *>(buffer.ptr);
                            if (buffer.size) {
                                conv_params[p_idx] = std::vector<Param_t>{
                                    ptr, ptr + buffer.size};
                            }
                        } else {
                            PL_ABORT(
                                "Parameter datatype not current supported");
                        }
                    },
                    params[p_idx]);
            }
            return ObsDatum<PrecisionT>(names, conv_params, wires);
        }))
        .def("__repr__",
             [](const ObsDatum<PrecisionT> &obs) {
                 using namespace Pennylane::Util;
                 std::ostringstream obs_stream;
                 std::string obs_name = obs.getObsName()[0];
                 for (size_t o = 1; o < obs.getObsName().size(); o++) {
                     if (o < obs.getObsName().size()) {
                         obs_name += " @ ";
                     }
                     obs_name += obs.getObsName()[o];
                 }
                 obs_stream << "'wires' : " << obs.getObsWires();
                 return "Observable: { 'name' : " + obs_name + ", " +
                        obs_stream.str() + " }";
             })
        .def("get_name",
             [](const ObsDatum<PrecisionT> &obs) { return obs.getObsName(); })
        .def("get_wires",
             [](const ObsDatum<PrecisionT> &obs) { return obs.getObsWires(); })
        .def("get_params", [](const ObsDatum<PrecisionT> &obs) {
            py::list params;
            for (size_t i = 0; i < obs.getObsParams().size(); i++) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        if constexpr (std::is_same_v<
                                          p_t,
                                          std::vector<std::complex<Param_t>>>) {
                            params.append(py::array_t<std::complex<Param_t>>(
                                py::cast(param)));
                        } else if constexpr (std::is_same_v<
                                                 p_t, std::vector<Param_t>>) {
                            params.append(
                                py::array_t<Param_t>(py::cast(param)));
                        } else if constexpr (std::is_same_v<p_t,
                                                            std::monostate>) {
                            params.append(py::list{});
                        } else {
                            throw("Unsupported data type");
                        }
                    },
                    obs.getObsParams()[i]);
            }
            return params;
        });

    //***********************************************************************//
    //                              Operations
    //***********************************************************************//
    class_name = "OpsStructC" + bitsize;
    py::class_<OpsData<PrecisionT>>(m, class_name.c_str())
        .def(py::init<
             const std::vector<std::string> &,
             const std::vector<std::vector<Param_t>> &,
             const std::vector<std::vector<size_t>> &,
             const std::vector<bool> &,
             const std::vector<std::vector<std::complex<PrecisionT>>> &>())
        .def("__repr__", [](const OpsData<PrecisionT> &ops) {
            using namespace Pennylane::Util;
            std::ostringstream ops_stream;
            for (size_t op = 0; op < ops.getSize(); op++) {
                ops_stream << "{'name': " << ops.getOpsName()[op];
                ops_stream << ", 'params': " << ops.getOpsParams()[op];
                ops_stream << ", 'inv': " << ops.getOpsInverses()[op];
                ops_stream << "}";
                if (op < ops.getSize() - 1) {
                    ops_stream << ",";
                }
            }
            return "Operations: [" + ops_stream.str() + "]";
        });

    class_name = "AdjointJacobianC" + bitsize;
    py::class_<AdjointJacobian<PrecisionT>>(m, class_name.c_str())
        .def(py::init<>())
        .def("create_ops_list", &AdjointJacobian<PrecisionT>::createOpsData)
        .def("create_ops_list",
             [](AdjointJacobian<PrecisionT> &adj,
                const std::vector<std::string> &ops_name,
                const std::vector<np_arr_r> &ops_params,
                const std::vector<std::vector<size_t>> &ops_wires,
                const std::vector<bool> &ops_inverses,
                const std::vector<np_arr_c> &ops_matrices) {
                 std::vector<std::vector<PrecisionT>> conv_params(
                     ops_params.size());
                 std::vector<std::vector<std::complex<PrecisionT>>>
                     conv_matrices(ops_matrices.size());
                 static_cast<void>(adj);
                 for (size_t op = 0; op < ops_name.size(); op++) {
                     const auto p_buffer = ops_params[op].request();
                     const auto m_buffer = ops_matrices[op].request();
                     if (p_buffer.size) {
                         const auto *const p_ptr =
                             static_cast<const Param_t *>(p_buffer.ptr);
                         conv_params[op] =
                             std::vector<Param_t>{p_ptr, p_ptr + p_buffer.size};
                     }
                     if (m_buffer.size) {
                         const auto m_ptr =
                             static_cast<const std::complex<Param_t> *>(
                                 m_buffer.ptr);
                         conv_matrices[op] = std::vector<std::complex<Param_t>>{
                             m_ptr, m_ptr + m_buffer.size};
                     }
                 }
                 return OpsData<PrecisionT>{ops_name, conv_params, ops_wires,
                                            ops_inverses, conv_matrices};
             })
        .def("adjoint_jacobian", &AdjointJacobian<PrecisionT>::adjointJacobian)
        .def("adjoint_jacobian",
             [](AdjointJacobian<PrecisionT> &adj,
                const StateVecBinder<PrecisionT> &sv,
                const std::vector<ObsDatum<PrecisionT>> &observables,
                const OpsData<PrecisionT> &operations,
                const set<size_t> &trainableParams, size_t num_params) {
                 std::vector<std::vector<PrecisionT>> jac(
                     observables.size(),
                     std::vector<PrecisionT>(num_params, 0));
                 adj.adjointJacobian(sv.getData(), sv.getLength(), jac,
                                     observables, operations, trainableParams);
                 return py::array_t<Param_t>(py::cast(jac));
             });
}

/**
 * @brief Add C++ classes, methods and functions to Python module.
 */
PYBIND11_MODULE(lightning_qubit_ops, // NOLINT: No control over Pybind internals
                m) {
    // Suppress doxygen autogenerated signatures

    py::options options;
    options.disable_function_signatures();

    m.doc() = "lightning.qubit apply() method";
    m.def(
        "apply",
        py::overload_cast<py::array_t<complex<double>> &,
                          const vector<string> &,
                          const vector<vector<size_t>> &, const vector<bool> &,
                          const vector<vector<double>> &>(apply<double>),
        "lightning.qubit apply() method");
    m.def(
        "apply",
        py::overload_cast<py::array_t<complex<float>> &, const vector<string> &,
                          const vector<vector<size_t>> &, const vector<bool> &,
                          const vector<vector<float>> &>(apply<float>),
        "lightning.qubit apply() method");

    m.def("generateBitPatterns",
          py::overload_cast<const vector<size_t> &, size_t>(
              &StateVector<double>::generateBitPatterns),
          "Get statevector indices for gate application");
    m.def("getIndicesAfterExclusion",
          py::overload_cast<const vector<size_t> &, size_t>(
              &StateVector<double>::getIndicesAfterExclusion),
          "Get statevector indices for gate application");

    lightning_class_bindings<float, float>(m);
    lightning_class_bindings<double, double>(m);
}

} // namespace
  /// @endcond
