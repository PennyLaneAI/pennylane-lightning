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

#include <iostream>

#include "AdjointDiff.hpp"
#include "IndicesUtil.hpp"
#include "JacobianProd.hpp"
#include "StateVectorBase.hpp"
#include "StateVectorRaw.hpp"
#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

/// @cond DEV
namespace {
using namespace Pennylane::Algorithms;
using Pennylane::StateVectorBase;
using Pennylane::StateVectorRaw;
using Pennylane::Internal::implementedGatesForKernel;
using std::complex;
using std::set;
using std::string;
using std::vector;
} // namespace
/// @endcond

namespace py = pybind11;

/**
 * @brief Binding class for exposing C++ methods to Python.
 *
 * @tparam PrecisionT Floating point precision type.
 */
template <class PrecisionT = double>
class StateVecBinder
    : public StateVectorBase<PrecisionT, StateVecBinder<PrecisionT>> {
  public:
    using scalar_type_t = PrecisionT;
    using ComplexPrecisionT = std::complex<PrecisionT>;
    using Base = StateVectorBase<PrecisionT, StateVecBinder<PrecisionT>>;

  private:
    ComplexPrecisionT *data_;
    size_t length_;

  public:
    /**
     * @brief Construct a binding class inheriting from `%StateVector`.
     *
     * @param stateNumpyArray Complex numpy statevector data array.
     */
    explicit StateVecBinder(
        const py::array_t<ComplexPrecisionT> &stateNumpyArray)
        : Base(Util::log2PerfectPower(
              static_cast<size_t>(stateNumpyArray.request().shape[0]))) {
        length_ = static_cast<size_t>(stateNumpyArray.request().shape[0]);
        data_ = static_cast<ComplexPrecisionT *>(stateNumpyArray.request().ptr);
    }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return const ComplexPrecisionT* Pointer to statevector data.
     */
    [[nodiscard]] auto getData() const -> ComplexPrecisionT * { return data_; }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return ComplexPrecisionT* Pointer to statevector data.
     */
    auto getData() -> ComplexPrecisionT * { return data_; }

    /**
     * @brief Redefine statevector data pointer.
     *
     * @param data_ptr New data pointer.
     */
    void setData(ComplexPrecisionT *data) { data_ = data; }

    /**
     * @brief Redefine the length of the statevector and number of qubits.
     *
     * @param length New number of elements in statevector.
     */
    void setLength(size_t length) {
        if (!Util::isPerfectPowerOf2(length)) {
            PL_ABORT("The length of the array for StateVector must be "
                     "a perfect power of 2. But " +
                     std::to_string(length) +
                     " is given."); // TODO: change to std::format in C++20
        }
        length_ = length;
        setNumQubits(Util::log2PerfectPower(length_));
    }
    /**
     * @brief Redefine the number of qubits in the statevector and number of
     * elements.
     *
     * @param qubits New number of qubits represented by statevector.
     */
    void setNumQubits(size_t num_qubits) {
        setNumQubits(num_qubits);
        length_ = Util::exp2(num_qubits);
    }

    /**
     * @brief Get the number of data elements in the statevector array.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getLength() const -> std::size_t { return length_; }

    /**
     * @brief Apply the given operations to the statevector data array.
     *
     * @param ops Operations to apply to the statevector.
     * @param wires Wires on which to apply each operation from `ops`.
     * @param inverse Indicate whether a given operation is an inverse.
     * @param params Parameters for each given operation in `ops`.
     */
    void apply(const std::vector<std::string> &ops,
               const std::vector<std::vector<size_t>> &wires,
               const std::vector<bool> &inverse,
               const std::vector<std::vector<PrecisionT>> &params) {
        this->applyOperations(ops, wires, inverse, params);
    }

    /**
     * @brief Apply the given operations to the statestd::vector data array.
     *
     * @param ops Operations to apply to the statestd::vector.
     * @param wires Wires on which to apply each operation from `ops`.
     * @param inverse Indicate whether a given operation is an inverse.
     */
    void apply(const std::vector<std::string> &ops,
               const std::vector<std::vector<size_t>> &wires,
               const std::vector<bool> &inverse) {
        Base::template applyOperations(ops, wires, inverse);
    }

    /**
     * @brief Apply PauliX gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyPauliX(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applyPauliX_<kernel>(wires, inverse);
    }

    /**
     * @brief Apply PauliY gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyPauliY(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applyPauliY_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply PauliZ gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyPauliZ(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applyPauliZ_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply Hadamard gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void
    applyHadamard(const std::vector<size_t> &wires, bool inverse,
                  [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applyHadamard_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply S gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyS(const std::vector<size_t> &wires, bool inverse,
                [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applyS_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply T gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyT(const std::vector<size_t> &wires, bool inverse,
                [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applyT_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply CNOT (CX) gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyCNOT(const std::vector<size_t> &wires, bool inverse,
                   [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applyCNOT_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply SWAP gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation. First and second indices for
     * target wires.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applySWAP(const std::vector<size_t> &wires, bool inverse,
                   [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applySWAP_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply CZ gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyCZ(const std::vector<size_t> &wires, bool inverse,
                 [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applyCZ_<kernel>(wires, inverse);
    }

    /**
     * @brief Apply CSWAP gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation. First index for control wire,
     * second and third indices for target wires.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyCSWAP(const std::vector<size_t> &wires, bool inverse,
                    [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applyCSWAP_<kernel>(wires, inverse);
    }

    /**
     * @brief Apply Toffoli (CCX) gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation. First index and second indices for
     * control wires, third index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyToffoli(const std::vector<size_t> &wires, bool inverse,
                      [[maybe_unused]] const std::vector<ParamT> &params = {}) {
        Base::template applyToffoli_<kernel>(wires, inverse);
    }

    /**
     * @brief Apply Phase-shift (\f$\textrm{diag}(1, \exp(i\theta))\f$) gate to
     * the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyPhaseShift(const std::vector<size_t> &wires, bool inverse,
                         const std::vector<ParamT> &params) {
        Base::template applyPhaseShift_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply controlled phase-shift
     * (\f$\textrm{diag}(1,1,1,\exp(i\theta))\f$) gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyControlledPhaseShift(const std::vector<size_t> &wires,
                                   bool inverse,
                                   const std::vector<ParamT> &params) {
        Base::template applyControlledPhaseShift_<kernel>(wires, inverse,
                                                          params[0]);
    }

    /**
     * @brief Apply RX (\f$exp(-i\theta\sigma_x/2)\f$) gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @tparam ParamT Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyRX(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<ParamT> &params) {
        Base::template applyRX_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply RY (\f$exp(-i\theta\sigma_y/2)\f$) gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @tparam ParamT Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyRY(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<ParamT> &params) {
        Base::template applyRY_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply RZ (\f$exp(-i\theta\sigma_z/2)\f$) gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @tparam ParamT Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyRZ(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<ParamT> &params) {
        Base::template applyRZ_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply controlled RX gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @tparam ParamT Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyCRX(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<ParamT> &params) {
        Base::template applyCRX_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply controlled RY gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @tparam ParamT Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyCRY(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<ParamT> &params) {
        Base::template applyCRY_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply controlled RZ gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @tparam ParamT Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyCRZ(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<ParamT> &params) {
        Base::template applyCRZ_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply Rot gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @tparam ParamT Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameters for given gate. Requires 3 values.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyRot(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<ParamT> &params) {
        Base::template applyRot_<kernel>(wires, inverse, params[0], params[1],
                                         params[2]);
    }
    /**
     * @brief Apply controlled Rot gate to the given wires.
     *
     * @tparam kernel Kernel to run the operation
     * @tparam ParamT Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameters for given gate. Requires 3 values.
     */
    template <KernelType kernel, class ParamT = PrecisionT>
    void applyCRot(const std::vector<size_t> &wires, bool inverse,
                   const std::vector<ParamT> &params) {
        Base::template applyCRot_<kernel>(wires, inverse, params[0], params[1],
                                          params[2]);
    }

    /**
     * @brief Directly apply a given matrix to the specified wires. Data in 1/2D
     * numpy complex array format.
     *
     * @tparam kernel Kernel to run the operation
     * @param matrix Numpy complex data representing matrix to apply.
     * @param wires Wires to apply given matrix.
     * @param inverse Indicate whether to take adjoint.
     */
    template <KernelType kernel>
    void applyMatrix(
        const py::array_t<ComplexPrecisionT,
                          py::array::c_style | py::array::forcecast> &matrix,
        const std::vector<size_t> &wires, bool inverse = false) {
        Base::template applyMatrix_<kernel>(
            static_cast<ComplexPrecisionT *>(matrix.request().ptr), wires,
            inverse);
    }
};

template <class PrecisionT, class ParamT, KernelType kernel>
struct AllBinderGateOpPairs {
    constexpr static std::array value = {
        std::pair{
            GateOperations::PauliX,
            &StateVecBinder<PrecisionT>::template applyPauliX<kernel, ParamT>},
        std::pair{
            GateOperations::PauliY,
            &StateVecBinder<PrecisionT>::template applyPauliY<kernel, ParamT>},
        std::pair{
            GateOperations::PauliZ,
            &StateVecBinder<PrecisionT>::template applyPauliZ<kernel, ParamT>},
        std::pair{GateOperations::Hadamard,
                  &StateVecBinder<PrecisionT>::template applyHadamard<kernel,
                                                                      ParamT>},
        std::pair{GateOperations::S,
                  &StateVecBinder<PrecisionT>::template applyS<kernel, ParamT>},
        std::pair{GateOperations::T,
                  &StateVecBinder<PrecisionT>::template applyT<kernel, ParamT>},
        std::pair{
            GateOperations::PhaseShift,
            &StateVecBinder<PrecisionT>::template applyPhaseShift<kernel,
                                                                  ParamT>},
        std::pair{
            GateOperations::RX,
            &StateVecBinder<PrecisionT>::template applyRX<kernel, ParamT>},
        std::pair{
            GateOperations::RY,
            &StateVecBinder<PrecisionT>::template applyRY<kernel, ParamT>},
        std::pair{
            GateOperations::RZ,
            &StateVecBinder<PrecisionT>::template applyRZ<kernel, ParamT>},
        std::pair{
            GateOperations::Rot,
            &StateVecBinder<PrecisionT>::template applyRot<kernel, ParamT>},
        std::pair{
            GateOperations::CNOT,
            &StateVecBinder<PrecisionT>::template applyCNOT<kernel, ParamT>},
        std::pair{
            GateOperations::CZ,
            &StateVecBinder<PrecisionT>::template applyCZ<kernel, ParamT>},
        std::pair{
            GateOperations::SWAP,
            &StateVecBinder<PrecisionT>::template applySWAP<kernel, ParamT>},
        std::pair{
            GateOperations::ControlledPhaseShift,
            &StateVecBinder<PrecisionT>::template applyControlledPhaseShift<
                kernel, ParamT>},
        std::pair{
            GateOperations::CRX,
            &StateVecBinder<PrecisionT>::template applyCRX<kernel, ParamT>},
        std::pair{
            GateOperations::CRY,
            &StateVecBinder<PrecisionT>::template applyCRY<kernel, ParamT>},
        std::pair{
            GateOperations::CRZ,
            &StateVecBinder<PrecisionT>::template applyCRZ<kernel, ParamT>},
        std::pair{
            GateOperations::CRot,
            &StateVecBinder<PrecisionT>::template applyCRot<kernel, ParamT>},
        std::pair{
            GateOperations::Toffoli,
            &StateVecBinder<PrecisionT>::template applyToffoli<kernel, ParamT>},
        std::pair{
            GateOperations::CSWAP,
            &StateVecBinder<PrecisionT>::template applyCSWAP<kernel, ParamT>},
    };
};

/**
 * @brief Create a `%StateVector` object from a 1D numpy complex data array.
 *
 * @tparam PrecisionT Precision data type
 * @param numpyArray Numpy data array.
 * @return StateVector<PrecisionT> `%StateVector` object.
 */
template <class PrecisionT = double>
static auto create(py::array_t<complex<PrecisionT>> &numpyArray)
    -> StateVectorRaw<PrecisionT> {
    py::buffer_info numpyArrayInfo = numpyArray.request();

    if (numpyArrayInfo.ndim != 1) {
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    }
    if (numpyArrayInfo.itemsize != sizeof(complex<PrecisionT>)) {
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    }
    auto *data_ptr = static_cast<complex<PrecisionT> *>(numpyArrayInfo.ptr);
    return StateVectorRaw<PrecisionT>(
        {data_ptr, static_cast<size_t>(numpyArrayInfo.shape[0])});
}

/**
 * @brief Apply given list of operations to Numpy data array using C++
 * `%StateVector` class.
 *
 * @tparam PrecisionT Precision data type
 * @param stateNumpyArray Complex numpy data array representing statevector.
 * @param ops Operations to apply to the statevector using the C++ backend.
 * @param wires Wires on which to apply each operation from `ops`.
 * @param inverse Indicate whether a given operation is an inverse.
 * @param params Parameters for each given operation in `ops`.
 */
template <class PrecisionT = double>
void apply(py::array_t<complex<PrecisionT>> &stateNumpyArray,
           const vector<string> &ops, const vector<vector<size_t>> &wires,
           const vector<bool> &inverse,
           const vector<vector<PrecisionT>> &params) {
    auto state = create<PrecisionT>(stateNumpyArray);
    state.applyOperations(ops, wires, inverse, params);
}

/**
 * @brief For given kernel, register all implemented gate operations and apply
 * matrix.
 *
 * @tparam PrecisionT type for state-vector precision
 * @tparam ParamT type for parameters for the gate operation
 * @tparam kernel Kernel to register
 * @tparam PyClass pybind11 class type
 */
template <class PrecisionT, class ParamT, KernelType kernel, class PyClass>
void registerKernelGateOps(PyClass &&pyclass) {
    const auto kernel_name =
        std::string(lookup(Constant::available_kernels, kernel));
    const auto gate_op_pairs =
        AllBinderGateOpPairs<PrecisionT, ParamT, kernel>::value;

    for (auto gate_op : SelectGateOps<PrecisionT, kernel>::implemented_gates) {
        if (std::find(std::begin(Constant::gates_to_pyexport),
                      std::end(Constant::gates_to_pyexport),
                      gate_op) == std::end(Constant::gates_to_pyexport)) {
            /* There are gates we do not export to Python */
            continue;
        }

        if (gate_op == GateOperations::Matrix) { // applyMatrix
            const std::string name = "applyMatrix_" + kernel_name;
            pyclass.def(
                name.c_str(),
                py::overload_cast<const py::array_t<complex<PrecisionT>,
                                                    py::array::c_style |
                                                        py::array::forcecast> &,
                                  const vector<size_t> &, bool>(
                    &StateVecBinder<PrecisionT>::template applyMatrix<kernel>),
                "Apply a given matrix to wires.");
        } else {
            const auto gate_name =
                std::string(lookup(Constant::gate_names, gate_op));
            auto func = lookup(gate_op_pairs, gate_op);
            const std::string name = gate_name + "_" + kernel_name;
            // TODO: Change to std::format in C++20
            const std::string doc = "Apply the " + gate_name + " gate using " +
                                    kernel_name + " kernel.";
            pyclass.def(name.c_str(), func, doc.c_str());
        }
    }
}

/**
 * TODO: change to constexpr st::foreach in C++20
 * */
template <class PrecisionT, class ParamT, size_t idx, class PyClass>
void registerKernelsToPyexportIter(PyClass &&pyclass) {
    if constexpr (idx < Constant::kernels_to_pyexport.size()) {
        registerKernelGateOps<PrecisionT, ParamT,
                              Constant::kernels_to_pyexport[idx]>(pyclass);
        registerKernelsToPyexportIter<PrecisionT, ParamT, idx + 1>(pyclass);
    }
}

/**
 * @brief register gates for each kernel in kernels_to_pyexport
 */
template <class PrecisionT, class ParamT, class PyClass>
void registerKernelsToPyexport(PyClass &&pyclass) {
    registerKernelsToPyexportIter<PrecisionT, ParamT, 0>(pyclass);
}

/**
 * @brief Templated class to build all required precisions for Python module.
 *
 * @tparam PrecisionT Precision of the statevector data.
 * @tparam ParamT Precision of the parameter data.
 * @param m Pybind11 module.
 */
template <class PrecisionT, class ParamT>
void lightning_class_bindings(py::module &m) {
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    //***********************************************************************//
    //                              StateVector
    //***********************************************************************//

    std::string class_name = "StateVectorC" + bitsize;
    auto pyclass =
        py::class_<StateVecBinder<PrecisionT>>(m, class_name.c_str());
    pyclass.def(
        py::init<py::array_t<complex<PrecisionT>,
                             py::array::c_style | py::array::forcecast> &>());

    registerKernelsToPyexport<PrecisionT, ParamT>(pyclass);

    //***********************************************************************//
    //                              Observable
    //***********************************************************************//

    class_name = "ObsStructC" + bitsize;
    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;
    using np_arr_r =
        py::array_t<ParamT, py::array::c_style | py::array::forcecast>;

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
                            auto ptr =
                                static_cast<std::complex<ParamT> *>(buffer.ptr);
                            if (buffer.size) {
                                conv_params[p_idx] =
                                    std::vector<std::complex<ParamT>>{
                                        ptr, ptr + buffer.size};
                            }
                        } else if constexpr (std::is_same_v<p_t, np_arr_r>) {
                            auto buffer = param.request();

                            auto *ptr = static_cast<ParamT *>(buffer.ptr);
                            if (buffer.size) {
                                conv_params[p_idx] =
                                    std::vector<ParamT>{ptr, ptr + buffer.size};
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
                                          std::vector<std::complex<ParamT>>>) {
                            params.append(py::array_t<std::complex<ParamT>>(
                                py::cast(param)));
                        } else if constexpr (std::is_same_v<
                                                 p_t, std::vector<ParamT>>) {
                            params.append(py::array_t<ParamT>(py::cast(param)));
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
             const std::vector<std::vector<ParamT>> &,
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
                             static_cast<const ParamT *>(p_buffer.ptr);
                         conv_params[op] =
                             std::vector<ParamT>{p_ptr, p_ptr + p_buffer.size};
                     }
                     if (m_buffer.size) {
                         const auto m_ptr =
                             static_cast<const std::complex<ParamT> *>(
                                 m_buffer.ptr);
                         conv_matrices[op] = std::vector<std::complex<ParamT>>{
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
                const std::vector<size_t> &trainableParams, size_t num_params) {
                 std::vector<std::vector<PrecisionT>> jac(
                     observables.size(),
                     std::vector<PrecisionT>(num_params, 0));
                 adj.adjointJacobian(sv.getData(), sv.getLength(), jac,
                                     observables, operations, trainableParams);
                 return py::array_t<ParamT>(py::cast(jac));
             });

    class_name = "VectorJacobianProductC" + bitsize;
    py::class_<VectorJacobianProduct<PrecisionT>>(m, class_name.c_str())
        .def(py::init<>())
        .def("create_ops_list",
             &VectorJacobianProduct<PrecisionT>::createOpsData)
        .def("create_ops_list",
             [](VectorJacobianProduct<PrecisionT> &v,
                const std::vector<std::string> &ops_name,
                const std::vector<np_arr_r> &ops_params,
                const std::vector<std::vector<size_t>> &ops_wires,
                const std::vector<bool> &ops_inverses,
                const std::vector<np_arr_c> &ops_matrices) {
                 std::vector<std::vector<PrecisionT>> conv_params(
                     ops_params.size());
                 std::vector<std::vector<std::complex<PrecisionT>>>
                     conv_matrices(ops_matrices.size());
                 static_cast<void>(v);
                 for (size_t op = 0; op < ops_name.size(); op++) {
                     const auto p_buffer = ops_params[op].request();
                     const auto m_buffer = ops_matrices[op].request();
                     if (p_buffer.size) {
                         const auto *const p_ptr =
                             static_cast<const ParamT *>(p_buffer.ptr);
                         conv_params[op] =
                             std::vector<ParamT>{p_ptr, p_ptr + p_buffer.size};
                     }
                     if (m_buffer.size) {
                         const auto m_ptr =
                             static_cast<const std::complex<ParamT> *>(
                                 m_buffer.ptr);
                         conv_matrices[op] = std::vector<std::complex<ParamT>>{
                             m_ptr, m_ptr + m_buffer.size};
                     }
                 }
                 return OpsData<PrecisionT>{ops_name, conv_params, ops_wires,
                                            ops_inverses, conv_matrices};
             })
        .def("compute_vjp_from_jac",
             &VectorJacobianProduct<PrecisionT>::computeVJP)
        .def("compute_vjp_from_jac",
             [](VectorJacobianProduct<PrecisionT> &v,
                const std::vector<PrecisionT> &jac,
                const std::vector<PrecisionT> &dy_row, size_t m, size_t n) {
                 std::vector<PrecisionT> vjp_res(n);
                 v._computeVJP(vjp_res, jac, dy_row, m, n);
                 return py::array_t<ParamT>(py::cast(vjp_res));
             })
        .def("vjp", &VectorJacobianProduct<PrecisionT>::vectorJacobianProduct)
        .def("vjp", [](VectorJacobianProduct<PrecisionT> &v,
                       const std::vector<PrecisionT> &dy,
                       const StateVecBinder<PrecisionT> &sv,
                       const std::vector<ObsDatum<PrecisionT>> &observables,
                       const OpsData<PrecisionT> &operations,
                       const std::vector<size_t> &trainableParams,
                       size_t num_params) {
            std::vector<std::vector<PrecisionT>> jac(
                observables.size(), std::vector<PrecisionT>(num_params, 0));
            std::vector<PrecisionT> vjp_res(num_params);
            v.vectorJacobianProduct(vjp_res, jac, dy, sv.getData(),
                                    sv.getLength(), observables, operations,
                                    trainableParams);
            return py::make_tuple(py::array_t<ParamT>(py::cast(jac)),
                                  py::array_t<ParamT>(py::cast(vjp_res)));
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
              &IndicesUtil::generateBitPatterns),
          "Get statevector indices for gate application");
    m.def("getIndicesAfterExclusion",
          py::overload_cast<const vector<size_t> &, size_t>(
              &IndicesUtil::getIndicesAfterExclusion),
          "Get statevector indices for gate application");

    /* Add EXPORTED_KERNELS */
    std::vector<std::pair<std::string, std::string>> exported_kernel_ops;

    std::set<GateOperations> gates_to_pyexport(
        std::begin(Constant::gates_to_pyexport),
        std::end(Constant::gates_to_pyexport));
    for (auto kernel : Constant::kernels_to_pyexport) {
        auto kernel_name =
            std::string(lookup(Constant::available_kernels, kernel));
        auto implemeted_gates = implementedGatesForKernel<float>(kernel);
        for (auto gate_op : implemeted_gates) {
            if (gates_to_pyexport.count(gate_op) != 0) {
                auto gate_name =
                    std::string(lookup(Constant::gate_names, gate_op));
                exported_kernel_ops.emplace_back(kernel_name, gate_name);
            }
        }
    }

    m.attr("EXPORTED_KERNEL_OPS") = py::cast(exported_kernel_ops);

    /* Add DEFAULT_KERNEL_FOR_OPS */
    std::map<std::string, std::string> default_kernel_ops_map;
    for (const auto &[gate_op, name] : Constant::gate_names) {
        auto kernel = lookup(Constant::default_kernel_for_ops, gate_op);
        auto kernel_name =
            std::string(lookup(Constant::available_kernels, kernel));
        default_kernel_ops_map.emplace(std::string(name), kernel_name);
    }
    m.attr("DEFAULT_KERNEL_FOR_OPS") = py::cast(default_kernel_ops_map);

    lightning_class_bindings<float, float>(m);
    lightning_class_bindings<double, double>(m);
}

#ifndef NDEBUG // if debug

/**
 * @brief Test whether BinderGateOpPairs are defined for all gates in pyexport
 */
template <typename PrecisionT, typename ParamT, KernelType kernel, size_t idx>
constexpr void testBinderGateOpPairsForKernelIter() {
    if constexpr (idx < Constant::gates_to_pyexport.size()) {
        constexpr auto op_pairs =
            AllBinderGateOpPairs<PrecisionT, ParamT, kernel>::value;
        constexpr auto gate_op = Constant::gates_to_pyexport[idx];
        static_assert(array_has_elt(Util::first_elts_of(op_pairs), gate_op) ||
                          gate_op == GateOperations::Matrix,
                      "AllBinderGateOpPairs should have elementes for all gate "
                      "operations to pyexport.");
        testBinderGateOpPairsForKernelIter<PrecisionT, ParamT, kernel,
                                           idx + 1>();
    }
}

template <typename PrecisionT, typename ParamT, KernelType kernel>
constexpr void testBinderGateOpPairsForKernel() {
    testBinderGateOpPairsForKernelIter<PrecisionT, ParamT, kernel, 0>();
}

/**
 * @brief
 */
template <typename PrecisionT, typename ParamT, size_t idx>
constexpr void testBinderGateOpPairsIter() {
    if constexpr (idx < Constant::kernels_to_pyexport.size()) {
        testBinderGateOpPairsForKernel<PrecisionT, ParamT,
                                       Constant::kernels_to_pyexport[idx]>();
        testBinderGateOpPairsIter<PrecisionT, ParamT, idx + 1>();
    }
}

template <typename PrecisionT, typename ParamT>
constexpr bool testBinderGateOpPairs() {
    testBinderGateOpPairsIter<PrecisionT, ParamT, 0>();
    return true;
}

static_assert(
    testBinderGateOpPairs<float, float>(),
    "AllBinderGateOpPairs should be well defined for all kernels to pyexport.");

static_assert(
    testBinderGateOpPairs<double, double>(),
    "AllBinderGateOpPairs should be well defined for all kernels to pyexport.");

#endif
