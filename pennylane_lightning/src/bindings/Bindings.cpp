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

#define PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(GATE_NAME)                    \
    {                                                                          \
        std::string name = (#GATE_NAME "_") + kernel_name;                     \
        std::string doc = ("Apply the " #GATE_NAME " gate using ") +           \
                          kernel_name + " kernel.";                            \
        pyclass.def(                                                           \
            name.c_str(),                                                      \
            py::overload_cast<const std::vector<size_t> &, bool,               \
                              const std::vector<Param_t> &>(                   \
                &StateVecBinder<PrecisionT>::template apply##GATE_NAME<        \
                    kernel, Param_t>),                                         \
            doc.c_str());                                                      \
    }

/// @cond DEV
namespace {
using namespace Pennylane::Algorithms;
using Pennylane::StateVectorBase;
using Pennylane::StateVectorRaw;
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
static auto create(py::array_t<complex<fp_t>> &numpyArray)
    -> StateVectorRaw<fp_t> {
    py::buffer_info numpyArrayInfo = numpyArray.request();

    if (numpyArrayInfo.ndim != 1) {
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    }
    if (numpyArrayInfo.itemsize != sizeof(complex<fp_t>)) {
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    }
    auto *data_ptr = static_cast<complex<fp_t> *>(numpyArrayInfo.ptr);
    return StateVectorRaw<fp_t>(
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
    auto state = create<fp_t>(stateNumpyArray);
    state.applyOperations(ops, wires, inverse, params);
}

/**
 * @brief Binding class for exposing C++ methods to Python.
 *
 * @tparam fp_t Floating point precision type.
 */
template <class fp_t = double>
class StateVecBinder : public StateVectorBase<fp_t, StateVecBinder<fp_t>> {
  public:
    using scalar_type_t = fp_t;
    using CFP_t = std::complex<fp_t>;
    using Base = StateVectorBase<fp_t, StateVecBinder<fp_t>>;

  private:
    CFP_t *data_;
    size_t length_;

  public:
    /**
     * @brief Construct a binding class inheriting from `%StateVector`.
     *
     * @param stateNumpyArray Complex numpy statevector data array.
     */
    explicit StateVecBinder(const py::array_t<CFP_t> &stateNumpyArray)
        : Base(Util::log2PerfectPower(
              static_cast<size_t>(stateNumpyArray.request().shape[0]))) {
        length_ = static_cast<size_t>(stateNumpyArray.request().shape[0]);
        data_ = static_cast<CFP_t *>(stateNumpyArray.request().ptr);
    }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return const CFP_t* Pointer to statevector data.
     */
    [[nodiscard]] auto getData() const -> CFP_t * { return data_; }

    /**
     * @brief Get the underlying data pointer.
     *
     * @return CFP_t* Pointer to statevector data.
     */
    auto getData() -> CFP_t * { return data_; }

    /**
     * @brief Redefine statevector data pointer.
     *
     * @param data_ptr New data pointer.
     */
    void setData(CFP_t *data) { data_ = data; }

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
        Base::template applyOperations(ops, wires, inverse);
    }

    /**
     * @brief Apply PauliX gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyPauliX(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applyPauliX_<kernel>(wires, inverse);
    }

    /**
     * @brief Apply PauliY gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyPauliY(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applyPauliY_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply PauliZ gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyPauliZ(const std::vector<size_t> &wires, bool inverse,
                     [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applyPauliZ_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply Hadamard gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void
    applyHadamard(const std::vector<size_t> &wires, bool inverse,
                  [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applyHadamard_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply S gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyS(const std::vector<size_t> &wires, bool inverse,
                [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applyS_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply T gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyT(const std::vector<size_t> &wires, bool inverse,
                [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applyT_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply CNOT (CX) gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyCNOT(const std::vector<size_t> &wires, bool inverse,
                   [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applyCNOT_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply SWAP gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation. First and second indices for
     * target wires.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applySWAP(const std::vector<size_t> &wires, bool inverse,
                   [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applySWAP_<kernel>(wires, inverse);
    }
    /**
     * @brief Apply CZ gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyCZ(const std::vector<size_t> &wires, bool inverse,
                 [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applyCZ_<kernel>(wires, inverse);
    }

    /**
     * @brief Apply CSWAP gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation. First index for control wire,
     * second and third indices for target wires.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyCSWAP(const std::vector<size_t> &wires, bool inverse,
                    [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applyCSWAP_<kernel>(wires, inverse);
    }

    /**
     * @brief Apply Toffoli (CCX) gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation. First index and second indices for
     * control wires, third index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void
    applyToffoli(const std::vector<size_t> &wires, bool inverse,
                 [[maybe_unused]] const std::vector<Param_t> &params = {}) {
        Base::template applyToffoli_<kernel>(wires, inverse);
    }

    /**
     * @brief Apply Phase-shift (\f$\textrm{diag}(1, \exp(i\theta))\f$) gate to
     * the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyPhaseShift(const std::vector<size_t> &wires, bool inverse,
                         const std::vector<Param_t> &params) {
        Base::template applyPhaseShift_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply controlled phase-shift
     * (\f$\textrm{diag}(1,1,1,\exp(i\theta))\f$) gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyControlledPhaseShift(const std::vector<size_t> &wires,
                                   bool inverse,
                                   const std::vector<Param_t> &params) {
        Base::template applyControlledPhaseShift_<kernel>(wires, inverse,
                                                          params[0]);
    }

    /**
     * @brief Apply RX (\f$exp(-i\theta\sigma_x/2)\f$) gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyRX(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<Param_t> &params) {
        Base::template applyRX_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply RY (\f$exp(-i\theta\sigma_y/2)\f$) gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyRY(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<Param_t> &params) {
        Base::template applyRY_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply RZ (\f$exp(-i\theta\sigma_z/2)\f$) gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyRZ(const std::vector<size_t> &wires, bool inverse,
                 const std::vector<Param_t> &params) {
        Base::template applyRZ_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply controlled RX gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyCRX(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        Base::template applyCRX_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply controlled RY gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyCRY(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        Base::template applyCRY_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply controlled RZ gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameter(s) for given gate. First parameter used only.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyCRZ(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        Base::template applyCRZ_<kernel>(wires, inverse, params[0]);
    }

    /**
     * @brief Apply Rot gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameters for given gate. Requires 3 values.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyRot(const std::vector<size_t> &wires, bool inverse,
                  const std::vector<Param_t> &params) {
        Base::template applyRot_<kernel>(wires, inverse, params[0], params[1],
                                         params[2]);
    }
    /**
     * @brief Apply controlled Rot gate to the given wires.
     *
     * @tparam kenel Kernel to run the operation
     * @tparam Param_t Type of parameter data.
     * @param wires Wires to apply operation. First index for control wire,
     * second index for target wire.
     * @param inverse Indicate whether to use adjoint of operation.
     * @param params Parameters for given gate. Requires 3 values.
     */
    template <KernelType kernel, class Param_t = fp_t>
    void applyCRot(const std::vector<size_t> &wires, bool inverse,
                   const std::vector<Param_t> &params) {
        Base::template applyCRot_<kernel>(wires, inverse, params[0], params[1],
                                          params[2]);
    }

    /**
     * @brief Directly apply a given matrix to the specified wires. Data in 1/2D
     * numpy complex array format.
     *
     * @tparam kenel Kernel to run the operation
     * @param matrix Numpy complex data representing matrix to apply.
     * @param wires Wires to apply given matrix.
     * @param inverse Indicate whether to take adjoint.
     */
    template <KernelType kernel>
    void applyMatrix(const py::array_t<CFP_t, py::array::c_style |
                                                  py::array::forcecast> &matrix,
                     const vector<size_t> &wires, bool inverse = false) {
        Base::template applyMatrix_<kernel>(
            static_cast<CFP_t *>(matrix.request().ptr), wires, inverse);
    }
};

template <KernelType kernel, class PrecisionT, class Param_t, class PyClass>
void registerKernelFunctions_(PyClass &&pyclass) {
    auto kernel_name = std::string(kernel_to_string(kernel));

    /* Single-qubit gates */
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(PauliX)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(PauliY)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(PauliZ)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(Hadamard)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(S)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(T)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(RX)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(RY)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(RZ)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(PhaseShift)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(Rot)
    /* Two-qubit gates */
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(ControlledPhaseShift)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(CNOT)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(CZ)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(SWAP)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(CRX)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(CRY)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(CRZ)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(CRot)
    /* Three-qubit gates */
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(Toffoli)
    PENNYLANE_BIND_PYCLASS_METHOD_FOR_KERNEL(CSWAP)

    { // applyMatrix
        std::string name = "applyMatrix_" + kernel_name;
        pyclass.def(
            name.c_str(),
            py::overload_cast<
                const py::array_t<complex<PrecisionT>,
                                  py::array::c_style | py::array::forcecast> &,
                const vector<size_t> &, bool>(
                &StateVecBinder<PrecisionT>::template applyMatrix<kernel>),
            "Apply a given matrix to wires.");
    }
}

/**
 * TODO: change to constexpr st::foreach in C++20
 * */
template <size_t idx, class PrecisionT, class Param_t, class PyClass>
void registerKernelFunctionsIter(PyClass &&pyclass) {
    if constexpr (idx < KERNELS_TO_PYEXPORT.size()) {
        registerKernelFunctions_<KERNELS_TO_PYEXPORT[idx], PrecisionT, Param_t>(
            pyclass);
        registerKernelFunctionsIter<idx + 1, PrecisionT, Param_t>(pyclass);
    }
}

constexpr auto createTupleForPybind() {
}

template <class PrecisionT, class Param_t, class PyClass>
void registerKernelFunctions(PyClass &&pyclass) {
    registerKernelFunctionsIter<0, PrecisionT, Param_t>(pyclass);
}

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

    //***********************************************************************//
    //                              StateVector
    //***********************************************************************//

    auto pyclass =
        py::class_<StateVecBinder<PrecisionT>>(m, class_name.c_str());
    pyclass.def(
        py::init<py::array_t<complex<PrecisionT>,
                             py::array::c_style | py::array::forcecast> &>());

    registerKernelFunctions<PrecisionT, Param_t>(pyclass);

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
                const std::vector<size_t> &trainableParams, size_t num_params) {
                 std::vector<std::vector<PrecisionT>> jac(
                     observables.size(),
                     std::vector<PrecisionT>(num_params, 0));
                 adj.adjointJacobian(sv.getData(), sv.getLength(), jac,
                                     observables, operations, trainableParams);
                 return py::array_t<Param_t>(py::cast(jac));
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
        .def("compute_vjp_from_jac",
             &VectorJacobianProduct<PrecisionT>::computeVJP)
        .def("compute_vjp_from_jac",
             [](VectorJacobianProduct<PrecisionT> &v,
                const std::vector<PrecisionT> &jac,
                const std::vector<PrecisionT> &dy_row, size_t m, size_t n) {
                 std::vector<PrecisionT> vjp_res(n);
                 v._computeVJP(vjp_res, jac, dy_row, m, n);
                 return py::array_t<Param_t>(py::cast(vjp_res));
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
            return py::make_tuple(py::array_t<Param_t>(py::cast(jac)),
                                  py::array_t<Param_t>(py::cast(vjp_res)));
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
    std::vector<std::string> exported_kernels;
    for (auto kernel : KERNELS_TO_PYEXPORT) {
        auto kernel_name = std::string(lookup(AVAILABLE_KERNELS, kernel));
        exported_kernels.emplace_back(kernel_name);
    }
    m.attr("EXPORTED_KERNELS") = py::cast(exported_kernels);

    /* Add DEFAULT_KERNEL_FOR_OPS */
    std::map<std::string, std::string> default_kernel_ops_map;
    for (const auto &[gate_op, name] : GATE_NAMES) {
        auto kernel = dynamic_lookup(DEFAULT_KERNEL_FOR_OPS, gate_op);
        auto kernel_name = std::string(lookup(AVAILABLE_KERNELS, kernel));
        default_kernel_ops_map.emplace(std::string(name), kernel_name);
    }
    m.attr("DEFAULT_KERNEL_FOR_OPS") = py::cast(default_kernel_ops_map);

    lightning_class_bindings<float, float>(m);
    lightning_class_bindings<double, double>(m);
}

} // namespace
  /// @endcond
