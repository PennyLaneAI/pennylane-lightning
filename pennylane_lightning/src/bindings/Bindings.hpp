// Copyright 2022 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/**
 * @file Bindings.hpp
 * Defines operations to export to Python and other utility functions
 * interfacing with Pybind11
 */
#pragma once
#include "AdjointDiff.hpp"
#include "CPUMemoryModel.hpp"
#include "JacobianProd.hpp"
#include "Measures.hpp"
#include "Memory.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "StateVectorManagedCPU.hpp"

#include "pybind11/complex.h"
#include "pybind11/functional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include <cassert>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>

namespace Pennylane {
/**
 * @brief Create a `%StateVector` object from a 1D numpy complex data array.
 *
 * @tparam PrecisionT Precision data type
 * @param numpyArray Numpy data array.
 * @return StateVector<PrecisionT> `%StateVector` object.
 */
template <class PrecisionT = double>
auto createRaw(const pybind11::array_t<std::complex<PrecisionT>> &numpyArray)
    -> StateVectorRawCPU<PrecisionT> {
    pybind11::buffer_info numpyArrayInfo = numpyArray.request();

    if (numpyArrayInfo.ndim != 1) {
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    }
    if (numpyArrayInfo.itemsize != sizeof(std::complex<PrecisionT>)) {
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    }
    auto *data_ptr =
        static_cast<std::complex<PrecisionT> *>(numpyArrayInfo.ptr);
    return StateVectorRawCPU<PrecisionT>(
        {data_ptr, static_cast<size_t>(numpyArrayInfo.shape[0])});
}

template <class PrecisionT = double>
auto createManaged(
    const pybind11::array_t<std::complex<PrecisionT>> &numpyArray)
    -> StateVectorManagedCPU<PrecisionT> {
    pybind11::buffer_info numpyArrayInfo = numpyArray.request();

    if (numpyArrayInfo.ndim != 1) {
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    }
    if (numpyArrayInfo.itemsize != sizeof(std::complex<PrecisionT>)) {
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    }
    auto *data_ptr =
        static_cast<std::complex<PrecisionT> *>(numpyArrayInfo.ptr);
    return StateVectorManagedCPU<PrecisionT>(
        {data_ptr, static_cast<size_t>(numpyArrayInfo.shape[0])});
}

template <class PrecisionT = double>
auto toNumpyArray(const StateVectorManagedCPU<PrecisionT> &sv)
    -> pybind11::array_t<std::complex<PrecisionT>> {
    return pybind11::array_t<std::complex<PrecisionT>>(
        {sv.getLength()}, {2 * sizeof(PrecisionT)}, sv.getData());
}

auto getNumpyArrayAlignment(const pybind11::array &numpyArray)
    -> CPUMemoryModel {
    return getMemoryModel(numpyArray.request().ptr);
}

void deallocateArray(void *ptr) { std::free(ptr); }

/**
 * @brief We return an numpy array whose underlying data is allocated by
 * lightning.
 *
 * See https://github.com/pybind/pybind11/issues/1042#issuecomment-325941022
 * for capsule usage.
 */
auto allocateAlignedArray(size_t size, pybind11::dtype dt) -> pybind11::array {

    auto memory_model = bestCPUMemoryModel();

    if (dt.is(pybind11::dtype::of<float>())) {
        void *ptr = std::aligned_alloc(getAlignment<float>(memory_model),
                                       sizeof(float) * size);
        auto capsule = pybind11::capsule(ptr, &deallocateArray);

        return pybind11::array{dt, {size}, {sizeof(float)}, ptr, capsule};
    } else if (dt.is(pybind11::dtype::of<double>())) {
        void *ptr = std::aligned_alloc(getAlignment<double>(memory_model),
                                       sizeof(double) * size);
        auto capsule = pybind11::capsule(ptr, &deallocateArray);

        return pybind11::array{dt, {size}, {sizeof(double)}, ptr, capsule};
    } else if (dt.is(pybind11::dtype::of<std::complex<float>>())) {
        void *ptr =
            std::aligned_alloc(getAlignment<std::complex<float>>(memory_model),
                               sizeof(std::complex<float>) * size);
        auto capsule = pybind11::capsule(ptr, &deallocateArray);

        return pybind11::array{
            dt, {size}, {sizeof(std::complex<float>)}, ptr, capsule};
    } else if (dt.is(pybind11::dtype::of<std::complex<double>>())) {
        void *ptr =
            std::aligned_alloc(getAlignment<std::complex<double>>(memory_model),
                               sizeof(std::complex<double>) * size);
        auto capsule = pybind11::capsule(ptr, &deallocateArray);

        return pybind11::array{
            dt, {size}, {sizeof(std::complex<double>)}, ptr, capsule};
    } else {
        throw pybind11::type_error("Unsupported datatype.");
    }
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
void apply(pybind11::array_t<std::complex<PrecisionT>> &stateNumpyArray,
           const std::vector<std::string> &ops,
           const std::vector<std::vector<size_t>> &wires,
           const std::vector<bool> &inverse,
           const std::vector<std::vector<PrecisionT>> &params) {
    auto state = createRaw<PrecisionT>(stateNumpyArray);
    state.applyOperations(ops, wires, inverse, params);
}

/**
 * @brief Return a specific lambda function for the given kernel and gate
 * operation
 *
 * We do not expect template parameters kernel and gate_op can be function
 * parameters as we want the lambda function to be a stateless.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam kernel Kernel to register
 * @tparam gate_op Gate operation
 */
/*
template <class PrecisionT, class ParamT, Gates::KernelType kernel,
          Gates::GateOperation gate_op>
constexpr auto getLambdaForKernelGateOp() {
    namespace py = pybind11;
    using namespace Pennylane::Gates;
    using GateImplementation = SelectKernel<kernel>;

    static_assert(array_has_elt(GateImplementation::implemented_gates, gate_op),
                  "The operator to register must be implemented.");

    if constexpr (gate_op != GateOperation::Matrix) {
        return
            [](StateVectorRawCPU<PrecisionT> &st, const std::vector<size_t>
&wires, bool inverse, const std::vector<ParamT> &params) { constexpr auto
func_ptr = GateOpToMemberFuncPtr<PrecisionT, ParamT, GateImplementation,
gate_op>::value; callGateOps(func_ptr, st.getData(), st.getNumQubits(), wires,
                            inverse, params);
            };
    } else {
        return [](StateVectorRawCPU<PrecisionT> &st,
                  const py::array_t<std::complex<PrecisionT>,
                                    py::array::c_style | py::array::forcecast>
                      &matrix,
                  const std::vector<size_t> &wires, bool inverse = false) {
            st.template applyMatrix_<kernel>(
                static_cast<std::complex<PrecisionT> *>(matrix.request().ptr),
                wires, inverse);
        };
    }
};
*/
/*
/// @cond DEV
template <class PrecisionT, class ParamT, Gates::KernelType kernel,
          size_t gate_idx>
constexpr auto getGateOpLambdaPairsIter() {
    using Pennylane::Gates::SelectKernel;
    if constexpr (gate_idx < SelectKernel<kernel>::implemented_gates.size()) {
        constexpr auto gate_op =
            SelectKernel<kernel>::implemented_gates[gate_idx];
        return prepend_to_tuple(
            std::pair{gate_op, getLambdaForKernelGateOp<PrecisionT, ParamT,
                                                        kernel, gate_op>()},
            getGateOpLambdaPairsIter<PrecisionT, ParamT, kernel,
                                     gate_idx + 1>());
    } else {
        return std::tuple{};
    }
}
/// @endcond
*/
/**
 * @brief Create a tuple of lambda functions to bind
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam kernel Kernel to register
 */
/*
template <class PrecisionT, class ParamT, Gates::KernelType kernel>
constexpr auto getGateOpLambdaPairs() {
    return getGateOpLambdaPairsIter<PrecisionT, ParamT, kernel, 0>();
}
*/

/**
 * @brief For given kernel, register all implemented gate operations and apply
 * matrix.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam Kernel Kernel to register
 * @tparam PyClass Pybind11 class type
 */
/*
template <class PrecisionT, class ParamT, class PyClass>
void registerImplementedGatesForKernel(PyClass &pyclass) {
    using namespace Pennylane::Gates;

    auto registerToPyclass =
        [&pyclass](auto &&gate_op_lambda_pair) -> GateOperation {
        const auto &[gate_op, func] = gate_op_lambda_pair;
        if (gate_op == GateOperation::Matrix) {
            const std::string name = "applyMatrix_" + kernel_name;
            const std::string doc = "Apply a given matrix to wires.";
            pyclass.def(name.c_str(), func, doc.c_str());
        } else {
            const auto gate_name =
                std::string(lookup(Constant::gate_names, gate_op));
            const std::string doc = "Apply the " + gate_name + " gate.";
            auto func = [&gate_name](StateVectorManagedCPU<PrecisionT>& sv,
                                     const std::vector<size_t> &wires,
                                     bool inverse,
                                     const std::vector<ParamT> &params) {
                sv.applyOperation(gate_name, wires, inverse, params);
            }
            pyclass.def(name.c_str(), , doc.c_str());
        }
        return gate_op;
    };

    [[maybe_unused]] const auto registerd_gate_ops = std::apply(
        [&registerToPyclass](auto... elt) {
            return std::make_tuple(registerToPyclass(elt)...);
        },
        gate_op_lambda_pairs);
}
*/
/// @cond DEV
template <class PrecisionT, class ParamT, class SVType, class PyClass>
void registerGatesForStateVector(PyClass &pyclass) {
    using Gates::GateOperation;
    namespace Constant = Gates::Constant;

    static_assert(std::is_same_v<typename SVType::PrecisionT, PrecisionT>);

    { // Register matrix
        const std::string doc = "Apply a given matrix to wires.";
        auto func =
            [](SVType &st,
               const pybind11::array_t<std::complex<PrecisionT>,
                                       pybind11::array::c_style |
                                           pybind11::array::forcecast> &matrix,
               const std::vector<size_t> &wires, bool inverse = false) {
                st.applyMatrix(static_cast<const std::complex<PrecisionT> *>(
                                   matrix.request().ptr),
                               wires, inverse);
            };
        pyclass.def("applyMatrix", func, doc.c_str());
    }

    Util::for_each_enum<GateOperation>([&pyclass](GateOperation gate_op) {
        const auto gate_name =
            std::string(lookup(Constant::gate_names, gate_op));
        const std::string doc = "Apply the " + gate_name + " gate.";
        auto func = [gate_name = gate_name](
                        SVType &sv, const std::vector<size_t> &wires,
                        bool inverse, const std::vector<ParamT> &params) {
            sv.applyOperation(gate_name, wires, inverse, params);
        };
        pyclass.def(gate_name.c_str(), func, doc.c_str());
    });
}
} // namespace Pennylane
