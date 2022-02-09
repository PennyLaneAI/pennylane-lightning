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
#include "JacobianProd.hpp"
#include "Measures.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "StateVectorRaw.hpp"

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
static auto create(pybind11::array_t<std::complex<PrecisionT>> &numpyArray)
    -> StateVectorRaw<PrecisionT> {
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
void apply(pybind11::array_t<std::complex<PrecisionT>> &stateNumpyArray,
           const std::vector<std::string> &ops,
           const std::vector<std::vector<size_t>> &wires,
           const std::vector<bool> &inverse,
           const std::vector<std::vector<PrecisionT>> &params) {
    auto state = create<PrecisionT>(stateNumpyArray);
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
            [](StateVectorRaw<PrecisionT> &st, const std::vector<size_t> &wires,
               bool inverse, const std::vector<ParamT> &params) {
                constexpr auto func_ptr =
                    GateOpToMemberFuncPtr<PrecisionT, ParamT,
                                          GateImplementation, gate_op>::value;
                callGateOps(func_ptr, st.getData(), st.getNumQubits(), wires,
                            inverse, params);
            };
    } else {
        return [](StateVectorRaw<PrecisionT> &st,
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

/**
 * @brief Create a tuple of lambda functions to bind
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam kernel Kernel to register
 */
template <class PrecisionT, class ParamT, Gates::KernelType kernel>
constexpr auto getGateOpLambdaPairs() {
    return getGateOpLambdaPairsIter<PrecisionT, ParamT, kernel, 0>();
}

/**
 * @brief For given kernel, register all implemented gate operations and apply
 * matrix.
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam Kernel Kernel to register
 * @tparam PyClass Pybind11 class type
 */
template <class PrecisionT, class ParamT, Gates::KernelType kernel,
          class PyClass>
void registerImplementedGatesForKernel(PyClass &pyclass) {
    using namespace Pennylane::Gates;
    const auto kernel_name = std::string(SelectKernel<kernel>::name);

    constexpr auto gate_op_lambda_pairs =
        getGateOpLambdaPairs<PrecisionT, ParamT, kernel>();

    auto registerToPyclass =
        [&pyclass, &kernel_name](auto &&gate_op_lambda_pair) -> GateOperation {
        const auto &[gate_op, func] = gate_op_lambda_pair;
        if (gate_op == GateOperation::Matrix) {
            const std::string name = "applyMatrix_" + kernel_name;
            const std::string doc = "Apply a given matrix to wires.";
            pyclass.def(name.c_str(), func, doc.c_str());
        } else {
            const auto gate_name =
                std::string(lookup(Constant::gate_names, gate_op));
            const std::string name = gate_name + "_" + kernel_name;
            const std::string doc = "Apply the " + gate_name + " gate using " +
                                    kernel_name + " kernel.";
            pyclass.def(name.c_str(), func, doc.c_str());
        }
        return gate_op;
    };

    [[maybe_unused]] const auto registerd_gate_ops = std::apply(
        [&registerToPyclass](auto... elt) {
            return std::make_tuple(registerToPyclass(elt)...);
        },
        gate_op_lambda_pairs);
}

/// @cond DEV
template <class PrecisionT, class ParamT, size_t kernel_idx, class PyClass>
void registerKernelsToPyexportIter(PyClass &pyclass) {
    if constexpr (kernel_idx < kernels_to_pyexport.size()) {
        constexpr auto kernel = kernels_to_pyexport[kernel_idx];
        registerImplementedGatesForKernel<PrecisionT, ParamT, kernel>(pyclass);
        registerKernelsToPyexportIter<PrecisionT, ParamT, kernel_idx + 1>(
            pyclass);
    }
}
/// @endcond

/**
 * @brief register gates for each kernel in kernels_to_pyexport
 *
 *
 * @tparam PrecisionT Floating point precision of underlying statevector data
 * @tparam ParamT Floating point type of gate parameters
 * @tparam PyClass Pyclass type
 */
template <class PrecisionT, class ParamT, class PyClass>
void registerKernelsToPyexport(PyClass &pyclass) {
    registerKernelsToPyexportIter<PrecisionT, ParamT, 0>(pyclass);
}
} // namespace Pennylane
