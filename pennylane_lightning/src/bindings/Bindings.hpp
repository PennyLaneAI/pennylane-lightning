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

template <typename T>
auto alignedNumpyArray(CPUMemoryModel memory_model, size_t size)
    -> pybind11::array {
    if (getAlignment<T>(memory_model) > alignof(std::max_align_t)) {
        void *ptr =
            alignedAlloc(getAlignment<T>(memory_model), sizeof(T) * size);
        auto capsule = pybind11::capsule(ptr, &alignedFree);
        return pybind11::array{
            pybind11::dtype::of<T>(), {size}, {sizeof(T)}, ptr, capsule};
    } // else
    void *ptr = malloc(sizeof(T) * size);
    auto capsule = pybind11::capsule(ptr, free);
    return pybind11::array{
        pybind11::dtype::of<T>(), {size}, {sizeof(T)}, ptr, capsule};
}

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
        return alignedNumpyArray<float>(memory_model, size);
    } else if (dt.is(pybind11::dtype::of<double>())) {
        return alignedNumpyArray<double>(memory_model, size);
    } else if (dt.is(pybind11::dtype::of<std::complex<float>>())) {
        return alignedNumpyArray<std::complex<float>>(memory_model, size);
    } else if (dt.is(pybind11::dtype::of<std::complex<double>>())) {
        return alignedNumpyArray<std::complex<double>>(memory_model, size);
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
