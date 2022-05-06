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
#include "Macros.hpp"
#include "Measures.hpp"
#include "OpToMemberFuncPtr.hpp"
#include "RuntimeInfo.hpp"
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
static auto createRaw(pybind11::array_t<std::complex<PrecisionT>> &numpyArray)
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
 * @brief Create a StateVectorManagedCPU object from a 1D numpy array
 * by copying the internal data.
 *
 * @tparam PrecisionT Floating point precision type
 * @param numpyArray Numpy array data-type
 * @return StateVectorManagedCPU object.
 */
template <class PrecisionT = double>
auto createManaged(
    const pybind11::array_t<std::complex<PrecisionT>> &numpyArray)
    -> StateVectorManaged<PrecisionT> {
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
    return StateVectorManaged<PrecisionT>(
        {data_ptr, static_cast<size_t>(numpyArrayInfo.size)});
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
 * @brief Register StateVector class to pybind.
 *
 * @tparam PrecisionT Floating point type for statevector
 * @tparam ParamT Parameter type of gate operations for statevector
 * @tparam SVType Statevector type to register
 * @tparam Pyclass Pybind11's class object type
 *
 * @param pyclass Pybind11's class object to bind statevector
 */
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
            std::string(Util::lookup(Constant::gate_names, gate_op));
        const std::string doc = "Apply the " + gate_name + " gate.";
        auto func = [gate_name = gate_name](
                        SVType &sv, const std::vector<size_t> &wires,
                        bool inverse, const std::vector<ParamT> &params) {
            sv.applyOperation(gate_name, wires, inverse, params);
        };
        pyclass.def(gate_name.c_str(), func, doc.c_str());
    });
}

/**
 * @brief Return basic information of the compiled binary.
 */
auto getCompileInfo() -> pybind11::dict {
    using namespace Util::Constant;
    using namespace pybind11::literals;

    const std::string_view cpu_arch_str = [] {
        switch (cpu_arch) {
        case CPUArch::X86_64:
            return "x86_64";
        case CPUArch::PPC64:
            return "PPC64";
        case CPUArch::ARM:
            return "ARM";
        default:
            return "Unknown";
        }
    }();

    const std::string_view compiler_name_str = [] {
        switch (compiler) {
        case Compiler::GCC:
            return "GCC";
        case Compiler::Clang:
            return "Clang";
        case Compiler::MSVC:
            return "MSVC";
        case Compiler::NVCC:
            return "NVCC";
        case Compiler::NVHPC:
            return "NVHPC";
        default:
            return "Unknown";
        }
    }();

    const auto compiler_version_str = getCompilerVersion<compiler>();

    return pybind11::dict("cpu.arch"_a = cpu_arch_str,
                          "compiler.name"_a = compiler_name_str,
                          "compiler.version"_a = compiler_version_str,
                          "AVX2"_a = use_avx2, "AVX512F"_a = use_avx512f);
}

/**
 * @brief Return basic information of runtime environment
 */
auto getRuntimeInfo() -> pybind11::dict {
    using Util::RuntimeInfo;
    using namespace pybind11::literals;

    return pybind11::dict("AVX"_a = RuntimeInfo::AVX(),
                          "AVX2"_a = RuntimeInfo::AVX2(),
                          "AVX512F"_a = RuntimeInfo::AVX512F());
}
} // namespace Pennylane
