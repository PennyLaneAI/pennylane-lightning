// Copyright 2024 Xanadu Quantum Technologies Inc.

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
 * @file BindingsLTensor.hpp
 * Defines device-agnostic operations to export to Python and other utility
 * functions interfacing with Pybind11.
 */

#pragma once
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "CPUMemoryModel.hpp" // CPUMemoryModel, getMemoryModel, bestCPUMemoryModel, getAlignment
#include "Macros.hpp" // CPUArch
#include "Util.hpp"   // for_each_enum

#include "BindingsBase.hpp"
#include "LTensorTNCudaBindings.hpp"
#include "MeasurementsTNCuda.hpp"
#include "ObservablesTNCuda.hpp"

namespace py = pybind11;

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using namespace Pennylane::LightningTensor::TNCuda::Measures;
} // namespace
/// @endcond

namespace Pennylane {
/**
 * @brief Return basic information of the compiled binary.
 */
auto getCompileInfo() -> py::dict {
    using namespace Pennylane::Util;
    using namespace py::literals;

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

    return py::dict("cpu.arch"_a = cpu_arch_str,
                    "compiler.name"_a = compiler_name_str,
                    "compiler.version"_a = compiler_version_str,
                    "AVX2"_a = use_avx2, "AVX512F"_a = use_avx512f);
}

/**
 * @brief Register bindings for general info.
 *
 * @param m Pybind11 module.
 */
void registerInfo(py::module_ &m) {
    /* Add compile info */
    m.def("compile_info", &getCompileInfo, "Compiled binary information.");
}

/**
 * @brief Register observable classes.
 *
 * @tparam StateTensorT
 * @param m Pybind module
 */
template <class StateTensorT>
void registerBackendAgnosticObservables(py::module_ &m) {
    using PrecisionT =
        typename StateTensorT::PrecisionT; // Statetensor's precision.
    using ComplexT =
        typename StateTensorT::ComplexT; // Statetensor's complex type.
    using ParamT = PrecisionT;           // Parameter's data precision

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;
    using np_arr_r = py::array_t<ParamT, py::array::c_style>;

    std::string class_name;

    class_name = "ObservableC" + bitsize;
    py::class_<ObservableTNCuda<StateTensorT>,
               std::shared_ptr<ObservableTNCuda<StateTensorT>>>(
        m, class_name.c_str(), py::module_local());

    class_name = "NamedObsC" + bitsize;
    py::class_<NamedObsTNCuda<StateTensorT>,
               std::shared_ptr<NamedObsTNCuda<StateTensorT>>,
               ObservableTNCuda<StateTensorT>>(m, class_name.c_str(),
                                               py::module_local())
        .def(py::init(
            [](const std::string &name, const std::vector<std::size_t> &wires) {
                return NamedObsTNCuda<StateTensorT>(name, wires);
            }))
        .def("__repr__", &NamedObsTNCuda<StateTensorT>::getObsName)
        .def("get_wires", &NamedObsTNCuda<StateTensorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const NamedObsTNCuda<StateTensorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<NamedObsTNCuda<StateTensorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<NamedObsTNCuda<StateTensorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HermitianObsC" + bitsize;
    py::class_<HermitianObsTNCuda<StateTensorT>,
               std::shared_ptr<HermitianObsTNCuda<StateTensorT>>,
               ObservableTNCuda<StateTensorT>>(m, class_name.c_str(),
                                               py::module_local())
        .def(py::init(
            [](const np_arr_c &matrix, const std::vector<std::size_t> &wires) {
                auto buffer = matrix.request();
                const auto *ptr = static_cast<ComplexT *>(buffer.ptr);
                return HermitianObsTNCuda<StateTensorT>(
                    std::vector<ComplexT>(ptr, ptr + buffer.size), wires);
            }))
        .def("__repr__", &HermitianObsTNCuda<StateTensorT>::getObsName)
        .def("get_wires", &HermitianObsTNCuda<StateTensorT>::getWires,
             "Get wires of observables")
        .def("get_matrix", &HermitianObsTNCuda<StateTensorT>::getMatrix,
             "Get matrix representation of Hermitian operator")
        .def(
            "__eq__",
            [](const HermitianObsTNCuda<StateTensorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<HermitianObsTNCuda<StateTensorT>>(other)) {
                    return false;
                }
                auto other_cast =
                    other.cast<HermitianObsTNCuda<StateTensorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "TensorProdObsC" + bitsize;
    py::class_<TensorProdObsTNCuda<StateTensorT>,
               std::shared_ptr<TensorProdObsTNCuda<StateTensorT>>,
               ObservableTNCuda<StateTensorT>>(m, class_name.c_str(),
                                               py::module_local())
        .def(py::init(
            [](const std::vector<
                std::shared_ptr<ObservableTNCuda<StateTensorT>>> &obs) {
                return TensorProdObsTNCuda<StateTensorT>(obs);
            }))
        .def("__repr__", &TensorProdObsTNCuda<StateTensorT>::getObsName)
        .def("get_wires", &TensorProdObsTNCuda<StateTensorT>::getWires,
             "Get wires of observables")
        .def("get_ops", &TensorProdObsTNCuda<StateTensorT>::getObs,
             "Get operations list")
        .def(
            "__eq__",
            [](const TensorProdObsTNCuda<StateTensorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<TensorProdObsTNCuda<StateTensorT>>(other)) {
                    return false;
                }
                auto other_cast =
                    other.cast<TensorProdObsTNCuda<StateTensorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HamiltonianC" + bitsize;
    using ObsPtr = std::shared_ptr<ObservableTNCuda<StateTensorT>>;
    py::class_<HamiltonianTNCuda<StateTensorT>,
               std::shared_ptr<HamiltonianTNCuda<StateTensorT>>,
               ObservableTNCuda<StateTensorT>>(m, class_name.c_str(),
                                               py::module_local())
        .def(py::init(
            [](const np_arr_r &coeffs, const std::vector<ObsPtr> &obs) {
                auto buffer = coeffs.request();
                const auto ptr = static_cast<const ParamT *>(buffer.ptr);
                return HamiltonianTNCuda<StateTensorT>{
                    std::vector(ptr, ptr + buffer.size), obs};
            }))
        .def("__repr__", &HamiltonianTNCuda<StateTensorT>::getObsName)
        .def("get_wires", &HamiltonianTNCuda<StateTensorT>::getWires,
             "Get wires of observables")
        .def("get_ops", &HamiltonianTNCuda<StateTensorT>::getObs,
             "Get operations contained by Hamiltonian")
        .def("get_coeffs", &HamiltonianTNCuda<StateTensorT>::getCoeffs,
             "Get Hamiltonian coefficients")
        .def(
            "__eq__",
            [](const HamiltonianTNCuda<StateTensorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<HamiltonianTNCuda<StateTensorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<HamiltonianTNCuda<StateTensorT>>();
                return self == other_cast;
            },
            "Compare two observables");
}

/**
 * @brief Register agnostic measurements class functionalities.
 *
 * @tparam StateTensorT
 * @tparam PyClass
 * @param pyclass Pybind11's measurements class to bind methods.
 */
template <class StateTensorT, class PyClass>
void registerBackendAgnosticMeasurements(PyClass &pyclass) {
    pyclass.def(
        "expval",
        [](MeasurementsTNCuda<StateTensorT> &M,
           const std::shared_ptr<ObservableTNCuda<StateTensorT>> &ob) {
            return M.expval(*ob);
        },
        "Expected value of an observable object.");
}

/**
 * @brief Templated class to build lightning class bindings.
 *
 * @tparam StateVectorT State vector type
 * @param m Pybind11 module.
 */
template <class StateTensorT> void lightningClassBindings(py::module_ &m) {
    using PrecisionT =
        typename StateTensorT::PrecisionT; // StateTensor's precision.
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    //***********************************************************************//
    //                              StateTensor
    //***********************************************************************//
    std::string class_name = "StateTensorC" + bitsize;
    auto pyclass =
        py::class_<StateTensorT>(m, class_name.c_str(), py::module_local());

    registerBackendClassSpecificBindings<StateTensorT>(pyclass);

    //***********************************************************************//
    //                              Observables
    //***********************************************************************//
    /* Observables submodule */
    py::module_ obs_submodule =
        m.def_submodule("observables", "Submodule for observables classes.");
    registerBackendAgnosticObservables<StateTensorT>(obs_submodule);

    //***********************************************************************//
    //                             Measurements
    //***********************************************************************//
    class_name = "MeasurementsC" + bitsize;
    auto pyclass_measurements = py::class_<MeasurementsTNCuda<StateTensorT>>(
        m, class_name.c_str(), py::module_local());

    pyclass_measurements.def(py::init<const StateTensorT &>());
    registerBackendAgnosticMeasurements<StateTensorT>(pyclass_measurements);
}

template <typename TypeList>
void registerLightningClassBindings(py::module_ &m) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateTensorT = typename TypeList::Type;
        lightningClassBindings<StateTensorT>(m);
        registerLightningClassBindings<typename TypeList::Next>(m);
        py::register_local_exception<Pennylane::Util::LightningException>(
            m, "LightningException");
    }
}
} // namespace Pennylane
