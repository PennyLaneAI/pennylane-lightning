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
#include <string>
#include <vector>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "BindingsBase.hpp"
#include "LTensorTNCudaBindings.hpp"
#include "MeasurementsTNCuda.hpp"
#include "ObservablesTNCuda.hpp"
#include "Util.hpp"

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

    using Observable = ObservableTNCuda<StateTensorT>;
    using NamedObs = NamedObsTNCuda<StateTensorT>;
    using HermitianObs = HermitianObsTNCuda<StateTensorT>;
    using TensorProdObs = TensorProdObsTNCuda<StateTensorT>;
    using Hamiltonian = HamiltonianTNCuda<StateTensorT>;

    class_name = "ObservableC" + bitsize;
    py::class_<Observable, std::shared_ptr<Observable>>(m, class_name.c_str(),
                                                        py::module_local());

    class_name = "NamedObsC" + bitsize;
    py::class_<NamedObs, std::shared_ptr<NamedObs>, Observable>(
        m, class_name.c_str(), py::module_local())
        .def(py::init(
            [](const std::string &name, const std::vector<std::size_t> &wires) {
                return NamedObs(name, wires);
            }))
        .def("__repr__", &NamedObs::getObsName)
        .def("get_wires", &NamedObs::getWires, "Get wires of observables")
        .def(
            "__eq__",
            [](const NamedObs &self, py::handle other) -> bool {
                if (!py::isinstance<NamedObs>(other)) {
                    return false;
                }
                auto other_cast = other.cast<NamedObs>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HermitianObsC" + bitsize;
    py::class_<HermitianObs, std::shared_ptr<HermitianObs>, Observable>(
        m, class_name.c_str(), py::module_local())
        .def(py::init([](const np_arr_c &matrix,
                         const std::vector<std::size_t> &wires) {
            auto buffer = matrix.request();
            const auto *ptr = static_cast<ComplexT *>(buffer.ptr);
            return HermitianObs(std::vector<ComplexT>(ptr, ptr + buffer.size),
                                wires);
        }))
        .def("__repr__", &HermitianObs::getObsName)
        .def("get_wires", &HermitianObs::getWires, "Get wires of observables")
        .def(
            "__eq__",
            [](const HermitianObs &self, py::handle other) -> bool {
                if (!py::isinstance<HermitianObs>(other)) {
                    return false;
                }
                auto other_cast = other.cast<HermitianObs>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "TensorProdObsC" + bitsize;
    py::class_<TensorProdObs, std::shared_ptr<TensorProdObs>, Observable>(
        m, class_name.c_str(), py::module_local())
        .def(py::init([](const std::vector<std::shared_ptr<Observable>> &obs) {
            return TensorProdObs(obs);
        }))
        .def("__repr__", &TensorProdObs::getObsName)
        .def("get_wires", &TensorProdObs::getWires, "Get wires of observables")
        .def(
            "__eq__",
            [](const TensorProdObs &self, py::handle other) -> bool {
                if (!py::isinstance<TensorProdObs>(other)) {
                    return false;
                }
                auto other_cast = other.cast<TensorProdObs>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HamiltonianC" + bitsize;
    using ObsPtr = std::shared_ptr<Observable>;
    py::class_<Hamiltonian, std::shared_ptr<Hamiltonian>, Observable>(
        m, class_name.c_str(), py::module_local())
        .def(py::init(
            [](const np_arr_r &coeffs, const std::vector<ObsPtr> &obs) {
                auto buffer = coeffs.request();
                const auto ptr = static_cast<const ParamT *>(buffer.ptr);
                return Hamiltonian{std::vector(ptr, ptr + buffer.size), obs};
            }))
        .def("__repr__", &Hamiltonian::getObsName)
        .def("get_wires", &Hamiltonian::getWires, "Get wires of observables")
        .def("get_coeffs", &Hamiltonian::getCoeffs,
             "Get Hamiltonian coefficients")
        .def(
            "__eq__",
            [](const Hamiltonian &self, py::handle other) -> bool {
                if (!py::isinstance<Hamiltonian>(other)) {
                    return false;
                }
                auto other_cast = other.cast<Hamiltonian>();
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
    using Measurements = MeasurementsTNCuda<StateTensorT>;
    using Observable = ObservableTNCuda<StateTensorT>;
    pyclass.def(
        "expval",
        [](Measurements &M, const std::shared_ptr<Observable> &ob) {
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
