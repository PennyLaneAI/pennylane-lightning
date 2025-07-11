// Copyright 2025 Xanadu Quantum Technologies Inc.

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
 * @file LTensorTNCudaBindings_nb.hpp
 * Defines lightning.tensor specific operations to export to Python using
 * Nanobind.
 */

#pragma once
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "BindingsUtils_nb.hpp"
#include "DevTag.hpp"
#include "Error.hpp"
#include "ExactTNCuda.cpp"
#include "MPSTNCuda.hpp"
#include "MeasurementsTNCuda.hpp"
#include "ObservablesTNCuda.hpp"
#include "TypeList.hpp"
#include "Util.hpp"
#include "cuda_helpers.hpp"
#include "tncuda_helpers.hpp"

namespace nb = nanobind;

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using namespace Pennylane::LightningTensor::TNCuda::Measures;
using Pennylane::NanoBindings::Utils::createNumpyArrayFromVector;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::NanoBindings {

/**
 * @brief Define StateVector backends for lightning.tensor
 */
using TensorNetworkBackends =
    Pennylane::Util::TypeList<MPSTNCuda<float>, MPSTNCuda<double>,
                              ExactTNCuda<float>, ExactTNCuda<double>, void>;

/**
 * @brief Get a gate kernel map for a tensor network using MPS.
 *
 * @tparam TensorNetT
 * @tparam PyClass
 * @param pyclass Nanobind's tensornet class to bind methods.
 */
template <class TensorNet, class PyClass>
void registerBackendClassSpecificBindingsMPS(PyClass &pyclass) {
    using PrecisionT = typename TensorNet::PrecisionT; // TensorNet's precision
    using ParamT = PrecisionT; // Parameter's data precision
    using ArrayT = nb::ndarray<std::complex<ParamT>, nb::c_contig>;

    pyclass.def(
        nb::init<std::size_t, std::size_t>()); // num_qubits, max_bond_dim
    pyclass.def(nb::init<std::size_t, std::size_t,
                         DevTag<int>>()); // num_qubits, max_bond_dim, dev-tag
    pyclass.def(
        "getState",
        [](TensorNet &tensor_network, ArrayT &state) {
            tensor_network.getData(state.data(), state.size());
        },
        "Copy StateVector data into a Numpy array.");
    pyclass.def(
        "updateMPSSitesData",
        [](TensorNet &tensor_network, std::vector<ArrayT> &tensors) {
            // Extract the incoming MPS shape
            std::vector<std::vector<std::size_t>> MPS_shape_source;
            // TODO: Question for reviewers: these are actually pointers to
            // int64_t, not size_t. Do we anticipate this being an issue?
            for (std::size_t idx = 0; idx < tensors.size(); idx++) {
                std::vector<std::size_t> MPS_site_source(
                    tensors[idx].shape_ptr(),
                    tensors[idx].shape_ptr() + tensors[idx].ndim());
                // TODO: Can probably do this without a for loop and emplace
                // back
                MPS_shape_source.emplace_back(std::move(MPS_site_source));
            }

            const auto &MPS_shape_dest = tensor_network.getSitesExtents();
            MPSShapeCheck(MPS_shape_dest, MPS_shape_source);

            for (std::size_t idx = 0; idx < tensors.size(); idx++) {
                tensor_network.updateSiteData(idx, tensors[idx].data(),
                                              tensors[idx].size());
            }
        },
        "Pass MPS site data to the C++ backend.");
    pyclass.def(
        "setBasisState",
        [](TensorNet &tensor_network, std::vector<std::size_t> &basisState) {
            tensor_network.setBasisState(basisState);
        },
        "Create Basis State on GPU.");
    pyclass.def(
        "applyMPOOperation",
        [](TensorNet &tensor_network, std::vector<ArrayT> &tensors,
           const std::vector<std::size_t> &wires, std::size_t MPOBondDims) {
            using ComplexT = typename TensorNet::ComplexT;
            std::vector<std::vector<ComplexT>> conv_tensors;
            for (const auto &tensor : tensors) {
                conv_tensors.push_back(std::vector<ComplexT>{
                    tensor.data(), tensor.data() + tensor.size()});
            }
            tensor_network.applyMPOOperation(conv_tensors, wires, MPOBondDims);
        },
        "Apply MPO to the tensor network graph.");
    pyclass.def(
        "appendMPSFinalState",
        [](TensorNet &tensor_network, double cutoff,
           const std::string &cutoff_mode) {
            tensor_network.append_mps_final_state(cutoff, cutoff_mode);
        },
        "Get the final state.");
    pyclass.def("reset", &TensorNet::reset, "Reset the statevector.");
}

/**
 * @brief Get a gate kernel map for a tensor network using ExactTN.
 *
 * @tparam TensorNetT
 * @tparam PyClass
 * @param pyclass Nanobind's tensornet class to bind methods.
 */
template <class TensorNet, class PyClass>
void registerBackendClassSpecificBindingsExactTNCuda(PyClass &pyclass) {
    using PrecisionT = typename TensorNet::PrecisionT; // TensorNet's precision
    using ParamT = PrecisionT; // Parameter's data precision
    using ArrayT = nb::ndarray<std::complex<ParamT>, nb::c_contig>;

    pyclass.def(nb::init<std::size_t>());              // num_qubits
    pyclass.def(nb::init<std::size_t, DevTag<int>>()); // num_qubits, dev-tag
    pyclass.def(
        "getState",
        [](TensorNet &tensor_network, ArrayT &state) {
            tensor_network.getData(state.data(), state.size());
        },
        "Copy StateVector data into a Numpy array.");
    pyclass.def(
        "setBasisState",
        [](TensorNet &tensor_network, std::vector<std::size_t> &basisState) {
            tensor_network.setBasisState(basisState);
        },
        "Create Basis State on GPU.");
    pyclass
        .def(
            "updateMPSSitesData",
            [](TensorNet &tensor_network, std::vector<ArrayT> &tensors) {
                for (std::size_t idx = 0; idx < tensors.size(); idx++) {
                    tensor_network.updateSiteData(idx, tensors[idx].data(),
                                                  tensors[idx].size());
                }
            },
            "Pass MPS site data to the C++ backend.")
        .def("reset", &TensorNet::reset, "Reset the statevector.");
}

/**
 * @brief Get a controlled matrix and kernel map for a statevector.
 * @tparam TensorNet
 * @tparam PyClass
 * @param pyclass Nanobind's tensornet class to bind methods.
 */
template <class TensorNet, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    if constexpr (std::is_same_v<TensorNet, MPSTNCuda<double>> ||
                  std::is_same_v<TensorNet, MPSTNCuda<float>>) {
        registerBackendClassSpecificBindingsMPS<TensorNet>(pyclass);
    }
    if constexpr (std::is_same_v<TensorNet, ExactTNCuda<double>> ||
                  std::is_same_v<TensorNet, ExactTNCuda<float>>) {
        registerBackendClassSpecificBindingsExactTNCuda<TensorNet>(pyclass);
    }
} // pyclass

/**
 * @brief Register backend specific measurements class functionalities.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's measurements class to bind methods.
 */
template <class TensorNetT, class PyClass>
void registerBackendSpecificMeasurements(PyClass &pyclass) {
    using MeasurementsT = MeasurementsTNCuda<TensorNetT>;
    using ObservableT = ObservableTNCuda<TensorNetT>;
    pyclass.def("generate_samples", [](MeasurementsT &M,
                                       const std::vector<std::size_t> &wires,
                                       std::size_t num_shots) {
        constexpr auto sz = sizeof(std::size_t);
        const std::size_t num_wires = wires.size();
        const std::size_t ndim = 2;
        const std::vector<std::size_t> shape{num_shots, num_wires};
        auto &&result = M.generate_samples(wires, num_shots);

        return createNumpyArrayFromVector<std::size_t>(result, num_shots,
                                                       num_wires);
    });
} // pyclass

/**
 * @brief Register backend specific observables.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendSpecificObservables(nb::module_ &) {} // m

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithms(nb::module_ &) {} // m

/**
 * @brief Provide backend information.
 */
auto getBackendInfo() -> nb::dict {
    nb::dict info;
    info["NAME"] = "lightning.tensor";
    return info;
}

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Nanobind module.
 */
void registerBackendSpecificInfo(nb::module_ &) {
    m.def("backend_info", &getBackendInfo, "Backend-specific information.");
    registerCudaUtils(m);
} // m

} // namespace Pennylane::LightningTensor::TNCuda::NanoBindings
