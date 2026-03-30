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
 * @file LTensorTNCudaBindings.hpp
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
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "BindingsUtils.hpp"
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

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using namespace Pennylane::LightningTensor::TNCuda::Measures;
using namespace Pennylane::Util::NanoBindings;
using Pennylane::NanoBindings::Utils::createNumpyArrayFromVector;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::NanoBindings {

namespace nb = nanobind;

/// @cond DEV
/**
 * @brief Define TensorNet backends for lightning.tensor
 */
using TensorNetworkBackends =
    Pennylane::Util::TypeList<MPSTNCuda<float>, MPSTNCuda<double>,
                              ExactTNCuda<float>, ExactTNCuda<double>, void>;
/// @endcond

/**
 * @brief Get a gate kernel map for a tensor network using MPS.
 *
 * @tparam TensorNetT
 * @tparam PyClass
 * @param pyclass Nanobind's tensornet class to bind methods.
 */
template <class TensorNetT, class PyClass>
void registerBackendClassSpecificBindingsMPS(PyClass &pyclass) {
    using PrecisionT =
        typename TensorNetT::PrecisionT; // TensorNetT's precision
    using ArrayT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;

    pyclass.def(
        nb::init<std::size_t, std::size_t>()); // num_qubits, max_bond_dim
    pyclass.def(nb::init<std::size_t, std::size_t,
                         DevTag<int>>()); // num_qubits, max_bond_dim, dev-tag
    pyclass.def(
        "getState",
        [](TensorNetT &tensor_network, ArrayT &state) {
            tensor_network.getData(state.data(), state.size());
        },
        "Copy tensor network data into a Numpy array.");
    pyclass.def(
        "updateMPSSitesData",
        [](TensorNetT &tensor_network, std::vector<ArrayT> &tensors) {
            // Extract the incoming MPS shape
            std::vector<std::vector<std::size_t>> MPS_shape_source;
            MPS_shape_source.resize(tensors.size());
            // Get shape of each tensor
            std::transform(tensors.begin(), tensors.end(),
                           MPS_shape_source.begin(), [](const ArrayT &tensor) {
                               std::vector<std::size_t> shape;
                               shape.resize(tensor.ndim());
                               // Fill the shape vector with tensor dimensions
                               for (std::size_t i = 0; i < tensor.ndim(); i++) {
                                   shape[i] = tensor.shape(i);
                               }
                               return shape;
                           });
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
        [](TensorNetT &tensor_network, std::vector<std::size_t> &basisState) {
            tensor_network.setBasisState(basisState);
        },
        "Create Basis State on GPU.");
    pyclass.def(
        "applyMPOOperation",
        [](TensorNetT &tensor_network, std::vector<ArrayT> &tensors,
           const std::vector<std::size_t> &wires, std::size_t MPOBondDims) {
            using ComplexT = typename TensorNetT::ComplexT;
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
        [](TensorNetT &tensor_network, double cutoff,
           const std::string &cutoff_mode) {
            tensor_network.append_mps_final_state(cutoff, cutoff_mode);
        },
        "Get the final state.");
    pyclass.def("reset", &TensorNetT::reset, "Reset the tensor network.");
    pyclass.def(
        "setWorksizePref",
        [](TensorNetT &tensor_network, std::string_view pref) {
            tensor_network.setWorksizePref(pref);
        },
        "Set Workspace Size Preference.");
}

/**
 * @brief Get a gate kernel map for a tensor network using ExactTN.
 *
 * @tparam TensorNetT
 * @tparam PyClass
 * @param pyclass Nanobind's tensornet class to bind methods.
 */
template <class TensorNetT, class PyClass>
void registerBackendClassSpecificBindingsExactTNCuda(PyClass &pyclass) {
    using PrecisionT =
        typename TensorNetT::PrecisionT; // TensorNetT's precision
    using ArrayT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;

    pyclass.def(nb::init<std::size_t>());              // num_qubits
    pyclass.def(nb::init<std::size_t, DevTag<int>>()); // num_qubits, dev-tag
    pyclass.def(
        "getState",
        [](TensorNetT &tensor_network, ArrayT &state) {
            tensor_network.getData(state.data(), state.size());
        },
        "Copy tensor network data into a Numpy array.");
    pyclass.def(
        "setBasisState",
        [](TensorNetT &tensor_network, std::vector<std::size_t> &basisState) {
            tensor_network.setBasisState(basisState);
        },
        "Create Basis State on GPU.");
    pyclass.def(
        "updateMPSSitesData",
        [](TensorNetT &tensor_network, std::vector<ArrayT> &tensors) {
            for (std::size_t idx = 0; idx < tensors.size(); idx++) {
                tensor_network.updateSiteData(idx, tensors[idx].data(),
                                              tensors[idx].size());
            }
        },
        "Pass MPS site data to the C++ backend.");
    pyclass.def("reset", &TensorNetT::reset, "Reset the tensor network.");
    pyclass.def(
        "setWorksizePref",
        [](TensorNetT &tensor_network, std::string_view pref) {
            tensor_network.setWorksizePref(pref);
        },
        "Set Workspace Size Preference.");
}

/**
 * @brief Get a controlled matrix and kernel map for a tensor network.
 * @tparam TensorNetT
 * @tparam PyClass
 * @param pyclass Nanobind's tensornet class to bind methods.
 */
template <class TensorNetT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    if constexpr (std::is_same_v<TensorNetT, MPSTNCuda<double>> ||
                  std::is_same_v<TensorNetT, MPSTNCuda<float>>) {
        registerBackendClassSpecificBindingsMPS<TensorNetT>(pyclass);
    }
    if constexpr (std::is_same_v<TensorNetT, ExactTNCuda<double>> ||
                  std::is_same_v<TensorNetT, ExactTNCuda<float>>) {
        registerBackendClassSpecificBindingsExactTNCuda<TensorNetT>(pyclass);
    }
} // pyclass

/**
 * @brief Register backend specific measurements class functionalities.
 *
 * @tparam TensorNetT
 * @tparam PyClass
 * @param pyclass Nanobind's measurements class to bind methods.
 */
template <class TensorNetT, class PyClass>
void registerBackendSpecificMeasurements(PyClass &pyclass) {
    using MeasurementsT = MeasurementsTNCuda<TensorNetT>;
    pyclass.def("generate_samples", [](MeasurementsT &M,
                                       const std::vector<std::size_t> &wires,
                                       std::size_t num_shots) {
        const std::size_t num_wires = wires.size();
        const std::vector<std::size_t> shape{num_shots, num_wires};
        auto &&result = M.generate_samples(wires, num_shots);

        return createNumpyArrayFromVector<std::size_t>(std::move(result),
                                                       num_shots, num_wires);
    });
} // pyclass

/**
 * @brief Register backend specific observables.
 *
 * @tparam TensorNetT
 * @param m Nanobind module
 */
template <class TensorNetT>
void registerBackendSpecificObservables(nb::module_ &) {} // m

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam TensorNetT
 * @param m Nanobind module
 */
template <class TensorNetT>
void registerBackendSpecificAlgorithms(nb::module_ &) {} // m

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Nanobind module.
 */
void registerBackendSpecificInfo(nb::module_ &m) {
    m.def(
        "backend_info",
        []() {
            nb::dict info;

            info["NAME"] = "lightning.tensor";

            return info;
        },
        "Backend-specific information.");
    registerCudaUtils(m);
} // m

} // namespace Pennylane::LightningTensor::TNCuda::NanoBindings
