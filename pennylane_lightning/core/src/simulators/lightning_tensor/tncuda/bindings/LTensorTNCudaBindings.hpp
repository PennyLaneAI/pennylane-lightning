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
 * @file LTensorTNCudaBindings.hpp
 * Defines LightningTensor-specific operations to export to Python, other
 * utility functions interfacing with Pybind11 and support to agnostic bindings.
 */

#pragma once
#include <vector>

#include "cuda.h"

#include "BindingsBase.hpp"
#include "DevTag.hpp"
#include "DevicePool.hpp"
#include "Error.hpp"
#include "MPSTNCuda.hpp"
#include "TypeList.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Bindings;
using Pennylane::LightningTensor::TNCuda::MPSTNCuda;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane::LightningTensor::TNCuda {
using TensorNetBackends =
    Pennylane::Util::TypeList<MPSTNCuda<float>, MPSTNCuda<double>, void>;

/**
 * @brief Get a gate kernel map for a tensor network.
 */
template <class TensorNet, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    registerGatesForTensorNet<TensorNet>(pyclass);

    pyclass
        .def(py::init<const std::size_t,
                      const std::size_t>()) // num_qubits, max_bond_dim
        .def(py::init<const std::size_t, const std::size_t,
                      DevTag<int>>()) // num_qubits, max_bond_dim, dev-tag
        .def(
            "setBasisState",
            [](TensorNet &tensor_network,
               std::vector<std::size_t> &basisState) {
                tensor_network.setBasisState(basisState);
            },
            "Create Basis State on GPU.")
        .def(
            "appendMPSFinalState",
            [](TensorNet &tensor_network, double cutoff,
               std::string cutoff_mode) {
                tensor_network.append_mps_final_state(cutoff, cutoff_mode);
            },
            "Get the final state.")
        .def("reset", &TensorNet::reset, "Reset the statevector.");
}

/**
 * @brief Provide backend information.
 */
auto getBackendInfo() -> py::dict {
    using namespace py::literals;

    return py::dict("NAME"_a = "lightning.tensor");
}

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Pybind11 module.
 */
// TODO Move this method to a separate module for both LGPU and LTensor usage.
void registerBackendSpecificInfo(py::module_ &m) {
    m.def("backend_info", &getBackendInfo, "Backend-specific information.");
    m.def("device_reset", &deviceReset, "Reset all GPU devices and contexts.");
    m.def("allToAllAccess", []() {
        for (int i = 0; i < static_cast<int>(getGPUCount()); i++) {
            cudaDeviceEnablePeerAccess(i, 0);
        }
    });

    m.def("is_gpu_supported", &isCuQuantumSupported,
          py::arg("device_number") = 0,
          "Checks if the given GPU device meets the minimum architecture "
          "support for the PennyLane-Lightning-Tensor device.");

    m.def("get_gpu_arch", &getGPUArch, py::arg("device_number") = 0,
          "Returns the given GPU major and minor GPU support.");
    py::class_<DevicePool<int>>(m, "DevPool")
        .def(py::init<>())
        .def("getActiveDevices", &DevicePool<int>::getActiveDevices)
        .def("isActive", &DevicePool<int>::isActive)
        .def("isInactive", &DevicePool<int>::isInactive)
        .def("acquireDevice", &DevicePool<int>::acquireDevice)
        .def("releaseDevice", &DevicePool<int>::releaseDevice)
        .def("syncDevice", &DevicePool<int>::syncDevice)
        .def_static("getTotalDevices", &DevicePool<int>::getTotalDevices)
        .def_static("getDeviceUIDs", &DevicePool<int>::getDeviceUIDs)
        .def_static("setDeviceID", &DevicePool<int>::setDeviceIdx);

    py::class_<DevTag<int>>(m, "DevTag")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init([](int device_id, void *stream_id) {
            // Note, streams must be handled externally for now.
            // Binding support provided through void* conversion to cudaStream_t
            return new DevTag<int>(device_id,
                                   static_cast<cudaStream_t>(stream_id));
        }))
        .def(py::init<const DevTag<int> &>())
        .def("getDeviceID", &DevTag<int>::getDeviceID)
        .def("getStreamID",
             [](DevTag<int> &dev_tag) {
                 // default stream points to nullptr, so just return void* as
                 // type
                 return static_cast<void *>(dev_tag.getStreamID());
             })
        .def("refresh", &DevTag<int>::refresh);
}

} // namespace Pennylane::LightningTensor::TNCuda
  /// @endcond
