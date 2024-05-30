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
#pragma once
#include <set>
#include <tuple>
#include <variant>
#include <vector>

#include "cuda.h"

#include "BindingsLTensorBase.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "DevTag.hpp"
#include "DevicePool.hpp"
#include "Error.hpp"
#include "MPSTNCuda.hpp"
#include "MeasurementsTNCuda.hpp"
#include "ObservablesTNCuda.hpp"
#include "TypeList.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Bindings;
using namespace Pennylane::LightningTensor::TNCuda::Measures;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using Pennylane::LightningTensor::TNCuda::MPSTNCuda;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane::LightningTensor::TNCuda {
using StateTensorBackends =
    Pennylane::Util::TypeList<MPSTNCuda<float>, MPSTNCuda<double>, void>;

/**
 * @brief Get a gate kernel map for a statetensor.
 */

template <class StateTensorT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    registerGatesForStateTensor<StateTensorT>(pyclass);

    pyclass
        .def(py::init<const std::size_t,
                      const std::size_t>()) // num_qubits, max_bond_dim
        .def(py::init<const std::size_t, const std::size_t,
                      DevTag<int>>()) // num_qubits, max_bond_dim, dev-tag
        .def(
            "setBasisState",
            [](StateTensorT &state_tensor,
               std::vector<std::size_t> &basisState) {
                state_tensor.setBasisState(basisState);
            },
            "Create Basis State on GPU.")
        .def(
            "getFinalState",
            [](StateTensorT &state_tensor) { state_tensor.get_final_state(); },
            "Get the final state.")
        .def("GetNumGPUs", &getGPUCount, "Get the number of available GPUs.")
        .def("getCurrentGPU", &getGPUIdx,
             "Get the GPU index for the statevector data.")
        .def("numQubits", &StateTensorT::getNumQubits)
        .def("reset", &StateTensorT::reset, "Reset the statevector.")
        .def("maxBondDim", &StateTensorT::getMaxBondDim,
             "Get the the max bond dimension.");
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
          "support for the PennyLane-Lightning-GPU device.");

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
