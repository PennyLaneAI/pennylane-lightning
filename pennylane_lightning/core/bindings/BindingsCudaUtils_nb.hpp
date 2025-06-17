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
 * @file BindingsCudaUtils.hpp
 * Defines CUDA device - specific operations to export to Python, other
 * utility functions interfacing with Nanobind and support to agnostic bindings.
 */

#pragma once

#include "DevTag.hpp"
#include "DevicePool.hpp"
#include "cuda_helpers.hpp"

namespace nb = nanobind;

namespace Pennylane::LightningGPU::Util {
/**
 * @brief Register bindings for CUDA utils.
 *
 * @param m Nanobind module.
 */
void registerCudaUtils(nb::module_ &m) {
    m.def("device_reset", &deviceReset, "Reset all GPU devices and contexts.");
    m.def("allToAllAccess", []() {
        for (int i = 0; i < static_cast<int>(getGPUCount()); i++) {
            cudaDeviceEnablePeerAccess(i, 0);
        }
    });

    m.def("is_gpu_supported", &isCuQuantumSupported,
          nb::arg("device_number") = 0,
          "Checks if the given GPU device meets the minimum architecture "
          "support for the PennyLane-Lightning-GPU device.");

    m.def("get_gpu_arch", &getGPUArch, nb::arg("device_number") = 0,
          "Returns the given GPU major and minor GPU support.");
    nb::class_<DevicePool<int>>(m, "DevPool")
        .def(nb::init<>())
        .def("getActiveDevices", &DevicePool<int>::getActiveDevices)
        .def("isActive", &DevicePool<int>::isActive)
        .def("isInactive", &DevicePool<int>::isInactive)
        .def("acquireDevice", &DevicePool<int>::acquireDevice)
        .def("releaseDevice", &DevicePool<int>::releaseDevice)
        .def("syncDevice", &DevicePool<int>::syncDevice)
        .def("refresh", &DevicePool<int>::refresh)
        .def_static("getTotalDevices", &DevicePool<int>::getTotalDevices)
        .def_static("getDeviceUIDs", &DevicePool<int>::getDeviceUIDs)
        .def_static("setDeviceID", &DevicePool<int>::setDeviceIdx)
        .def("__getstate__",
             []([[maybe_unused]] const DevicePool<int> &self) { // __getstate__
                 return nb::make_tuple();
             })
        .def("__setstate__",
             [](DevicePool<int> &self, nb::tuple &t) { // __setstate__
                 if (t.size() != 0) {
                     throw std::runtime_error("Invalid state!");
                 }

                 self.refresh();
             });

    nb::class_<DevTag<int>>(m, "DevTag")
        .def(nb::init<>())
        .def(nb::init<int>())
        .def(nb::init<const DevTag<int> &>())
        .def("__init__",
             [](int device_id, void *stream_id) {
                 // The lower level `__init__` needs to be defined directly to
                 // support type casting
                 return DevTag<int>(device_id,
                                    static_cast<cudaStream_t>(stream_id));
             })
        .def("getDeviceID", &DevTag<int>::getDeviceID)
        .def("getStreamID",
             [](DevTag<int> &dev_tag) {
                 // default stream points to nullptr, so just return void* as
                 // type
                 return static_cast<void *>(dev_tag.getStreamID());
             })
        .def("refresh", &DevTag<int>::refresh);
}

} // namespace Pennylane::LightningGPU::Util
