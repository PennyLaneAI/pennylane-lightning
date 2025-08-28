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
 * Defines CUDA device-specific operations to export to Python, other
 * utility functions interfacing with Nanobind and support to agnostic bindings.
 */

#pragma once

#include "BindingsUtils.hpp"
#include "DevTag.hpp"
#include "DevicePool.hpp"
#include "cuda_helpers.hpp"

#include <nanobind/nanobind.h>

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::Util;
} // namespace
/// @endcond
namespace Pennylane::Util::NanoBindings {
namespace nb = nanobind;

/**
 * @brief Register bindings for CUDA utils.
 *
 * Register the device_reset and allToAllAccess functions to the given module.
 *
 * @param m Nanobind module.
 */
void registerCudaUtils(nb::module_ &m) {
    /* device_reset function */
    m.def("device_reset", &deviceReset, "Reset all GPU devices and contexts.");

    /* allToAllAccess function */
    m.def("allToAllAccess", []() {
        for (int i = 0; i < static_cast<int>(getGPUCount()); i++) {
            cudaDeviceEnablePeerAccess(i, 0);
        }
    });

    /* is_gpu_supported function */
    m.def("is_gpu_supported", &isCuQuantumSupported,
          nb::arg("device_number") = 0,
          "Checks if the given GPU device meets the minimum architecture "
          "support for the PennyLane-Lightning-GPU device.");

    /* get_gpu_arch function */
    m.def("get_gpu_arch", &getGPUArch, nb::arg("device_number") = 0,
          "Returns the given GPU major and minor GPU support.");

    /* DevicePool class */
    auto pyclass_devpool = nb::class_<DevicePool<int>>(m, "DevPool");
    pyclass_devpool.def(nb::init<>());
    pyclass_devpool.def("getActiveDevices", &DevicePool<int>::getActiveDevices);
    pyclass_devpool.def("isActive", &DevicePool<int>::isActive);
    pyclass_devpool.def("isInactive", &DevicePool<int>::isInactive);
    pyclass_devpool.def("acquireDevice", &DevicePool<int>::acquireDevice);
    pyclass_devpool.def("releaseDevice", &DevicePool<int>::releaseDevice);
    pyclass_devpool.def("syncDevice", &DevicePool<int>::syncDevice);
    pyclass_devpool.def("refresh", &DevicePool<int>::refresh);
    pyclass_devpool.def_static("getTotalDevices",
                               &DevicePool<int>::getTotalDevices);
    pyclass_devpool.def_static("getDeviceUIDs",
                               &DevicePool<int>::getDeviceUIDs);
    pyclass_devpool.def_static("setDeviceID", &DevicePool<int>::setDeviceIdx);
    pyclass_devpool.def(
        "__getstate__",
        []([[maybe_unused]] const DevicePool<int> &self) { // __getstate__
            return nb::make_tuple();
        });
    pyclass_devpool.def("__setstate__", [](DevicePool<int> &self,
                                           nb::tuple &t) { // __setstate__
        if (t.size() != 0) {
            throw std::runtime_error("Invalid state!");
        }

        new (&self) DevicePool<int>(); // Reconstruct the object
    });

    /* DevTag class */
    auto pyclass_devtag = nb::class_<DevTag<int>>(m, "DevTag");
    pyclass_devtag.def(nb::init<>());
    pyclass_devtag.def(nb::init<int>());
    pyclass_devtag.def(nb::init<const DevTag<int> &>());
    pyclass_devtag.def("__init__", [](int device_id, void *stream_id) {
        // The lower level `__init__` needs to be defined directly to
        // support type casting
        return DevTag<int>(device_id, static_cast<cudaStream_t>(stream_id));
    });
    pyclass_devtag.def("getDeviceID", &DevTag<int>::getDeviceID);
    pyclass_devtag.def("getStreamID", [](DevTag<int> &dev_tag) {
        // default stream points to nullptr, so just return void* as
        // type
        return static_cast<void *>(dev_tag.getStreamID());
    });
    pyclass_devtag.def("refresh", &DevTag<int>::refresh);
}
} // namespace Pennylane::Util::NanoBindings
