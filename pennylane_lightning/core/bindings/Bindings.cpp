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
 * @file Bindings.cpp
 * Implements device-agnostic operations to export to Python using Nanobind.
 */

#include "Bindings.hpp"
#ifdef _ENABLE_MPI
#include "BindingsMPI.hpp"
#endif
#include "CPUMemoryModel.hpp"
#include "Memory.hpp"

namespace Pennylane::NanoBindings {

namespace nb = nanobind;

#if defined(LIGHTNING_MODULE_NAME)

/**
 * @brief Add Lightning State-vector C++ classes, methods and functions to
 * Python module using Nanobind.
 */
NB_MODULE(LIGHTNING_MODULE_NAME, m) {
#ifdef NDEBUG
    // Disable leak warnings in release mode. These are often false positives
    // caused by 3rd party libraries
    // https://nanobind.readthedocs.io/en/latest/refleaks.html#additional-sources-of-leaks
    nb::set_leak_warnings(false);
#endif

    // Register array alignment functionality
    registerArrayAlignmentBindings(m);

    // Register general info
    registerInfo(m);

    // Register backend-specific info
    registerBackendSpecificInfo(m);

    // Register lightning class bindings
    registerLightningClassBindings<StateVectorBackends>(m);
#ifdef _ENABLE_MPI
    registerInfoMPI(m);
    registerBackendSpecificInfoMPI(m);
    registerLightningClassBindingsMPI<StateVectorMPIBackends>(m);
#endif
}
#endif

#if defined(LIGHTNING_TENSOR_MODULE_NAME)
/**
 * @brief Add LightningTensor C++ classes, methods and functions to Python
 * module.
 */
NB_MODULE(LIGHTNING_TENSOR_MODULE_NAME, m) {
#ifdef NDEBUG
    // Disable leak warnings in release mode. These are often false positives
    // caused by 3rd party libraries
    // https://nanobind.readthedocs.io/en/latest/refleaks.html#additional-sources-of-leaks
    nb::set_leak_warnings(false);
#endif

    // Register general info
    registerInfo(m);

    // Register bindings for backend-specific info:
    registerBackendSpecificInfo(m);

    registerLightningClassBindings<TensorNetworkBackends>(m);
}
#endif
} // namespace Pennylane::NanoBindings
