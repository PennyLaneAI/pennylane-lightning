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
 * @file Bindings_nb.cpp
 * Implements device-agnostic operations to export to Python using Nanobind.
 */

#include "Bindings_nb.hpp"
#include "CPUMemoryModel.hpp"
#include "Memory.hpp"

namespace Pennylane::NanoBindings {

#if defined(LIGHTNING_MODULE_NAME)
/**
 * @brief Add Lightning State-vector C++ classes, methods and functions to
 * Python module using Nanobind.
 */
NB_MODULE(LIGHTNING_MODULE_NAME, m) {
    // Register array alignment functionality
    registerArrayAlignmentBindings(m);

    // Register general info
    registerInfo(m);

    // Register backend-specific info
    registerBackendSpecificInfo(m);

    // Register lightning class bindings
    registerLightningClassBindings<StateVectorBackends>(m);
#ifdef _ENABLE_PLGPU_MPI
    // registerLightningClassBindingsMPI<StateVectorMPIBackends>(m);
#endif
    // Register exception types - using nanobind's approach
    nb::exception<Pennylane::Util::LightningException>(m, "LightningException");
}
#endif

#if defined(LIGHTNING_TENSOR_MODULE_NAME)
/**
 * @brief Add LightningTensor C++ classes, methods and functions to Python
 * module.
 */
NB_MODULE(LIGHTNING_TENSOR_MODULE_NAME, m) {
    // Register bindings for backend-specific info:
    registerBackendSpecificInfo(m);

    registerLightningTensorClassBindings<TensorNetworkBackends>(m);
}
#endif
} // namespace Pennylane::NanoBindings
