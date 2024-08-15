// Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
 * Export C++ functions to Python using Pybind.
 */
#include "Bindings.hpp"
#ifdef _ENABLE_PLGPU_MPI
#include "BindingsMPI.hpp"
#endif
#include "pybind11/pybind11.h"

// Defining the module name.
#if defined(_ENABLE_PLQUBIT)
#define LIGHTNING_MODULE_NAME lightning_qubit_ops
#elif _ENABLE_PLKOKKOS == 1
#define LIGHTNING_MODULE_NAME lightning_kokkos_ops
#elif _ENABLE_PLGPU == 1
#define LIGHTNING_MODULE_NAME lightning_gpu_ops
#elif _ENABLE_PLTENSOR == 1
#define LIGHTNING_TENSOR_MODULE_NAME lightning_tensor_ops
#endif

/// @cond DEV
namespace {
using namespace Pennylane;
} // namespace
/// @endcond

#if defined(LIGHTNING_MODULE_NAME)
/**
 * @brief Add Lightning State-vector C++ classes, methods and functions to
 * Python module.
 */
PYBIND11_MODULE(
    LIGHTNING_MODULE_NAME, // NOLINT: No control over Pybind internals
    m) {
    // Suppress doxygen autogenerated signatures

    pybind11::options options;
    options.disable_function_signatures();

    // Register functionality for numpy array memory alignment:
    registerArrayAlignmentBindings(m);

    // Register bindings for general info:
    registerInfo(m);

    registerUtils(m);

    // Register bindings for backend-specific info:
    registerBackendSpecificInfo(m);

    registerLightningClassBindings<StateVectorBackends>(m);

#ifdef _ENABLE_PLGPU_MPI
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
PYBIND11_MODULE(
    LIGHTNING_TENSOR_MODULE_NAME, // NOLINT: No control over Pybind internals
    m) {
    // Suppress doxygen autogenerated signatures
    pybind11::options options;
    options.disable_function_signatures();

    registerUtils(m);

    // Register bindings for backend-specific info:
    registerBackendSpecificInfo(m);

    registerLightningTensorClassBindings<TensorNetBackends>(m);
}
#endif
