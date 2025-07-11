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
 * @file LQubitBindings_nb.hpp
 * Defines lightning.qubit specific operations to export to Python using
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

#include "StateVectorLQubitManaged.hpp"
#include "TypeList.hpp"

namespace nb = nanobind;

namespace Pennylane::LightningQubit::NanoBindings {

/**
 * @brief Define StateVector backends for lightning.qubit
 */
using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorLQubitManaged<float>,
                              StateVectorLQubitManaged<double>, void>;

/**
 * @brief Update state vector data from an array
 *
 * This function accepts any array-like object that follows the buffer protocol,
 * including NumPy arrays and JAX arrays (for example).
 *
 * Example with JAX:
 * ```python
 * import jax.numpy as jnp
 * import pennylane_lightning.lightning_qubit_nb as plq
 *
 * # Create a JAX array
 * jax_data = jnp.zeros(2**3, dtype=jnp.complex64)
 * jax_data = jax_data.at[0].set(1.0)  # Set to |000⟩ state
 *
 * # Create a state vector and update with JAX data
 * sv = plq.StateVectorC64(3)  # 3 qubits
 * sv.updateData(jax_data)     # Works with JAX arrays!
 * ```
 *
 * @tparam StateVectorT State vector type
 * @param sv State vector to update
 * @param data Array with new data
 */
template <class StateVectorT>
void updateStateVectorData(
    StateVectorT &sv,
    const nb::ndarray<const typename StateVectorT::ComplexT, nb::c_contig>
        &data) {
    using ComplexT = typename StateVectorT::ComplexT;

    // Check dimensions
    if (data.ndim() != 1) {
        throw std::invalid_argument("Array must be 1-dimensional");
    }

    // Get data pointer and size
    const ComplexT *data_ptr = data.data();
    std::size_t size = data.shape(0);

    // Update the state vector data
    sv.updateData(data_ptr, size);
}

/**
 * @brief Get a controlled matrix and kernel map for a statevector.
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &) {} // pyclass

/**
 * @brief Register backend specific state vector methods.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's state vector class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificStateVectorMethods(PyClass &pyclass) {
    // using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    // using ParamT = PrecisionT; // Parameter's data precision - unused for now

    // Add other methods (resetStateVector, setBasisState, etc.)
    pyclass.def("resetStateVector", &StateVectorT::resetStateVector,
                "Reset the state vector to |0...0>.");

    pyclass.def("updateData", &updateStateVectorData<StateVectorT>,
                "Update the state vector data from an array.",
                nb::arg("state"));
    pyclass.def(
        "setBasisState",
        [](StateVectorT &sv, const std::vector<std::size_t> &state,
           const std::vector<std::size_t> &wires) {
            sv.setBasisState(state, wires);
        },
        "Set the state vector to a basis state.", nb::arg("state"),
        nb::arg("wires"));

    pyclass.def(
        "setStateVector",
        [](StateVectorT &sv, const nb::ndarray<ComplexT, nb::c_contig> &state,
           const std::vector<std::size_t> &wires) {
            // Get data pointer directly from ndarray
            const ComplexT *data_ptr =
                static_cast<const ComplexT *>(state.data());
            sv.setStateVector(data_ptr, wires);
        },
        "Set the state vector to the data contained in `state`.",
        nb::arg("state"), nb::arg("wires"));

    pyclass.def(
        "getState",
        [](const StateVectorT &sv, nb::ndarray<ComplexT, nb::c_contig> &state) {
            // Check if array is large enough
            if (state.shape(0) < sv.getLength()) {
                throw std::invalid_argument("Output array is too small");
            }

            // Copy data to numpy array
            ComplexT *data_ptr = static_cast<ComplexT *>(state.data());
            std::copy(sv.getData(), sv.getData() + sv.getLength(), data_ptr);
        },
        "Copy StateVector data into a Numpy array.", nb::arg("state"));
}

/**
 * @brief Register backend specific measurements class functionalities.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificMeasurements(PyClass &) {} // pyclass

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
 * @brief Register bindings for backend-specific info.
 *
 * @param m Nanobind module.
 */
void registerBackendSpecificInfo(nb::module_ &) {} // m

} // namespace Pennylane::LightningQubit::NanoBindings
