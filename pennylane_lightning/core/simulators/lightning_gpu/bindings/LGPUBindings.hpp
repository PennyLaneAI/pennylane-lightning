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
 * @file LGPUBindings.hpp
 * Defines lightning.gpu specific operations to export to Python using Nanobind.
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

#include "MeasurementsGPU.hpp"
#include "StateVectorCudaManaged.hpp"
#include "TypeList.hpp"

namespace nb = nanobind;

/// @cond DEV
namespace {
using namespace Pennylane::Util::NanoBindings;
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Observables;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::NanoBindings {

///
/**
 * @brief Define StateVector backends for lightning.gpu
 */
using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorCudaManaged<float>,
                              StateVectorCudaManaged<double>, void>;
/// @endcond

/**
 * @brief Register backend specific measurements class functionalities.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificMeasurements(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type

    using ArrayComplexT = nb::ndarray<ComplexT, nb::c_contig>;
    using IndexT =
        typename std::conditional<std::is_same<PrecisionT, float>::value,
                                  int32_t, int64_t>::type;
    using ArrayIndexT = nb::ndarray<IndexT, nb::c_contig>;

    pyclass.def(
        "expval",
        [](Measurements<StateVectorT> &M, const std::string &operation,
           const std::vector<std::size_t> &wires) {
            return M.expval(operation, wires);
        },
        "Expected value of an operation by name.");
    pyclass.def(
        "expval",
        [](Measurements<StateVectorT> &M, const ArrayIndexT &row_map,
           const ArrayIndexT &entries, const ArrayComplexT &values) {
            return M.expval(
                row_map.data(),
                static_cast<int64_t>(
                    row_map.size()), // int64_t is required by cusparse
                entries.data(), values.data(),
                static_cast<int64_t>(
                    values.size())); // int64_t is required by cusparse
        },
        "Expected value of a sparse Hamiltonian.");
    pyclass.def(
        "expval",
        [](Measurements<StateVectorT> &M,
           const std::vector<std::string> &pauli_words,
           const std::vector<std::vector<std::size_t>> &target_wires,
           const ArrayComplexT &coeffs) {
            return M.expval(pauli_words, target_wires, coeffs.data());
        },
        "Expected value of Hamiltonian represented by Pauli words.");
    pyclass.def(
        "expval",
        [](Measurements<StateVectorT> &M,
           const std::vector<std::string> &pauli_words,
           const std::vector<std::vector<std::size_t>> &target_wires,
           const std::vector<ComplexT> &coeffs) {
            // Required to be able to accept `coeffs` as a python tuple
            return M.expval(pauli_words, target_wires, coeffs.data());
        },
        "Expected value of Hamiltonian represented by Pauli words.");
    pyclass.def(
        "expval",
        [](Measurements<StateVectorT> &M, const ArrayComplexT &matrix,
           const std::vector<std::size_t> &wires) {
            const std::size_t matrix_size = exp2(2 * wires.size());
            std::vector<ComplexT> matrix_v{matrix.data(),
                                           matrix.data() + matrix_size};
            return M.expval(matrix_v, wires);
        },
        "Expected value of a Hermitian observable.");
    pyclass.def("var",
                [](Measurements<StateVectorT> &M, const std::string &operation,
                   const std::vector<std::size_t> &wires) {
                    return M.var(operation, wires);
                });
    pyclass.def("var",
                static_cast<PrecisionT (Measurements<StateVectorT>::*)(
                    const std::string &, const std::vector<std::size_t> &)>(
                    &Measurements<StateVectorT>::var),
                "Variance of an operation by name.");
    pyclass.def(
        "var",
        [](Measurements<StateVectorT> &M, const ArrayIndexT &row_map,
           const ArrayIndexT &entries, const ArrayComplexT &values) {
            return M.var(row_map.data(), static_cast<int64_t>(row_map.size()),
                         entries.data(), values.data(),
                         static_cast<int64_t>(values.size()));
        },
        "Variance of a sparse Hamiltonian.");
} // pyclass

/**
 * @brief Register backend specific observables.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendSpecificObservables(nb::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type.
    using ArrayComplexT = nb::ndarray<ComplexT, nb::c_contig>;

    const std::string bitsize = std::to_string(sizeof(ComplexT) * 8);
    std::string class_name;

    class_name = "SparseHamiltonianC" + bitsize;
    using IndexT = typename SparseHamiltonian<StateVectorT>::IdxT;
    using ArrayIndexT = nb::ndarray<IndexT, nb::c_contig>;

    auto pyclass =
        nb::class_<SparseHamiltonian<StateVectorT>, Observable<StateVectorT>>(
            m, class_name.c_str());
    pyclass.def("__init__", [](SparseHamiltonian<StateVectorT> *self,
                               const ArrayComplexT &data,
                               const ArrayIndexT &indices,
                               const ArrayIndexT &offsets,
                               const std::vector<std::size_t> &wires) {
        new (self) SparseHamiltonian<StateVectorT>{
            std::vector<ComplexT>({data.data(), data.data() + data.size()}),
            std::vector<IndexT>(
                {indices.data(), indices.data() + indices.size()}),
            std::vector<IndexT>(
                {offsets.data(), offsets.data() + offsets.size()}),
            wires};
    });
    pyclass.def("__repr__", &SparseHamiltonian<StateVectorT>::getObsName);
    pyclass.def("get_wires", &SparseHamiltonian<StateVectorT>::getWires,
                "Get wires of observables");
    pyclass.def(
        "__eq__",
        []([[maybe_unused]] const SparseHamiltonian<StateVectorT> &self,
           nb::handle other) -> bool {
            if (!nb::isinstance<SparseHamiltonian<StateVectorT>>(other)) {
                return false;
            }
            auto other_cast = nb::cast<SparseHamiltonian<StateVectorT>>(other);
            return self == other_cast;
        },
        "Compare two observables");
} // m

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithms(nb::module_ &) {}

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

            info["NAME"] = "lightning.gpu";

            return info;
        },
        "Backend-specific information.");
    registerCudaUtils(m);
} // m

/**
 * @brief Update state vector data from an array
 *
 * This function accepts any array-like object that follows the buffer protocol,
 * including NumPy arrays and JAX arrays (for example).
 *
 * Example with JAX:
 * ```python
 * import jax.numpy as jnp
 * import pennylane_lightning.lightning_qubit_ops as plq
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
    const nb::ndarray<typename StateVectorT::ComplexT, nb::c_contig> &data) {
    using ComplexT = typename StateVectorT::ComplexT;

    // Check dimensions
    if (data.ndim() != 1) {
        throw std::invalid_argument("Array must be 1-dimensional");
    }

    // Get data pointer and size
    const ComplexT *data_ptr = static_cast<const ComplexT *>(data.data());
    std::size_t size = data.shape(0);

    // Update the state vector data
    sv.updateData(data_ptr, size);
}

/**
 * @brief Register backend specific state vector methods.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's state vector class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificStateVectorMethods(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT = typename StateVectorT::ComplexT;
    using ArrayComplexT = nb::ndarray<ComplexT, nb::c_contig>;

    pyclass.def(nb::init<std::size_t>());              // qubits, device
    pyclass.def(nb::init<std::size_t, DevTag<int>>()); // qubits, dev-tag
    pyclass.def("__init__", [](PyClass *self, const ArrayComplexT &arr) {
        new (self) StateVectorT(arr.data(), arr.size());
    });
    pyclass.def("updateData", &updateStateVectorData<StateVectorT>,
                "Update the state vector data from an array.",
                nb::arg("state"));
    pyclass.def(
        "setBasisState",
        [](StateVectorT &sv, const std::vector<std::size_t> &state,
           const std::vector<std::size_t> &wires,
           const bool async) { sv.setBasisState(state, wires, async); },
        nb::arg("state") = nullptr, nb::arg("wires") = nullptr,
        nb::arg("async") = false,
        "Set the state vector to a basis state on GPU.");
    pyclass.def(
        "setStateVector",
        [](StateVectorT &sv, const ArrayComplexT &state,
           const std::vector<std::size_t> &wires, const bool async = false) {
            sv.setStateVector(state.data(), state.size(), wires, async);
        },
        nb::arg("state"), nb::arg("wires"), nb::arg("async") = false,
        "Set State Vector on GPU with values for the state vector and "
        "wires on the host memory.");
    pyclass.def(
        "DeviceToDevice",
        [](StateVectorT &sv, const StateVectorT &other, bool async) {
            sv.updateData(other, async);
        },
        "Synchronize data from another GPU device to current device.");
    pyclass.def("DeviceToHost",
                nb::overload_cast<ComplexT *, std::size_t, bool>(
                    &StateVectorT::CopyGpuDataToHost, nb::const_),
                "Synchronize data from the GPU device to host.");
    pyclass.def(
        "DeviceToHost",
        [](const StateVectorT &gpu_sv, ArrayComplexT &cpu_sv, bool) {
            if (cpu_sv.size()) {
                gpu_sv.CopyGpuDataToHost(cpu_sv.data(), cpu_sv.size());
            }
        },
        "Synchronize data from the GPU device to host.");
    pyclass.def("HostToDevice",
                nb::overload_cast<const ComplexT *, std::size_t, bool>(
                    &StateVectorT::CopyHostDataToGpu),
                "Synchronize data from the host device to GPU.");
    pyclass.def(
        "HostToDevice",
        [](StateVectorT &gpu_sv, const ArrayComplexT &cpu_sv, bool async) {
            if (cpu_sv.size()) {
                gpu_sv.CopyHostDataToGpu(cpu_sv.data(), cpu_sv.size(), async);
            }
        },
        "Synchronize data from the host device to GPU.");
    pyclass.def("GetNumGPUs", &getGPUCount,
                "Get the number of available GPUs.");
    pyclass.def("getCurrentGPU", &getGPUIdx,
                "Get the GPU index for the statevector data.");
    pyclass.def("numQubits", &StateVectorT::getNumQubits);
    pyclass.def("dataLength", &StateVectorT::getLength);
    pyclass.def(
        "resetStateVector",
        [](StateVectorT &gpu_sv, bool async) {
            gpu_sv.resetStateVector(async);
        },
        nb::arg("async") = false,
        "Initialize the statevector data to the |0...0> state");
    pyclass.def(
        "collapse", &StateVectorT::collapse,
        "Collapse the statevector onto the 0 or 1 branch of a given wire.");
    pyclass.def(
        "apply",
        [](StateVectorT &sv, const std::string &gate_name,
           const std::vector<std::size_t> &controlled_wires,
           const std::vector<bool> &controlled_values,
           const std::vector<std::size_t> &wires, bool inverse,
           const std::vector<PrecisionT> &params) {
            sv.applyOperation(gate_name, controlled_wires, controlled_values,
                              wires, inverse, params);
        },
        "Apply operation via the gate matrix");
    pyclass.def(
        "apply",
        [](StateVectorT &sv, const std::string &str,
           const std::vector<std::size_t> &wires, bool inv,
           const std::vector<std::vector<PrecisionT>> &params,
           const ArrayComplexT &gate_matrix) {
            if (params.empty()) {
                sv.applyOperation(str, wires, inv, std::vector<PrecisionT>{},
                                  gate_matrix.data(), gate_matrix.size());
            } else {
                PL_ABORT_IF(
                    params.size() != 1,
                    "Invalid parameter structure for gate operation with "
                    "custom matrix. "
                    "Expected exactly one parameter list (List[List[float]] "
                    "with size=1), "
                    "but received " +
                        std::to_string(params.size()) +
                        " parameter lists. "
                        "Each gate operation should provide its parameters as "
                        "a single nested list, e.g., [[param1, param2, ...]].");
                sv.applyOperation(str, wires, inv, params[0],
                                  gate_matrix.data(), gate_matrix.size());
            }
        },
        "Apply operation via the gate matrix");
    pyclass.def(
        "getState",
        [](const StateVectorT &sv, ArrayComplexT &state) {
            sv.CopyGpuDataToHost(state.data(), state.size());
        },
        "Copy state vector data to a numpy array.", nb::arg("state"));
} // registerBackendSpecificStateVectorMethods

/**
 * @brief Get a controlled matrix and kernel map for a statevector.
 *
 * @tparam StateVectorT The type of the state vector.
 * @tparam PyClass Nanobind's class to bind methods.
 * @param pyclass Nanobind's statevector class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    registerBackendSpecificStateVectorMethods<StateVectorT>(pyclass);
}

} // namespace Pennylane::LightningGPU::NanoBindings
