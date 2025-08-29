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

#include "cuda.h"

#include "Bindings.hpp"
#include "BindingsUtils.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "DevTag.hpp"
#include "DevicePool.hpp"
#include "Error.hpp"
#include "MPIManagerGPU.hpp"
#include "MeasurementsGPUMPI.hpp"
#include "ObservablesGPUMPI.hpp"
#include "StateVectorCudaMPI.hpp"
#include "TypeList.hpp"
#include "Util.hpp" // exp2
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::NanoBindings;
using namespace Pennylane::LightningGPU::Algorithms;
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Observables;
using Pennylane::LightningGPU::StateVectorCudaMPI;
using Pennylane::LightningGPU::Util::MPIManagerGPU;
using Pennylane::Util::exp2;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::NanoBindings {

namespace nb = nanobind;
/// @cond DEV
using StateVectorMPIBackends =
    Pennylane::Util::TypeList<StateVectorCudaMPI<float>,
                              StateVectorCudaMPI<double>, void>;
/// @endcond

/**
 * @brief Register backend specific state vector methods for MPI.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's state vector class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificStateVectorMethodsMPI(PyClass &pyclass) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    using ArrayComplexT = nb::ndarray<ComplexT, nb::c_contig>;

    // Register gates for state vector
    registerGates<StateVectorT>(pyclass);

    pyclass.def(nb::init<MPIManagerGPU &, const DevTag<int> &, std::size_t,
                         std::size_t, std::size_t>(),
                "Constructor with MPI manager, device tag, buffer size, global "
                "qubits, local qubits");
    pyclass.def(
        nb::init<const DevTag<int> &, std::size_t, std::size_t, std::size_t>(),
        "Constructor with device tag, buffer size, global qubits, local "
        "qubits");
    pyclass.def(
        "setBasisState",
        [](StateVectorT &sv, const std::vector<std::size_t> &state,
           const std::vector<std::size_t> &wires,
           const bool use_async) { sv.setBasisState(state, wires, use_async); },
        nb::arg("state") = nullptr, nb::arg("wires") = nullptr,
        nb::arg("async") = false,
        "Set the state vector to a basis state on GPU.");
    pyclass.def(
        "setStateVector",
        [](StateVectorT &sv, const ArrayComplexT &state,
           const std::vector<std::size_t> &wires, const bool async) {
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
        "apply",
        [](StateVectorT &sv, const std::string &str,
           const std::vector<std::size_t> &wires, bool inv,
           const std::vector<std::vector<PrecisionT>> &params,
           const ArrayComplexT &gate_matrix) {
            std::vector<ComplexT> conv_matrix;
            if (gate_matrix.size() > 0) {
                conv_matrix = std::vector<ComplexT>{gate_matrix.data(),
                                                    gate_matrix.data() +
                                                        gate_matrix.size()};
            }

            if (params.empty()) {
                sv.applyOperation(str, wires, inv, std::vector<PrecisionT>{},
                                  conv_matrix);
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
                sv.applyOperation(str, wires, inv, params[0], conv_matrix);
            }
        },
        "Apply operation via the gate matrix");
}

/**
 * @brief Register backend specific measurements class functionalities.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificMeasurementsMPI(PyClass &pyclass) {
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
        [](MeasurementsMPI<StateVectorT> &M, const std::string &operation,
           const std::vector<std::size_t> &wires) {
            return M.expval(operation, wires);
        },
        "Expected value of an operation by name.");
    pyclass.def(
        "expval",
        [](MeasurementsMPI<StateVectorT> &M, const ArrayIndexT &row_map,
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
        [](MeasurementsMPI<StateVectorT> &M,
           const std::vector<std::string> &pauli_words,
           const std::vector<std::vector<std::size_t>> &target_wires,
           const ArrayComplexT &coeffs) {
            return M.expval(pauli_words, target_wires, coeffs.data());
        },
        "Expected value of Hamiltonian represented by Pauli words.");
    pyclass.def(
        "expval",
        [](MeasurementsMPI<StateVectorT> &M,
           const std::vector<std::string> &pauli_words,
           const std::vector<std::vector<std::size_t>> &target_wires,
           const std::vector<ComplexT> &coeffs) {
            // Required to be able to accept `coeffs` as a python tuple
            return M.expval(pauli_words, target_wires, coeffs.data());
        },
        "Expected value of Hamiltonian represented by Pauli words.");
    pyclass.def(
        "expval",
        [](MeasurementsMPI<StateVectorT> &M, const ArrayComplexT &matrix,
           const std::vector<std::size_t> &wires) {
            const std::size_t matrix_size = exp2(2 * wires.size());
            std::vector<ComplexT> matrix_v{matrix.data(),
                                           matrix.data() + matrix_size};
            return M.expval(matrix_v, wires);
        },
        "Expected value of a Hermitian observable.");
    pyclass.def("var", [](MeasurementsMPI<StateVectorT> &M,
                          const std::string &operation,
                          const std::vector<std::size_t> &wires) {
        return M.var(operation, wires);
    });
    pyclass.def("var",
                static_cast<PrecisionT (MeasurementsMPI<StateVectorT>::*)(
                    const std::string &, const std::vector<std::size_t> &)>(
                    &MeasurementsMPI<StateVectorT>::var),
                "Variance of an operation by name.");
    pyclass.def(
        "var",
        [](MeasurementsMPI<StateVectorT> &M, const ArrayIndexT &row_map,
           const ArrayIndexT &entries, const ArrayComplexT &values) {
            return M.var(row_map.data(), static_cast<int64_t>(row_map.size()),
                         entries.data(), values.data(),
                         static_cast<int64_t>(values.size()));
        },
        "Variance of a sparse Hamiltonian.");
}

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithmsMPI(nb::module_ &m) {}

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Nanobind module.
 */
void registerBackendSpecificInfoMPI(nb::module_ &m) {
    using ArrayComplex64T = nb::ndarray<std::complex<float>, nb::c_contig>;
    using ArrayComplex128T = nb::ndarray<std::complex<double>, nb::c_contig>;

    auto mpi_manager_class = nb::class_<MPIManagerGPU>(m, "MPIManagerGPU");
    mpi_manager_class.def(nb::init<>());
    mpi_manager_class.def(nb::init<MPIManagerGPU &>());
    mpi_manager_class.def("Barrier", &MPIManagerGPU::Barrier);
    mpi_manager_class.def("getRank", &MPIManagerGPU::getRank);
    mpi_manager_class.def("getSize", &MPIManagerGPU::getSize);
    mpi_manager_class.def("getSizeNode", &MPIManagerGPU::getSizeNode);
    mpi_manager_class.def("getTime", &MPIManagerGPU::getTime);
    mpi_manager_class.def("getVendor", &MPIManagerGPU::getVendor);
    mpi_manager_class.def("getVersion", &MPIManagerGPU::getVersion);
    mpi_manager_class.def(
        "Scatter",
        [](MPIManagerGPU &mpi_manager, ArrayComplex64T &sendBuf,
           ArrayComplex64T &recvBuf, int root) {
            auto send_ptr = static_cast<std::complex<float> *>(sendBuf.data());
            auto recv_ptr = static_cast<std::complex<float> *>(recvBuf.data());
            mpi_manager.template Scatter<std::complex<float>>(
                send_ptr, recv_ptr, static_cast<std::size_t>(recvBuf.size()),
                root);
        },
        "MPI Scatter for complex float arrays.");
    mpi_manager_class.def(
        "Scatter",
        [](MPIManagerGPU &mpi_manager, ArrayComplex128T &sendBuf,
           ArrayComplex128T &recvBuf, int root) {
            auto send_ptr = static_cast<std::complex<double> *>(sendBuf.data());
            auto recv_ptr = static_cast<std::complex<double> *>(recvBuf.data());
            mpi_manager.template Scatter<std::complex<double>>(
                send_ptr, recv_ptr, static_cast<std::size_t>(recvBuf.size()),
                root);
        },
        "MPI Scatter for complex double arrays.");
}
} // namespace Pennylane::LightningGPU::NanoBindings
/// @endcond
