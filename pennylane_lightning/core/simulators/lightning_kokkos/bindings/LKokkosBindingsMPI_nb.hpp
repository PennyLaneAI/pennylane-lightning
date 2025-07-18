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
 * @file LKokkosBindingsMPI_nb.hpp
 * Defines lightning.kokkos specific MPI operations to export to Python using
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

#include "BindingsUtils_nb.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "GateOperation.hpp"
#include "Kokkos_Core.hpp"
#include "MPIManagerKokkos.hpp"
#include "MeasurementsKokkosMPI.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkosMPI.hpp"
#include "TypeList.hpp"
#include "Util.hpp" // exp2

namespace nb = nanobind;

namespace Pennylane::LightningKokkos::NanoBindings {

/**
 * @brief Define StateVector backends for lightning.kokkos MPI
 */
using StateVectorMPIBackends =
    Pennylane::Util::TypeList<StateVectorKokkosMPI<float>,
                              StateVectorKokkosMPI<double>, void>;

/**
 * @brief Register backend class specific bindings for MPI.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's state vector class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindingsMPI(PyClass &pyclass) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    using arr_c = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;

    // Register gates for state vector
    registerGatesForStateVector<StateVectorT>(pyclass);
    registerControlledGate<StateVectorT>(pyclass);

    pyclass
        .def(
            nb::init([](MPIManagerKokkos &mpi_manager, std::size_t num_qubits) {
                return new StateVectorT(mpi_manager, num_qubits);
            }))
        .def(nb::init([](MPIManagerKokkos &mpi_manager, std::size_t num_qubits,
                         const InitializationSettings &kokkos_args) {
            return new StateVectorT(mpi_manager, num_qubits, kokkos_args);
        }))
        .def(nb::init([](std::size_t num_qubits) {
            return new StateVectorT(num_qubits);
        }))
        .def(nb::init([](std::size_t num_qubits,
                         const InitializationSettings &kokkos_args) {
            return new StateVectorT(num_qubits, kokkos_args);
        }))
        .def("resetStateVector", &StateVectorT::resetStateVector)
        .def(
            "setBasisState",
            [](StateVectorT &sv, const std::vector<std::size_t> &state,
               const std::vector<std::size_t> &wires) {
                sv.setBasisState(state, wires);
            },
            "Set the state vector to a basis state.")
        .def(
            "setStateVector",
            [](StateVectorT &sv, const arr_c &state,
               const std::vector<std::size_t> &wires) {
                const auto buffer = state.request();
                sv.setStateVector(static_cast<const ComplexT *>(buffer.ptr),
                                  wires);
            },
            "Set the state vector to the data contained in `state`.")
        .def(
            "DeviceToHost",
            [](StateVectorT &device_sv, arr_c &host_sv) {
                auto buffer_info = host_sv.request();
                auto *data_ptr = static_cast<ComplexT *>(buffer_info.ptr);
                if (host_sv.size()) {
                    device_sv.DeviceToHost(data_ptr, host_sv.size());
                }
            },
            "Synchronize data from the Kokkos device to host.")
        .def("HostToDevice",
             nb::overload_cast<ComplexT *, std::size_t>(
                 &StateVectorT::HostToDevice),
             "Synchronize data from the host device to Kokkos.")
        .def(
            "HostToDevice",
            [](StateVectorT &device_sv, const arr_c &host_sv) {
                const auto buffer_info = host_sv.request();
                auto *data_ptr = static_cast<ComplexT *>(buffer_info.ptr);
                const auto length =
                    static_cast<std::size_t>(buffer_info.shape[0]);
                if (length) {
                    device_sv.HostToDevice(data_ptr, length);
                }
            },
            "Synchronize data from the host device to Kokkos.")
        .def(
            "apply",
            [](StateVectorT &sv, const std::string &str,
               const std::vector<std::size_t> &wires, bool inv,
               [[maybe_unused]] const std::vector<std::vector<PrecisionT>>
                   &params,
               [[maybe_unused]] const arr_c &gate_matrix) {
                const auto m_buffer = gate_matrix.request();
                std::vector<Kokkos::complex<PrecisionT>> conv_matrix;
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const Kokkos::complex<PrecisionT> *>(
                            m_buffer.ptr);
                    conv_matrix = std::vector<Kokkos::complex<PrecisionT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
                sv.applyOperation(str, wires, inv, std::vector<PrecisionT>{},
                                  conv_matrix);
            },
            "Apply a matrix operation.")
        .def(
            "applyPauliRot",
            [](StateVectorT &sv, const std::vector<std::size_t> &wires,
               const bool inverse, const std::vector<PrecisionT> &params,
               const std::string &word) {
                sv.applyPauliRot(wires, inverse, params, word);
            },
            "Apply a Pauli rotation.")
        .def("applyControlledMatrix", &applyControlledMatrix<StateVectorT>,
             "Apply controlled operation")
        .def("collapse", &StateVectorT::collapse,
             "Collapse the statevector onto the 0 or 1 branch of a given wire.")
        // MPI related functions
        .def(
            "getNumLocalWires",
            [](StateVectorT &sv) { return sv.getNumLocalWires(); },
            "Get number of local wires.")
        .def(
            "getNumGlobalWires",
            [](StateVectorT &sv) { return sv.getNumGlobalWires(); },
            "Get number of global wires.")
        .def(
            "swapGlobalLocalWires",
            [](StateVectorT &sv,
               const std::vector<std::size_t> &global_wires_to_swap,
               const std::vector<std::size_t> &local_wires_to_swap) {
                sv.swapGlobalLocalWires(global_wires_to_swap,
                                        local_wires_to_swap);
            },
            "Swap global and local wires - global_wire_to_swap must be in "
            "global_wires_ and local_wires_to_swap must be in local_wires_")
        .def(
            "getLocalBlockSize",
            [](StateVectorT &sv) { return sv.getLocalBlockSize(); },
            "Get Local Block Size, i.e. size of SV on a single rank.")
        .def(
            "resetIndices", [](StateVectorT &sv) { sv.resetIndices(); },
            "Reset indices including global_wires, local_wires_, and "
            "mpi_rank_to_global_index_map_.")
        .def(
            "reorderAllWires", [](StateVectorT &sv) { sv.reorderAllWires(); },
            "Reorder all wires so that global_wires_ = {0, 1, ...} and "
            "local_wires_ = {..., num_qubit-1}.");
}

/**
 * @brief Register backend specific measurements class functionalities for MPI.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificMeasurementsMPI(PyClass &pyclass) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    // Keep only Kokkos-specific measurement methods here
    // Common methods have been moved to registerBackendAgnosticMeasurementsMPI
}

/**
 * @brief Register backend specific observables for MPI.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendSpecificObservablesMPI(nb::module_ &m) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    const std::string bitsize =
        std::is_same_v<PrecisionT, float> ? "64" : "128";

    using arr_c = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;

    // Register only Kokkos-specific observables here
    // Common observables have been moved to registerBackendAgnosticObservables
}

/**
 * @brief Register backend specific adjoint Jacobian methods for MPI.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithmsMPI(nb::module_ &m) {
    // This function is intentionally left empty as there are no
    // backend-specific algorithms for Kokkos MPI
}

/**
 * @brief Register bindings for backend-specific info for MPI.
 *
 * @param m Nanobind module.
 */
void registerBackendSpecificInfoMPI(nb::module_ &m) {
    using np_arr_c64 = nb::ndarray<std::complex<float>, nb::c_contig>;
    using np_arr_c128 = nb::ndarray<std::complex<double>, nb::c_contig>;

    nb::class_<MPIManagerKokkos>(m, "MPIManagerKokkos")
        .def(nb::init<>())
        .def(nb::init<MPIManagerKokkos &>())
        .def("Barrier", &MPIManagerKokkos::Barrier)
        .def("getRank", &MPIManagerKokkos::getRank)
        .def("getSize", &MPIManagerKokkos::getSize)
        .def("getSizeNode", &MPIManagerKokkos::getSizeNode)
        .def("getTime", &MPIManagerKokkos::getTime)
        .def("getVendor", &MPIManagerKokkos::getVendor)
        .def("getVersion", &MPIManagerKokkos::getVersion)
        .def(
            "Scatter",
            [](MPIManagerKokkos &mpi_manager, np_arr_c64 &sendBuf,
               np_arr_c64 &recvBuf, int root) {
                auto send_ptr =
                    static_cast<std::complex<float> *>(sendBuf.request().ptr);
                auto recv_ptr =
                    static_cast<std::complex<float> *>(recvBuf.request().ptr);
                mpi_manager.template Scatter<std::complex<float>>(
                    send_ptr, recv_ptr, recvBuf.size(), root);
            },
            "MPI Scatter.")
        .def(
            "Scatter",
            [](MPIManagerKokkos &mpi_manager, np_arr_c128 &sendBuf,
               np_arr_c128 &recvBuf, int root) {
                auto send_ptr =
                    static_cast<std::complex<double> *>(sendBuf.request().ptr);
                auto recv_ptr =
                    static_cast<std::complex<double> *>(recvBuf.request().ptr);
                mpi_manager.template Scatter<std::complex<double>>(
                    send_ptr, recv_ptr, recvBuf.size(), root);
            },
            "MPI Scatter.");
}

} // namespace Pennylane::LightningKokkos::NanoBindings
