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
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "BindingsBase.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "GateOperation.hpp"
#include "MPIManagerKokkos.hpp"
#include "MeasurementsKokkosMPI.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkosMPI.hpp"
#include "TypeList.hpp"
#include "Util.hpp" // exp2

/// @cond DEV
namespace {
using namespace Pennylane::Bindings;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using Kokkos::InitializationSettings;
using Pennylane::LightningKokkos::StateVectorKokkosMPI;
using Pennylane::Util::exp2;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane::LightningKokkos {
/// @cond DEV
using StateVectorMPIBackends =
    Pennylane::Util::TypeList<StateVectorKokkosMPI<float>,
                              StateVectorKokkosMPI<double>, void>;
/// @endcond

/**
 * @brief Get a gate kernel map for a statevector.
 */
template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindingsMPI(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT = typename StateVectorT::ComplexT;
    using ParamT = PrecisionT; // Parameter's data precision
    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;
    registerGatesForStateVector<StateVectorT>(pyclass);
    registerControlledGate<StateVectorT>(pyclass);
    pyclass
        .def(
            py::init([](MPIManagerKokkos &mpi_manager, std::size_t num_qubits) {
                return new StateVectorT(mpi_manager, num_qubits);
            }))
        .def(py::init([](MPIManagerKokkos &mpi_manager, std::size_t num_qubits,
                         const InitializationSettings &kokkos_args) {
            return new StateVectorT(mpi_manager, num_qubits, kokkos_args);
        }))
        .def(py::init([](std::size_t num_qubits) {
            return new StateVectorT(num_qubits);
        }))
        .def(py::init([](std::size_t num_qubits,
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
            [](StateVectorT &sv, const np_arr_c &state,
               const std::vector<std::size_t> &wires) {
                const auto buffer = state.request();
                sv.setStateVector(static_cast<const ComplexT *>(buffer.ptr),
                                  wires);
            },
            "Set the state vector to the data contained in `state`.")
        .def(
            "DeviceToHost",
            [](StateVectorT &device_sv, np_arr_c &host_sv) {
                py::buffer_info numpyArrayInfo = host_sv.request();
                auto *data_ptr = static_cast<ComplexT *>(numpyArrayInfo.ptr);
                if (host_sv.size()) {
                    device_sv.DeviceToHost(data_ptr, host_sv.size());
                }
            },
            "Synchronize data from the Kokkos device to host.")
        .def("HostToDevice",
             py::overload_cast<ComplexT *, std::size_t>(
                 &StateVectorT::HostToDevice),
             "Synchronize data from the host device to Kokkos.")
        .def(
            "HostToDevice",
            [](StateVectorT &device_sv, const np_arr_c &host_sv) {
                const py::buffer_info numpyArrayInfo = host_sv.request();
                auto *data_ptr = static_cast<ComplexT *>(numpyArrayInfo.ptr);
                const auto length =
                    static_cast<std::size_t>(numpyArrayInfo.shape[0]);
                if (length) {
                    device_sv.HostToDevice(data_ptr, length);
                }
            },
            "Synchronize data from the host device to Kokkos.")
        .def(
            "apply",
            [](StateVectorT &sv, const std::string &str,
               const std::vector<std::size_t> &wires, bool inv,
               [[maybe_unused]] const std::vector<std::vector<ParamT>> &params,
               [[maybe_unused]] const np_arr_c &gate_matrix) {
                const auto m_buffer = gate_matrix.request();
                std::vector<Kokkos::complex<ParamT>> conv_matrix;
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const Kokkos::complex<ParamT> *>(
                            m_buffer.ptr);
                    conv_matrix = std::vector<Kokkos::complex<ParamT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
                sv.applyOperation(str, wires, inv, std::vector<ParamT>{},
                                  conv_matrix);
            },
            "Apply operation via the gate matrix")
        .def(
            "applyPauliRot",
            [](StateVectorT &sv, const std::vector<std::size_t> &wires,
               const bool inverse, const std::vector<ParamT> &params,
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
 * @brief Register backend specific measurements class functionalities.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Pybind11's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificMeasurementsMPI(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type
    using ParamT = PrecisionT;           // Parameter's data precision

    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;

    pyclass
        .def("expval",
             static_cast<PrecisionT (MeasurementsMPI<StateVectorT>::*)(
                 const std::string &, const std::vector<std::size_t> &)>(
                 &MeasurementsMPI<StateVectorT>::expval),
             "Expected value of an operation by name.")
        .def(
            "expval",
            [](MeasurementsMPI<StateVectorT> &M, const np_arr_c &matrix,
               const std::vector<std::size_t> &wires) {
                const std::size_t matrix_size = exp2(2 * wires.size());
                auto matrix_data =
                    static_cast<ComplexT *>(matrix.request().ptr);
                std::vector<ComplexT> matrix_v{matrix_data,
                                               matrix_data + matrix_size};
                return M.expval(matrix_v, wires);
            },
            "Expected value of a Hermitian observable.")
        .def(
            "expval",
            [](MeasurementsMPI<StateVectorT> &M,
               const std::vector<std::string> &pauli_words,
               const std::vector<std::vector<std::size_t>> &target_wires,
               const std::vector<ParamT> &coeffs) {
                return M.expval(pauli_words, target_wires, coeffs);
            },
            "Expected value of a Hamiltonian represented by Pauli words.")
        .def("var",
             [](MeasurementsMPI<StateVectorT> &M, const std::string &operation,
                const std::vector<std::size_t> &wires) {
                 return M.var(operation, wires);
             })
        .def("var",
             static_cast<PrecisionT (MeasurementsMPI<StateVectorT>::*)(
                 const std::string &, const std::vector<std::size_t> &)>(
                 &MeasurementsMPI<StateVectorT>::var),
             "Variance of an operation by name.");
}

/**
 * @brief Register observable classes.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendSpecificObservablesMPI(py::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ParamT = PrecisionT;             // Parameter's data precision

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;

    std::string class_name;

    class_name = "SparseHamiltonianC" + bitsize;
    py::class_<SparseHamiltonian<StateVectorT>,
               std::shared_ptr<SparseHamiltonian<StateVectorT>>,
               Observable<StateVectorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init([](const np_arr_c &data,
                         const std::vector<std::size_t> &indices,
                         const std::vector<std::size_t> &indptr,
                         const std::vector<std::size_t> &wires) {
            using ComplexT = typename StateVectorT::ComplexT;
            const py::buffer_info buffer_data = data.request();
            const auto *data_ptr = static_cast<ComplexT *>(buffer_data.ptr);

            return SparseHamiltonian<StateVectorT>{
                std::vector<ComplexT>({data_ptr, data_ptr + data.size()}),
                indices, indptr, wires};
        }))
        .def("__repr__", &SparseHamiltonian<StateVectorT>::getObsName)
        .def("get_wires", &SparseHamiltonian<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const SparseHamiltonian<StateVectorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<SparseHamiltonian<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<SparseHamiltonian<StateVectorT>>();
                return self == other_cast;
            },
            "Compare two observables");
}

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam StateVectorT
 * @param m Pybind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithmsMPI([[maybe_unused]] py::module_ &m) {}

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Pybind11 module.
 */
void registerBackendSpecificInfoMPI(py::module_ &m) {
    using np_arr_c64 = py::array_t<std::complex<float>,
                                   py::array::c_style | py::array::forcecast>;
    using np_arr_c128 = py::array_t<std::complex<double>,
                                    py::array::c_style | py::array::forcecast>;
    py::class_<MPIManagerKokkos>(m, "MPIManagerKokkos")
        .def(py::init<>())
        .def(py::init<MPIManagerKokkos &>())
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
                    send_ptr, recv_ptr, recvBuf.request().size, root);
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
                    send_ptr, recv_ptr, recvBuf.request().size, root);
            },
            "MPI Scatter.");
}
} // namespace Pennylane::LightningKokkos
