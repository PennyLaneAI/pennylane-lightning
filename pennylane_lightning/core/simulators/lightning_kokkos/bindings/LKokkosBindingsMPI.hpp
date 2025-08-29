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
 * @file LKokkosBindingsMPI.hpp
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

#include "Bindings.hpp"
#include "BindingsUtils.hpp"
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

/// @cond DEV
namespace {
using namespace Pennylane::NanoBindings;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Observables;
using Kokkos::InitializationSettings;
using Pennylane::LightningKokkos::StateVectorKokkos;
using Pennylane::LightningKokkos::Util::MPIManagerKokkos;
using Pennylane::Util::exp2;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::NanoBindings {

namespace nb = nanobind;

/// @cond DEV
/**
 * @brief Define StateVector backends for lightning.kokkos MPI
 */
using StateVectorMPIBackends =
    Pennylane::Util::TypeList<StateVectorKokkosMPI<float>,
                              StateVectorKokkosMPI<double>, void>;
/// @endcond

/**
 * @brief Register backend class specific bindings for MPI.
 *
 * @tparam StateVectorT The type of the state vector.
 * @tparam PyClass Nanobind's class to bind methods.
 * @param pyclass Nanobind's class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendSpecificStateVectorMethodsMPI(PyClass &pyclass) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;

    // Register gates for state vector
    registerGates<StateVectorT>(pyclass);
    registerControlledGates<StateVectorT>(pyclass);

    pyclass.def(nb::init<std::size_t>());
    pyclass.def(nb::init<MPIManagerKokkos &, std::size_t>());
    pyclass.def(nb::init<MPIManagerKokkos &, std::size_t,
                         const InitializationSettings &>());
    pyclass.def(nb::init<std::size_t, const InitializationSettings &>());

    pyclass.def("resetStateVector", &StateVectorT::resetStateVector);
    pyclass.def(
        "setBasisState",
        [](StateVectorT &sv, const std::vector<std::size_t> &state,
           const std::vector<std::size_t> &wires) {
            sv.setBasisState(state, wires);
        },
        "Set the state vector to a basis state.");
    pyclass.def(
        "setStateVector",
        [](StateVectorT &sv, const ArrayComplexT &state,
           const std::vector<std::size_t> &wires) {
            sv.setStateVector(PL_reinterpret_cast<const ComplexT>(state.data()),
                              wires);
        },
        "Set the state vector to the data contained in `state`.");
    pyclass.def(
        "DeviceToHost",
        [](StateVectorT &device_sv, ArrayComplexT &host_sv) {
            auto *data_ptr = PL_reinterpret_cast<ComplexT>(host_sv.data());
            if (host_sv.size()) {
                device_sv.DeviceToHost(data_ptr, host_sv.size());
            }
        },
        "Synchronize data from the Kokkos device to host.");
    pyclass.def(
        "HostToDevice",
        nb::overload_cast<ComplexT *, std::size_t>(&StateVectorT::HostToDevice),
        "Synchronize data from the host device to Kokkos.");
    pyclass.def(
        "HostToDevice",
        [](StateVectorT &device_sv, const ArrayComplexT &host_sv) {
            auto *data_ptr = const_cast<ComplexT *>(
                PL_reinterpret_cast<ComplexT>(host_sv.data()));
            if (host_sv.size()) {
                device_sv.HostToDevice(data_ptr, host_sv.size());
            }
        },
        "Synchronize data from the host device to Kokkos.");
    pyclass.def(
        "apply",
        [](StateVectorT &sv, const std::string &str,
           const std::vector<std::size_t> &wires, bool inv,
           [[maybe_unused]] const std::vector<std::vector<PrecisionT>> &params,
           [[maybe_unused]] const ArrayComplexT &gate_matrix) {
            std::vector<ComplexT> conv_matrix;
            if (gate_matrix.size()) {
                conv_matrix = std::vector<ComplexT>{gate_matrix.data(),
                                                    gate_matrix.data() +
                                                        gate_matrix.size()};
            }
            sv.applyOperation(str, wires, inv, std::vector<PrecisionT>{},
                              conv_matrix);
        },
        "Apply a matrix operation.");
    pyclass.def(
        "applyPauliRot",
        [](StateVectorT &sv, const std::vector<std::size_t> &wires,
           const bool inverse, const std::vector<PrecisionT> &params,
           const std::string &word) {
            sv.applyPauliRot(wires, inverse, params, word);
        },
        "Apply a Pauli rotation.");
    pyclass.def("applyControlledMatrix", &applyControlledMatrix<StateVectorT>,
                "Apply controlled operation");
    pyclass.def(
        "collapse", &StateVectorT::collapse,
        "Collapse the statevector onto the 0 or 1 branch of a given wire.");
    pyclass.def(
        "getNumLocalWires",
        [](StateVectorT &sv) { return sv.getNumLocalWires(); },
        "Get number of local wires.");
    pyclass.def(
        "getNumGlobalWires",
        [](StateVectorT &sv) { return sv.getNumGlobalWires(); },
        "Get number of global wires.");
    pyclass.def(
        "swapGlobalLocalWires",
        [](StateVectorT &sv,
           const std::vector<std::size_t> &global_wires_to_swap,
           const std::vector<std::size_t> &local_wires_to_swap) {
            sv.swapGlobalLocalWires(global_wires_to_swap, local_wires_to_swap);
        },
        "Swap global and local wires - global_wire_to_swap must be in "
        "global_wires_ and local_wires_to_swap must be in local_wires_");
    pyclass.def(
        "getLocalBlockSize",
        [](StateVectorT &sv) { return sv.getLocalBlockSize(); },
        "Get Local Block Size, i.e. size of SV on a single rank.");
    pyclass.def(
        "resetIndices", [](StateVectorT &sv) { sv.resetIndices(); },
        "Reset indices including global_wires, local_wires_, and "
        "mpi_rank_to_global_index_map_.");
    pyclass.def(
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
    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;

    pyclass.def(
        "expval",
        [](MeasurementsMPI<StateVectorT> &M, const std::string &operation,
           const std::vector<std::size_t> &wires) {
            return M.expval(operation, wires);
        },
        "Expected value of an operation by name.");
    pyclass.def(
        "expval",
        [](MeasurementsMPI<StateVectorT> &M, const ArrayComplexT &matrix,
           const std::vector<std::size_t> &wires) {
            const std::size_t matrix_size = exp2(2 * wires.size());
            auto matrix_data =
                PL_reinterpret_cast<const ComplexT>(matrix.data());
            std::vector<ComplexT> matrix_v{matrix_data,
                                           matrix_data + matrix_size};
            return M.expval(matrix_v, wires);
        },
        "Expected value of a Hermitian observable.");
    pyclass.def(
        "expval",
        [](MeasurementsMPI<StateVectorT> &M,
           const std::vector<std::string> &pauli_words,
           const std::vector<std::vector<std::size_t>> &target_wires,
           const std::vector<PrecisionT> &coeffs) {
            return M.expval(pauli_words, target_wires, coeffs);
        },
        "Expected value of a Hamiltonian represented by Pauli words.");
    pyclass.def(
        "var",
        [](MeasurementsMPI<StateVectorT> &M, const std::string &operation,
           const std::vector<std::size_t> &wires) {
            return M.var(operation, wires);
        },
        "Variance of an operation by name.");
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

    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;
    using SparseIndexT = std::size_t;
    using ArrSparseIndT = nb::ndarray<SparseIndexT, nb::c_contig>;

    std::string class_name = "SparseHamiltonianC" + bitsize;
    auto sparse_hamiltonian_class =
        nb::class_<SparseHamiltonian<StateVectorT>>(m, class_name.c_str());

    sparse_hamiltonian_class.def(
        "__init__",
        [](SparseHamiltonian<StateVectorT> *self, const ArrayComplexT &data,
           const std::vector<std::size_t> &indices,
           const std::vector<std::size_t> &indptr,
           const std::vector<std::size_t> &wires) {
            const ComplexT *data_ptr =
                PL_reinterpret_cast<const ComplexT>(data.data());
            std::vector<ComplexT> data_vec(data_ptr, data_ptr + data.size());
            new (self) SparseHamiltonian<StateVectorT>(data_vec, indices,
                                                       indptr, wires);
        });

    sparse_hamiltonian_class.def("__repr__",
                                 &SparseHamiltonian<StateVectorT>::getObsName,
                                 "Get the name of the observable");
    sparse_hamiltonian_class.def("get_wires",
                                 &SparseHamiltonian<StateVectorT>::getWires,
                                 "Get wires of observables");

    sparse_hamiltonian_class.def(
        "__eq__",
        [](const SparseHamiltonian<StateVectorT> &self,
           nb::handle other) -> bool {
            if (!nb::isinstance<SparseHamiltonian<StateVectorT>>(other)) {
                return false;
            }
            auto other_cast = nb::cast<SparseHamiltonian<StateVectorT>>(other);
            return self == other_cast;
        },
        "Compare two observables");
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
    using ArrayComplex64T = nb::ndarray<std::complex<float>, nb::c_contig>;
    using ArrayComplex128T = nb::ndarray<std::complex<double>, nb::c_contig>;

    auto mpi_manager_class =
        nb::class_<MPIManagerKokkos>(m, "MPIManagerKokkos");
    mpi_manager_class.def(nb::init<>());
    mpi_manager_class.def(nb::init<MPIManagerKokkos &>());
    mpi_manager_class.def("Barrier", &MPIManagerKokkos::Barrier);
    mpi_manager_class.def("getRank", &MPIManagerKokkos::getRank);
    mpi_manager_class.def("getSize", &MPIManagerKokkos::getSize);
    mpi_manager_class.def("getSizeNode", &MPIManagerKokkos::getSizeNode);
    mpi_manager_class.def("getTime", &MPIManagerKokkos::getTime);
    mpi_manager_class.def("getVendor", &MPIManagerKokkos::getVendor);
    mpi_manager_class.def("getVersion", &MPIManagerKokkos::getVersion);
    mpi_manager_class.def(
        "Scatter",
        [](MPIManagerKokkos &mpi_manager, ArrayComplex64T &sendBuf,
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
        [](MPIManagerKokkos &mpi_manager, ArrayComplex128T &sendBuf,
           ArrayComplex128T &recvBuf, int root) {
            auto send_ptr = static_cast<std::complex<double> *>(sendBuf.data());
            auto recv_ptr = static_cast<std::complex<double> *>(recvBuf.data());
            mpi_manager.template Scatter<std::complex<double>>(
                send_ptr, recv_ptr, static_cast<std::size_t>(recvBuf.size()),
                root);
        },
        "MPI Scatter for complex double arrays.");
}

} // namespace Pennylane::LightningKokkos::NanoBindings
