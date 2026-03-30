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
 * @file LKokkosBindings.hpp
 * Defines lightning.kokkos specific operations to export to Python using
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

#include "BindingsUtils.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "GateOperation.hpp"
#include "Kokkos_Core.hpp"
#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"
#include "StateVectorKokkos.hpp"
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
using Pennylane::Util::exp2;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::NanoBindings {

namespace nb = nanobind;

/// @cond DEV
/**
 * @brief Define StateVector backends for lightning.kokkos
 */
using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorKokkos<float>,
                              StateVectorKokkos<double>, void>;
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
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;
    using SparseIndexT = std::size_t;
    using arr_sparse_ind = nb::ndarray<SparseIndexT, nb::c_contig>;

    // Named observable methods
    pyclass
        .def(
            "expval",
            [](Measurements<StateVectorT> &M, const std::string &operation,
               const std::vector<std::size_t> &wires) {
                return M.expval(operation, wires);
            },
            "Expected value of an operation by name.")
        .def(
            "var",
            [](Measurements<StateVectorT> &M, const std::string &operation,
               const std::vector<std::size_t> &wires) {
                return M.var(operation, wires);
            },
            "Variance of an operation by name.");

    // Kokkos-specific measurement methods
    pyclass
        .def(
            "expval",
            [](Measurements<StateVectorT> &M, const ArrayComplexT &matrix,
               const std::vector<std::size_t> &wires) {
                const std::size_t matrix_size = exp2(2 * wires.size());
                auto matrix_data =
                    PL_reinterpret_cast<const ComplexT>(matrix.data());
                std::vector<ComplexT> matrix_v{matrix_data,
                                               matrix_data + matrix_size};
                return M.expval(matrix_v, wires);
            },
            "Expected value of a Hermitian observable.")
        .def(
            "expval",
            [](Measurements<StateVectorT> &M,
               const std::vector<std::string> &pauli_words,
               const std::vector<std::vector<std::size_t>> &target_wires,
               const std::vector<PrecisionT> &coeffs) {
                return M.expval(pauli_words, target_wires, coeffs);
            },
            "Expected value of a Hamiltonian represented by Pauli words.")
        .def(
            "expval",
            [](Measurements<StateVectorT> &M, const arr_sparse_ind &row_map,
               const arr_sparse_ind &entries, const ArrayComplexT &values) {
                return M.expval(
                    static_cast<SparseIndexT *>(row_map.data()),
                    static_cast<SparseIndexT>(row_map.size()),
                    static_cast<SparseIndexT *>(entries.data()),
                    PL_reinterpret_cast<const ComplexT>(values.data()),
                    static_cast<SparseIndexT>(values.size()));
            },
            "Expected value of a sparse Hamiltonian.")
        .def(
            "var",
            [](Measurements<StateVectorT> &M, const arr_sparse_ind &row_map,
               const arr_sparse_ind &entries, const ArrayComplexT &values) {
                return M.var(static_cast<SparseIndexT *>(row_map.data()),
                             static_cast<SparseIndexT>(row_map.size()),
                             static_cast<SparseIndexT *>(entries.data()),
                             PL_reinterpret_cast<const ComplexT>(values.data()),
                             static_cast<SparseIndexT>(values.size()));
            },
            "Variance of a sparse Hamiltonian.");
}

/**
 * @brief Register backend specific observables.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendSpecificObservables(nb::module_ &m) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    const std::string bitsize =
        std::is_same_v<PrecisionT, float> ? "64" : "128";

    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;
    using SparseIndexT = std::size_t;

    // Register Kokkos-specific observables
    std::string class_name = "SparseHamiltonianC" + bitsize;
    auto sparse_hamiltonian_class =
        nb::class_<SparseHamiltonian<StateVectorT>, Observable<StateVectorT>>(
            m, class_name.c_str());

    sparse_hamiltonian_class.def(
        nb::init<std::vector<ComplexT>, std::vector<SparseIndexT>,
                 std::vector<SparseIndexT>, std::vector<std::size_t>>(),
        "Initialize SparseHamiltonian with data, indices, indptr, and wires");

    sparse_hamiltonian_class.def(
        "__init__",
        [](SparseHamiltonian<StateVectorT> *self, const ArrayComplexT &data,
           const std::vector<SparseIndexT> &indices,
           const std::vector<SparseIndexT> &indptr,
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
void registerBackendSpecificInfo(nb::module_ &m) {
    m.def("kokkos_initialize", []() { Kokkos::initialize(); });
    m.def("kokkos_initialize",
          [](const InitializationSettings &args) { Kokkos::initialize(args); });
    m.def("kokkos_finalize", []() { Kokkos::finalize(); });
    m.def("kokkos_is_initialized", []() { return Kokkos::is_initialized(); });
    m.def("kokkos_is_finalized", []() { return Kokkos::is_finalized(); });
    m.def(
        "print_configuration",
        []() {
            std::ostringstream buffer;
            Kokkos::print_configuration(buffer, true);
            return buffer.str();
        },
        "Kokkos configurations query.");

    nb::class_<InitializationSettings>(m, "InitializationSettings")
        .def(nb::init<>())
        .def("__post_init__",
             [](InitializationSettings &settings) {
                 settings.set_num_threads(0)
                     .set_device_id(0)
                     .set_map_device_id_by("")
                     .set_disable_warnings(0)
                     .set_print_configuration(0)
                     .set_tune_internals(0)
                     .set_tools_libs("")
                     .set_tools_help(0)
                     .set_tools_args("");
                 return settings;
             })
        .def("get_num_threads", &InitializationSettings::get_num_threads,
             "Number of threads to use with the host parallel backend. Must be "
             "greater than zero.")
        .def("get_device_id", &InitializationSettings::get_device_id,
             "Device to use with the device parallel backend. Valid IDs are "
             "zero to number of GPU(s) available for execution minus one.")
        .def(
            "get_map_device_id_by",
            &InitializationSettings::get_map_device_id_by,
            "Strategy to select a device automatically from the GPUs available "
            "for execution. Must be either mpi_rank"
            "for round-robin assignment based on the local MPI rank or random.")
        .def("get_disable_warnings",
             &InitializationSettings::get_disable_warnings,
             "Whether to disable warning messages.")
        .def("get_print_configuration",
             &InitializationSettings::get_print_configuration,
             "Whether to print the configuration after initialization.")
        .def("get_tune_internals", &InitializationSettings::get_tune_internals,
             "Whether to allow autotuning internals instead of using "
             "heuristics.")
        .def("get_tools_libs", &InitializationSettings::get_tools_libs,
             "Which tool dynamic library to load. Must either be the full path "
             "to library or the name of library if the path is present in the "
             "runtime library search path (e.g. LD_LIBRARY_PATH)")
        .def("get_tools_help", &InitializationSettings::get_tools_help,
             "Query the loaded tool for its command-line options support.")
        .def("get_tools_args", &InitializationSettings::get_tools_args,
             "Options to pass to the loaded tool as command-line arguments.")
        .def("has_num_threads", &InitializationSettings::has_num_threads)
        .def("has_device_id", &InitializationSettings::has_device_id)
        .def("has_map_device_id_by",
             &InitializationSettings::has_map_device_id_by)
        .def("has_disable_warnings",
             &InitializationSettings::has_disable_warnings)
        .def("has_print_configuration",
             &InitializationSettings::has_print_configuration)
        .def("has_tune_internals", &InitializationSettings::has_tune_internals)
        .def("has_tools_libs", &InitializationSettings::has_tools_libs)
        .def("has_tools_help", &InitializationSettings::has_tools_help)
        .def("has_tools_args", &InitializationSettings::has_tools_args)
        .def("set_num_threads", &InitializationSettings::set_num_threads)
        .def("set_device_id", &InitializationSettings::set_device_id)
        .def("set_map_device_id_by",
             &InitializationSettings::set_map_device_id_by)
        .def("set_disable_warnings",
             &InitializationSettings::set_disable_warnings)
        .def("set_print_configuration",
             &InitializationSettings::set_print_configuration)
        .def("set_tune_internals", &InitializationSettings::set_tune_internals)
        .def("set_tools_libs", &InitializationSettings::set_tools_libs)
        .def("set_tools_help", &InitializationSettings::set_tools_help)
        .def("set_tools_args", &InitializationSettings::set_tools_args)
        .def("__repr__", [](const InitializationSettings &args) {
            std::ostringstream args_stream;
            args_stream << "InitializationSettings:\n";
            args_stream << "num_threads = " << args.get_num_threads() << '\n';
            args_stream << "device_id = " << args.get_device_id() << '\n';
            args_stream << "map_device_id_by = " << args.get_map_device_id_by()
                        << '\n';
            args_stream << "disable_warnings = " << args.get_disable_warnings()
                        << '\n';
            args_stream << "print_configuration = "
                        << args.get_print_configuration() << '\n';
            args_stream << "tune_internals = " << args.get_tune_internals()
                        << '\n';
            args_stream << "tools_libs = " << args.get_tools_libs() << '\n';
            args_stream << "tools_help = " << args.get_tools_help() << '\n';
            args_stream << "tools_args = " << args.get_tools_args();
            return args_stream.str();
        });
    m.def(
        "backend_info",
        []() {
            nb::dict info;

            info["NAME"] = "lightning.kokkos";

            return info;
        },
        "Backend-specific information.");
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
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;

    // Add Pauli rotation - Kokkos specific implementation
    pyclass.def(
        "applyPauliRot",
        [](StateVectorT &sv, const std::vector<std::size_t> &wires,
           const bool inverse, const std::vector<PrecisionT> &params,
           const std::string &word) {
            sv.applyPauliRot(wires, inverse, params, word);
        },
        "Apply a Pauli rotation.");

    // Add Kokkos-specific constructors
    pyclass
        .def(
            "__init__",
            [](StateVectorT *self, std::size_t num_qubits) {
                new (self) StateVectorT(num_qubits);
            },
            "Initialize with number of qubits")
        .def(
            "__init__",
            [](StateVectorT *self, std::size_t num_qubits,
               const InitializationSettings &kokkos_args) {
                new (self) StateVectorT(num_qubits, kokkos_args);
            },
            "Initialize with number of qubits and Kokkos settings");

    // Kokkos-specific data transfer methods
    pyclass
        .def("resetStateVector", &StateVectorT::resetStateVector,
             "Reset the state vector to the zero state.")
        .def(
            "setBasisState",
            [](StateVectorT &sv, const std::vector<std::size_t> &state,
               const std::vector<std::size_t> &wires) {
                sv.setBasisState(state, wires);
            },
            "Set the state vector to a basis state.")
        .def(
            "setStateVector",
            [](StateVectorT &sv, const ArrayComplexT &state,
               const std::vector<std::size_t> &wires) {
                sv.setStateVector(
                    PL_reinterpret_cast<const ComplexT>(state.data()), wires);
            },
            "Set the state vector to the data contained in `state`.")
        .def(
            "DeviceToHost",
            [](StateVectorT &device_sv, ArrayComplexT &host_sv) {
                auto *data_ptr = PL_reinterpret_cast<ComplexT>(host_sv.data());
                if (host_sv.size()) {
                    device_sv.DeviceToHost(data_ptr, host_sv.size());
                }
            },
            "Synchronize data from the Kokkos device to host.")
        .def(
            "HostToDevice",
            [](StateVectorT &device_sv, const ArrayComplexT &host_sv) {
                auto *data_ptr = const_cast<ComplexT *>(
                    PL_reinterpret_cast<ComplexT>(host_sv.data()));
                if (host_sv.size()) {
                    device_sv.HostToDevice(data_ptr, host_sv.size());
                }
            },
            "Synchronize data from the host device to Kokkos.");

    // Apply operation method
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

    // Collapse method
    pyclass.def(
        "collapse", &StateVectorT::collapse,
        "Collapse the statevector onto the 0 or 1 branch of a given wire.");
}

/**
 * @brief Register backend specific state vector methods.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's state vector class to bind methods.
 *
 * @deprecated Use registerBackendSpecificStateVectorMethods instead
 */
template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    // This function is kept for backward compatibility
    // All functionality has been moved to
    // registerBackendSpecificStateVectorMethods
    registerBackendSpecificStateVectorMethods<StateVectorT>(pyclass);
}

} // namespace Pennylane::LightningKokkos::NanoBindings
