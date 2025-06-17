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
 * @file LGPUBindings_nb.hpp
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
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Observables;
} // namespace
/// @endcond

namespace Pennylane::LightningGPU::NanoBindings {

/**
 * @brief Define StateVector backends for lightning.gpu
 */
using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorCudaManaged<float>,
                              StateVectorCudaManaged<double>, void>;

/**
 * @brief Register controlled matrix kernel.
 */
template <class StateVectorT>
void applyControlledMatrix(
    StateVectorT &st,
    const nb::ndarray<std::complex<typename StateVectorT::PrecisionT>,
                      nb::c_contig> &matrix,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires, bool inverse = false) {
    st.applyControlledMatrix(matrix.data(), controlled_wires, controlled_values,
                             wires, inverse);
}

template <class StateVectorT, class PyClass>
void registerControlledGate(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ParamT = PrecisionT;             // Parameter's data precision

    using Pennylane::Gates::ControlledGateOperation;
    using Pennylane::Util::for_each_enum;
    namespace Constant = Pennylane::Gates::Constant;

    for_each_enum<ControlledGateOperation>(
        [&pyclass](ControlledGateOperation gate_op) {
            using Pennylane::Util::lookup;
            const auto gate_name =
                std::string(lookup(Constant::controlled_gate_names, gate_op));
            const std::string doc = "Apply the " + gate_name + " gate.";
            auto func = [gate_name](
                            StateVectorT &sv,
                            const std::vector<std::size_t> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<std::size_t> &wires, bool inverse,
                            const std::vector<ParamT> &params) {
                sv.applyOperation(gate_name, controlled_wires,
                                  controlled_values, wires, inverse, params);
            };
            pyclass.def(gate_name.c_str(), func, doc.c_str());
        });
}

/**
 * @brief Get a controlled matrix and kernel map for a statevector.
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's statevector class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    registerControlledGate<StateVectorT>(pyclass);
    registerBackendSpecificStateVectorMethods<StateVectorT>(pyclass);
}

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
    using ParamT = PrecisionT;           // Parameter's data precision

    using ArrayT = nb::ndarray<std::complex<ParamT>, nb::c_contig>;
    using IdxT = typename std::conditional<std::is_same<ParamT, float>::value,
                                           int32_t, int64_t>::type;
    using ArraySparseIndexT = nb::ndarray<IdxT, nb::c_contig>;

    pyclass
        .def(
            "expval",
            [](Measurements<StateVectorT> &M, const std::string &operation,
               const std::vector<std::size_t> &wires) {
                M.expval(operation, wires);
            },
            "Expected value of an operation by name.")
        .def(
            "expval",
            [](Measurements<StateVectorT> &M, const ArraySparseIndexT &row_map,
               const ArraySparseIndexT &entries, const ArrayT &values) {
                return M.expval(
                    row_map.data(),
                    static_cast<int64_t>(
                        row_map.size()), // int64_t is required by cusparse
                    entries.data(), values.data(),
                    static_cast<int64_t>(
                        values.size())); // int64_t is required by cusparse
            },
            "Expected value of a sparse Hamiltonian.")
        .def(
            "expval",
            [](Measurements<StateVectorT> &M,
               const std::vector<std::string> &pauli_words,
               const std::vector<std::vector<std::size_t>> &target_wires,
               const ArrayT &coeffs) {
                return M.expval(pauli_words, target_wires, coeffs.data());
            },
            "Expected value of Hamiltonian represented by Pauli words.")
        .def(
            "expval",
            [](Measurements<StateVectorT> &M, const ArrayT &matrix,
               const std::vector<std::size_t> &wires) {
                const std::size_t matrix_size = exp2(2 * wires.size());
                std::vector<ComplexT> matrix_v{matrix.data(),
                                               matrix.data() + matrix_size};
                return M.expval(matrix_v, wires);
            },
            "Expected value of a Hermitian observable.")
        .def("var",
             [](Measurements<StateVectorT> &M, const std::string &operation,
                const std::vector<std::size_t> &wires) {
                 return M.var(operation, wires);
             })
        .def("var",
             static_cast<PrecisionT (Measurements<StateVectorT>::*)(
                 const std::string &, const std::vector<std::size_t> &)>(
                 &Measurements<StateVectorT>::var),
             "Variance of an operation by name.")
        .def(
            "var",
            [](Measurements<StateVectorT> &M, const ArraySparseIndexT &row_map,
               const ArraySparseIndexT &entries, const ArrayT &values) {
                return M.var(row_map.data(),
                             static_cast<int64_t>(row_map.size()),
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
    using ParamT = PrecisionT;           // Parameter's data precision

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    using ArrayT = nb::ndarray<std::complex<ParamT>, nb::c_contig>;

    std::string class_name;

    class_name = "SparseHamiltonianC" + bitsize;
    using IdxT = typename SparseHamiltonian<StateVectorT>::IdxT;
    using ArraySparseIndexT = nb::ndarray<IdxT, nb::c_contig>;

    nb::class_<SparseHamiltonian<StateVectorT>, Observable<StateVectorT>>(
        m, class_name.c_str())
        .def("__init__",
             [](const ArrayT &data, const ArraySparseIndexT &indices,
                const ArraySparseIndexT &offsets,
                const std::vector<std::size_t> &wires) {
                 // TODO: We can probably avoid a copy here by not constructing
                 // a vector
                 return SparseHamiltonian<StateVectorT>{
                     std::vector<ComplexT>(
                         {data.data(), data.data() + data.size()}),
                     std::vector<IdxT>(
                         {indices.data(), indices.data() + indices.size()}),
                     std::vector<IdxT>(
                         {offsets.data(), offsets.data() + offsets.size()}),
                     wires};
             })
        .def("__repr__", &SparseHamiltonian<StateVectorT>::getObsName)
        .def("get_wires", &SparseHamiltonian<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            []([[maybe_unused]] const SparseHamiltonian<StateVectorT> &self,
               nb::handle other) -> bool {
                if (!nb::isinstance<SparseHamiltonian<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast =
                    nb::cast<SparseHamiltonian<StateVectorT>>(other);
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
void registerBackendSpecificAlgorithms([[maybe_unused]] nb::module_ &m) {} // m

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
    using ParamT = PrecisionT;             // Parameter's data precision
    using ArrayT = nb::ndarray<std::complex<ParamT>, nb::c_contig>;

    pyclass
        .def(nb::init<std::size_t>())              // qubits, device
        .def(nb::init<std::size_t, DevTag<int>>()) // qubits, dev-tag
        .def("__init__",
             [](const ArrayT &arr) {
                 return new StateVectorT(arr.data(), arr.size());
             })
        .def(
            "setBasisState",
            [](StateVectorT &sv, const std::vector<std::size_t> &state,
               const std::vector<std::size_t> &wires,
               const bool async) { sv.setBasisState(state, wires, async); },
            nb::arg("state") = nullptr, nb::arg("wires") = nullptr,
            nb::arg("async") = false,
            "Set the state vector to a basis state on GPU.")
        .def(
            "setStateVector",
            [](StateVectorT &sv, const ArrayT &state,
               const std::vector<std::size_t> &wires,
               const bool async = false) {
                // TODO: Check that adding in a default value for async here is
                // a reasonable API change
                sv.setStateVector(state.data(), state.size(), wires, async);
            },
            nb::arg("state"), nb::arg("wires"), nb::arg("async") = false,
            "Set State Vector on GPU with values for the state vector and "
            "wires on the host memory.")
        .def("applyControlledMatrix", &applyControlledMatrix<StateVectorT>,
             "Apply controlled operation")
        .def(
            "DeviceToDevice",
            [](StateVectorT &sv, const StateVectorT &other, bool async) {
                sv.updateData(other, async);
            },
            "Synchronize data from another GPU device to current device.")
        .def("DeviceToHost",
             nb::overload_cast<std::complex<PrecisionT> *, std::size_t, bool>(
                 &StateVectorT::CopyGpuDataToHost, nb::const_),
             "Synchronize data from the GPU device to host.")
        .def(
            "DeviceToHost",
            [](const StateVectorT &gpu_sv, ArrayT &cpu_sv, bool) {
                if (cpu_sv.size()) {
                    gpu_sv.CopyGpuDataToHost(cpu_sv.data(), cpu_sv.size());
                }
            },
            "Synchronize data from the GPU device to host.")
        .def("HostToDevice",
             nb::overload_cast<const std::complex<PrecisionT> *, std::size_t,
                               bool>(&StateVectorT::CopyHostDataToGpu),
             "Synchronize data from the host device to GPU.")
        .def(
            "HostToDevice",
            [](StateVectorT &gpu_sv, const ArrayT &cpu_sv, bool async) {
                if (cpu_sv.size()) {
                    gpu_sv.CopyHostDataToGpu(cpu_sv.data(), cpu_sv.size(),
                                             async);
                }
            },
            "Synchronize data from the host device to GPU.")
        .def("GetNumGPUs", &getGPUCount, "Get the number of available GPUs.")
        .def("getCurrentGPU", &getGPUIdx,
             "Get the GPU index for the statevector data.")
        .def("numQubits", &StateVectorT::getNumQubits)
        .def("dataLength", &StateVectorT::getLength)
        .def(
            "resetStateVector",
            [](StateVectorT &gpu_sv, bool async) {
                gpu_sv.resetStateVector(async);
            },
            nb::arg("async") = false,
            "Initialize the statevector data to the |0...0> state")
        .def("collapse", &StateVectorT::collapse,
             "Collapse the statevector onto the 0 or 1 branch of a given wire.")
        .def(
            "apply",
            [](StateVectorT &sv, const std::string &gate_name,
               const std::vector<std::size_t> &controlled_wires,
               const std::vector<bool> &controlled_values,
               const std::vector<std::size_t> &wires, bool inverse,
               const std::vector<ParamT> &params) {
                sv.applyOperation(gate_name, controlled_wires,
                                  controlled_values, wires, inverse, params);
            },
            "Apply operation via the gate matrix")
        .def(
            "apply",
            [](StateVectorT &sv, const std::string &str,
               const std::vector<std::size_t> &wires, bool inv,
               const std::vector<std::vector<ParamT>> &params,
               const ArrayT &gate_matrix) {
                if (params.empty()) {
                    sv.applyOperation(str, wires, inv, std::vector<ParamT>{},
                                      gate_matrix.data(), gate_matrix.size());
                } else {
                    PL_ABORT_IF(params.size() != 1,
                                "params should be a List[List[float]].")
                    sv.applyOperation(str, wires, inv, params[0],
                                      gate_matrix.data(), gate_matrix.size());
                }
            },
            "Apply operation via the gate matrix")
        .def(
            "getState",
            [](const StateVectorT &sv, ArrayT &state) {
                sv.CopyGpuDataToHost(state.data(), state.size());
            },
            "Copy state vector data to a numpy array.", nb::arg("state"));
} // pyclass

} // namespace Pennylane::LightningGPU::NanoBindings
