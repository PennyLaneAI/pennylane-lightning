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
 * @file LQubitBindings.hpp
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

#include "BindingsUtils.hpp"
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "DynamicDispatcher.hpp"
#include "GateOperation.hpp"
#include "MeasurementsLQubit.hpp"
#include "ObservablesLQubit.hpp"
#include "StateVectorLQubitManaged.hpp"
#include "TypeList.hpp"
#include "VectorJacobianProduct.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::NanoBindings;
using namespace Pennylane::LightningQubit::Algorithms;
using namespace Pennylane::LightningQubit::Measures;
using namespace Pennylane::LightningQubit::Observables;
using Pennylane::LightningQubit::StateVectorLQubitManaged;
using Pennylane::NanoBindings::Utils::createNumpyArrayFromVector;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::NanoBindings {

namespace nb = nanobind;

/// @cond DEV
/**
 * @brief Define StateVector backends for lightning.qubit
 */
using StateVectorBackends =
    Pennylane::Util::TypeList<StateVectorLQubitManaged<float>,
                              StateVectorLQubitManaged<double>, void>;
/// @endcond

/**
 * @brief Update state vector data from an array
 *
 * This function accepts any array-like object that follows the buffer protocol,
 * including NumPy arrays and JAX arrays (for example).
 */
template <class StateVectorT>
auto svKernelMap(const StateVectorT &sv) -> nb::dict {
    using PrecisionT = typename StateVectorT::PrecisionT;
    nb::dict res_map;
    namespace Constant = Pennylane::Gates::Constant;
    using Pennylane::Util::lookup;

    const auto &dispatcher = DynamicDispatcher<PrecisionT>::getInstance();

    auto [GateKernelMap, GeneratorKernelMap, MatrixKernelMap,
          ControlledGateKernelMap, ControlledGeneratorKernelMap,
          ControlledMatrixKernelMap] = sv.getSupportedKernels();

    for (const auto &[gate_op, kernel] : GateKernelMap) {
        const auto key = std::string(lookup(Constant::gate_names, gate_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[gen_op, kernel] : GeneratorKernelMap) {
        const auto key = std::string(lookup(Constant::generator_names, gen_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[mat_op, kernel] : MatrixKernelMap) {
        const auto key = std::string(lookup(Constant::matrix_names, mat_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[mat_op, kernel] : ControlledGateKernelMap) {
        const auto key =
            std::string(lookup(Constant::controlled_gate_names, mat_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[mat_op, kernel] : ControlledGeneratorKernelMap) {
        const auto key =
            std::string(lookup(Constant::controlled_generator_names, mat_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    for (const auto &[mat_op, kernel] : ControlledMatrixKernelMap) {
        const auto key =
            std::string(lookup(Constant::controlled_matrix_names, mat_op));
        const auto value = dispatcher.getKernelName(kernel);

        res_map[key.c_str()] = value;
    }

    return res_map;
}

/**
 * @brief Register sparse matrix operators for a statevector.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's state vector class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerSparseMatrixOperators(PyClass &pyclass) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;
    using SparseIndexT = std::size_t;
    using ArraySparseIndT = nb::ndarray<SparseIndexT, nb::c_contig>;

    pyclass.def(
        "applySparseMatrix",
        [](StateVectorT &st, const ArraySparseIndT &row_map,
           const ArraySparseIndT &col_idx, const ArrayComplexT &values,
           const std::vector<std::size_t> &wires, bool inverse) {
            st.applySparseMatrix(static_cast<SparseIndexT *>(row_map.data()),
                                 static_cast<SparseIndexT *>(col_idx.data()),
                                 static_cast<ComplexT *>(values.data()), wires,
                                 inverse);
        },
        "Apply a sparse matrix to the statevector.");

    pyclass.def(
        "applyControlledSparseMatrix",
        [](StateVectorT &st, const ArraySparseIndT &row_map,
           const ArraySparseIndT &col_idx, const ArrayComplexT &values,
           const std::vector<std::size_t> &controlled_wires,
           const std::vector<bool> &controlled_values,
           const std::vector<std::size_t> &wires, bool inverse) {
            st.applyControlledSparseMatrix(
                static_cast<SparseIndexT *>(row_map.data()),
                static_cast<SparseIndexT *>(col_idx.data()),
                static_cast<ComplexT *>(values.data()), controlled_wires,
                controlled_values, wires, inverse);
        },
        "Apply a controlled sparse matrix to the statevector.");
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

    // Register sparse matrix operators.
    registerSparseMatrixOperators<StateVectorT>(pyclass);

    pyclass.def(nb::init<std::size_t>(), "Initialize with number of qubits");

    // Add updateData method for LQubit
    pyclass.def(
        "updateData",
        [](StateVectorT &sv, const nb::ndarray<const std::complex<PrecisionT>,
                                               nb::c_contig> &data) {
            if (data.ndim() != 1) {
                throw std::invalid_argument("Array must be 1-dimensional");
            }
            std::size_t size = data.shape(0);
            sv.updateData(data.data(), size);
        },
        "Update the state vector data from an array.", nb::arg("data"));

    // Add Pauli rotation.
    pyclass.def(
        "applyPauliRot",
        [](StateVectorT &sv, const std::vector<std::size_t> &wires,
           const bool inverse, const std::vector<PrecisionT> &params,
           const std::string &word) {
            sv.applyPauliRot(wires, inverse, params, word);
        },
        "Apply a Pauli rotation.");

    // Collapse and normalize methods.
    pyclass.def(
        "collapse", &StateVectorT::collapse,
        "Collapse the statevector onto the 0 or 1 branch of a given wire.");

    pyclass.def("normalize", &StateVectorT::normalize,
                "Normalizes the statevector to norm 1.");

    // Kernel map.
    pyclass.def("kernel_map", &svKernelMap<StateVectorT>,
                "Get internal kernels for operations");

    pyclass.def(
        "getState",
        [](const StateVectorT &sv, nb::ndarray<ComplexT, nb::c_contig> &state) {
            // Check if array is large enough
            if (state.shape(0) < sv.getLength()) {
                throw std::invalid_argument("Output array is too small");
            }

            // Copy data to DLPack array
            ComplexT *data_ptr = state.data();
            std::copy(sv.getData(), sv.getData() + sv.getLength(), data_ptr);
        },
        "Copy StateVector data into a DLPack (Numpy-like) array.",
        nb::arg("state"));
}

/**
 * @brief Get a controlled matrix and kernel map for a statevector.
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's measurements class to bind methods.
 *
 * @deprecated Use registerBackendSpecificStateVectorMethods instead
 */
template <class StateVectorT, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    // This function is kept for backward compatibility
    // All functionality has been moved to
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
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;
    using SparseIndexT = std::size_t;
    using ArraySparseIndT = nb::ndarray<SparseIndexT, nb::c_contig>;

    pyclass.def(
        "expval",
        [](Measurements<StateVectorT> &M, const std::string &name,
           const std::vector<std::size_t> &wires) {
            return M.expval(name, wires);
        },
        "Expected value of an operation by name.");

    pyclass.def(
        "expval",
        [](Measurements<StateVectorT> &M, const ArrayComplexT &matrix,
           const std::vector<std::size_t> &wires) {
            const std::size_t matrix_size = matrix.size();
            auto matrix_data =
                static_cast<std::complex<PrecisionT> *>(matrix.data());
            std::vector<std::complex<PrecisionT>> matrix_v{
                matrix_data, matrix_data + matrix_size};
            return M.expval(matrix_v, wires);
        },
        "Expected value of a matrix.");

    pyclass.def(
        "expval",
        [](Measurements<StateVectorT> &M, const ArraySparseIndT &row_map,
           const ArraySparseIndT &col_idx, const ArrayComplexT &values) {
            return M.expval(static_cast<SparseIndexT *>(row_map.data()),
                            static_cast<SparseIndexT>(row_map.size()),
                            static_cast<SparseIndexT *>(col_idx.data()),
                            static_cast<ComplexT *>(values.data()),
                            static_cast<SparseIndexT>(values.size()));
        },
        "Expected value of a sparse Hamiltonian.");

    pyclass.def(
        "var",
        [](Measurements<StateVectorT> &M, const std::string &name,
           const std::vector<std::size_t> &wires) {
            return M.var(name, wires);
        },
        "Variance of an operation by name.");

    pyclass.def(
        "var",
        [](Measurements<StateVectorT> &M, const ArraySparseIndT &row_map,
           const ArraySparseIndT &col_idx, const ArrayComplexT &values) {
            return M.var(static_cast<SparseIndexT *>(row_map.data()),
                         static_cast<SparseIndexT>(row_map.size()),
                         static_cast<SparseIndexT *>(col_idx.data()),
                         static_cast<ComplexT *>(values.data()),
                         static_cast<SparseIndexT>(values.size()));
        },
        "Variance of a sparse Hamiltonian.");

    pyclass.def(
        "generate_samples",
        [](Measurements<StateVectorT> &M, const std::vector<std::size_t> &wires,
           const std::size_t num_shots) {
            return createNumpyArrayFromVector<std::size_t>(
                M.generate_samples(wires, num_shots), num_shots, wires.size());
        },
        "Generate samples from the statevector.");

    pyclass.def(
        "generate_mcmc_samples",
        [](Measurements<StateVectorT> &M, std::size_t num_wires,
           const std::string &kernelname, std::size_t num_burnin,
           std::size_t num_shots) {
            return createNumpyArrayFromVector<std::size_t>(
                M.generate_samples_metropolis(kernelname, num_burnin,
                                              num_shots),
                num_shots, num_wires);
        },
        "Generate samples using MCMC.");
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

    using SparseIndexT = std::size_t;
    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;

    const std::string bitsize =
        std::is_same_v<PrecisionT, float> ? "64" : "128";

    std::string class_name;

    class_name = "SparseHamiltonianC" + bitsize;
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
            const ComplexT *data_ptr = data.data();
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
 * @brief Register Vector Jacobian Product.
 */
template <class StateVectorT>
auto registerVJP(
    VectorJacobianProduct<StateVectorT> &calculate_vjp, const StateVectorT &sv,
    const OpsData<StateVectorT> &operations,
    const nb::ndarray<const typename StateVectorT::ComplexT, nb::c_contig> &dy,
    const std::vector<std::size_t> &trainableParams)
    -> nb::ndarray<typename StateVectorT::ComplexT, nb::numpy, nb::c_contig> {
    using ComplexT = typename StateVectorT::ComplexT;
    std::vector<ComplexT> vjp(trainableParams.size(), ComplexT{});

    const JacobianData<StateVectorT> jd{operations.getTotalNumParams(),
                                        sv.getLength(),
                                        sv.getData(),
                                        {},
                                        operations,
                                        trainableParams};

    calculate_vjp(std::span{vjp}, jd, std::span{dy.data(), dy.size()});

    return createNumpyArrayFromVector<ComplexT>(std::move(vjp));
}

/**
 * @brief Register backend specific adjoint Jacobian methods.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendSpecificAlgorithms(nb::module_ &m) {
    using PrecisionT = typename StateVectorT::PrecisionT;

    const std::string bitsize =
        std::is_same_v<PrecisionT, float> ? "64" : "128";

    std::string class_name;

    //***********************************************************************//
    //                        Vector Jacobian Product
    //***********************************************************************//
    class_name = "VectorJacobianProductC" + bitsize;
    nb::class_<VectorJacobianProduct<StateVectorT>>(m, class_name.c_str())
        .def(nb::init<>())
        .def("__call__", &registerVJP<StateVectorT>,
             "Vector Jacobian Product method.");
}

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

            info["NAME"] = "lightning.qubit";

            return info;
        },
        "Backend-specific information.");
} // m

} // namespace Pennylane::LightningQubit::NanoBindings
