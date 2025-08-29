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
 * @file BindingsMPI.hpp
 * Defines device-agnostic operations to export to Python and other utility
 * functions interfacing with Nanobind.
 */

#pragma once
#include <complex>
#include <span>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "BindingsUtils.hpp"
#include "JacobianData.hpp"

#if _ENABLE_PLGPU == 1

#include "AdjointJacobianGPUMPI.hpp"
#include "JacobianDataMPI.hpp"
#include "LGPUBindingsMPI.hpp"
#include "MPIManagerGPU.hpp"
#include "MeasurementsGPUMPI.hpp"
#include "ObservablesGPUMPI.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::Algorithms;
using namespace Pennylane::LightningGPU::Observables;
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::Util;

} // namespace
/// @endcond

#elif _ENABLE_PLKOKKOS == 1

#include "AdjointJacobianKokkosMPI.hpp"
#include "LKokkosBindingsMPI.hpp"
#include "MPIManagerKokkos.hpp"
#include "MeasurementsKokkosMPI.hpp"
#include "ObservablesKokkosMPI.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::Util;

} // namespace
/// @endcond

#else

static_assert(false, "Backend not found.");

#endif

namespace Pennylane::NanoBindings {

namespace nb = nanobind;

/**
 * @brief Register observable classes for MPI.
 *
 * Register Observable implementations for MPI.
 *
 * @tparam StateVectorT The type of the state vector.
 * @param m Nanobind module.
 */
template <class StateVectorT> void registerObservablesMPI(nb::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type.

    const std::string bitsize =
        std::is_same_v<PrecisionT, float> ? "64" : "128";

    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;
    using ObservableT = Observable<StateVectorT>;
    using ObsPtr = std::shared_ptr<ObservableT>;
    using NamedObsT = NamedObsMPI<StateVectorT>;
    using HermitianObsT = HermitianObsMPI<StateVectorT>;
    using TensorProdObsT = TensorProdObsMPI<StateVectorT>;
    using HamiltonianT = HamiltonianMPI<StateVectorT>;

    std::string class_name;

    // Register Observable base class
    class_name = "ObservableMPIC" + bitsize;
    nb::class_<ObservableT>(m, class_name.c_str());

    // Register NamedObsMPI class
    class_name = "NamedObsMPIC" + bitsize;
    auto named_obs_class =
        nb::class_<NamedObsT, ObservableT>(m, class_name.c_str());
    named_obs_class.def(
        nb::init<const std::string &, const std::vector<std::size_t> &>());
    named_obs_class.def("__repr__", &NamedObsT::getObsName);
    named_obs_class.def("get_wires", &NamedObsT::getWires,
                        "Get wires of observables");
    named_obs_class.def(
        "__eq__",
        [](const NamedObsT &self, const NamedObsT &other) -> bool {
            return self == other;
        },
        "Compare two observables");

    // Register HermitianObsMPI class
    class_name = "HermitianObsMPIC" + bitsize;
    auto hermitian_obs_class =
        nb::class_<HermitianObsT, ObservableT>(m, class_name.c_str());
    hermitian_obs_class.def(
        "__init__", [](HermitianObsT *self, const ArrayComplexT &matrix,
                       const std::vector<std::size_t> &wires) {
            const auto ptr = matrix.data();
            new (self) HermitianObsT(
                std::vector<ComplexT>(ptr, ptr + matrix.size()), wires);
        });
    hermitian_obs_class.def("__repr__", &HermitianObsT::getObsName);
    hermitian_obs_class.def("get_wires", &HermitianObsT::getWires,
                            "Get wires of observables");
    hermitian_obs_class.def(
        "__eq__",
        [](const HermitianObsT &self, const HermitianObsT &other) -> bool {
            return self == other;
        },
        "Compare two observables");

    // Register TensorProdObsMPI class
    class_name = "TensorProdObsMPIC" + bitsize;
    auto tensor_prod_obs_class =
        nb::class_<TensorProdObsT, ObservableT>(m, class_name.c_str());
    tensor_prod_obs_class.def(nb::init<const std::vector<ObsPtr> &>());
    tensor_prod_obs_class.def("__repr__", &TensorProdObsT::getObsName);
    tensor_prod_obs_class.def("get_wires", &TensorProdObsT::getWires,
                              "Get wires of observables");
    tensor_prod_obs_class.def(
        "__eq__",
        [](const TensorProdObsT &self, const TensorProdObsT &other) -> bool {
            return self == other;
        },
        "Compare two observables");

    // Register HamiltonianMPI class
    class_name = "HamiltonianMPIC" + bitsize;
    auto hamiltonian_class =
        nb::class_<HamiltonianT, ObservableT>(m, class_name.c_str());
    hamiltonian_class.def(nb::init<const std::vector<PrecisionT> &,
                                   const std::vector<ObsPtr> &>());
    hamiltonian_class.def(
        "__init__", [](HamiltonianT *self,
                       const nb::ndarray<PrecisionT, nb::c_contig> &coeffs,
                       const std::vector<ObsPtr> &obs) {
            const auto ptr = coeffs.data();
            new (self) HamiltonianT(
                std::vector<PrecisionT>(ptr, ptr + coeffs.size()), obs);
        });
    hamiltonian_class.def("__repr__", &HamiltonianT::getObsName);
    hamiltonian_class.def("get_wires", &HamiltonianT::getWires,
                          "Get wires of observables");
    hamiltonian_class.def("get_coeffs", &HamiltonianT::getCoeffs,
                          "Get coefficients");
    hamiltonian_class.def("get_ops", &HamiltonianT::getObs,
                          "Get operations list");
    hamiltonian_class.def(
        "__eq__",
        [](const HamiltonianT &self, const HamiltonianT &other) -> bool {
            return self == other;
        },
        "Compare two observables");

#if _ENABLE_PLGPU == 1
    using SparseIndexT =
        typename std::conditional<std::is_same<PrecisionT, float>::value,
                                  int32_t, int64_t>::type;

    // Register SparseHamiltonianMPI class
    class_name = "SparseHamiltonianMPIC" + bitsize;
    auto sparse_hamiltonian_class =
        nb::class_<SparseHamiltonianMPI<StateVectorT>, ObservableT>(
            m, class_name.c_str());
    sparse_hamiltonian_class.def(nb::init<const std::vector<ComplexT> &,
                                          const std::vector<SparseIndexT> &,
                                          const std::vector<SparseIndexT> &,
                                          const std::vector<std::size_t> &>());
    sparse_hamiltonian_class.def(
        "__repr__", &SparseHamiltonianMPI<StateVectorT>::getObsName);
    sparse_hamiltonian_class.def("get_wires",
                                 &SparseHamiltonianMPI<StateVectorT>::getWires,
                                 "Get wires of observables");
    sparse_hamiltonian_class.def(
        "__eq__",
        [](const SparseHamiltonianMPI<StateVectorT> &self,
           nb::handle other) -> bool {
            if (!nb::isinstance<SparseHamiltonianMPI<StateVectorT>>(other)) {
                return false;
            }
            auto other_cast =
                nb::cast<SparseHamiltonianMPI<StateVectorT>>(other);
            return self == other_cast;
        },
        "Compare two observables");
#endif
}

/**
 * @brief Register backend-agnostic measurement class functionalities for MPI.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's measurements class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendAgnosticMeasurementsMPI(PyClass &pyclass) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ObsPtr = std::shared_ptr<Observable<StateVectorT>>;

    pyclass
        .def("probs",
             [](MeasurementsMPI<StateVectorT> &M,
                const std::vector<std::size_t> &wires) {
                 return createNumpyArrayFromVector<PrecisionT>(M.probs(wires));
             })
        .def("probs",
             [](MeasurementsMPI<StateVectorT> &M) {
                 return createNumpyArrayFromVector<PrecisionT>(M.probs());
             })
        .def(
            "expval",
            [](MeasurementsMPI<StateVectorT> &M, const ObsPtr &ob) {
                return M.expval(*ob);
            },
            "Expected value of an observable object.")
        .def(
            "var",
            [](MeasurementsMPI<StateVectorT> &M, const ObsPtr &ob) {
                return M.var(*ob);
            },
            "Variance of an observable object.")
        .def("generate_samples",
             [](MeasurementsMPI<StateVectorT> &M, std::size_t num_wires,
                std::size_t num_shots) {
                 return createNumpyArrayFromVector<std::size_t>(
                     M.generate_samples(num_shots), num_shots, num_wires);
             })
        .def("set_random_seed", [](MeasurementsMPI<StateVectorT> &M,
                                   std::size_t seed) { M.setSeed(seed); });
}

/**
 * @brief Register the adjoint Jacobian method.
 *
 * Register the adjoint Jacobian method for the given state vector.
 *
 * @tparam StateVectorT The type of the state vector.
 * @param adjoint_jacobian The adjoint Jacobian object.
 * @param sv The state vector.
 * @param observables The observables.
 * @param operations The operations.
 */
template <class StateVectorT>
auto registerAdjointJacobianMPI(
    AdjointJacobianMPI<StateVectorT> &adjoint_jacobian, const StateVectorT &sv,
    const std::vector<std::shared_ptr<Observable<StateVectorT>>> &observables,
    const OpsData<StateVectorT> &operations,
    const std::vector<std::size_t> &trainableParams)
    -> nb::ndarray<typename StateVectorT::PrecisionT, nb::numpy, nb::c_contig> {
    using PrecisionT = typename StateVectorT::PrecisionT;
    std::vector<PrecisionT> jac(observables.size() * trainableParams.size(),
                                PrecisionT{0.0});
#if _ENABLE_PLGPU == 1
    const JacobianDataMPI<StateVectorT> jd{operations.getTotalNumParams(), sv,
                                           observables, operations,
                                           trainableParams};
#elif _ENABLE_PLKOKKOS == 1
    const JacobianData<StateVectorT> jd{operations.getTotalNumParams(),
                                        sv.getLength(),
                                        sv.getData(),
                                        observables,
                                        operations,
                                        trainableParams};
#endif
    adjoint_jacobian.adjointJacobian(std::span{jac}, jd, sv);

    return createNumpyArrayFromVector<PrecisionT>(std::move(jac));
}

/**
 * @brief Register backend-agnostic algorithms.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendAgnosticAlgorithmsMPI(nb::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type

    using ArrayComplexT = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;

    const std::string bitsize =
        std::is_same_v<PrecisionT, float> ? "64" : "128";

    std::string class_name;

    //***********************************************************************//
    //                              Operations
    //***********************************************************************//

    class_name = "OpsStructMPIC" + bitsize;
    nb::class_<OpsData<StateVectorT>>(m, class_name.c_str())
        .def(nb::init<const std::vector<std::string> &,
                      const std::vector<std::vector<PrecisionT>> &,
                      const std::vector<std::vector<std::size_t>> &,
                      const std::vector<bool> &,
                      const std::vector<std::vector<ComplexT>> &>())
        .def("__repr__", [](const OpsData<StateVectorT> &ops) {
            using namespace Pennylane::Util;
            std::ostringstream ops_stream;
            for (std::size_t op = 0; op < ops.getSize(); op++) {
                ops_stream << "{'name': " << ops.getOpsName()[op];
                ops_stream << ", 'params': " << ops.getOpsParams()[op];
                ops_stream << ", 'inv': " << ops.getOpsInverses()[op];
                ops_stream << "}";
                if (op < ops.getSize() - 1) {
                    ops_stream << ",";
                }
            }
            return "Operations: [" + ops_stream.str() + "]";
        });

    /**
     * Create operation list.
     */
    std::string function_name = "create_ops_listMPIC" + bitsize;
    m.def(
        function_name.c_str(),
        [](const std::vector<std::string> &ops_name,
           const std::vector<std::vector<PrecisionT>> &ops_params,
           const std::vector<std::vector<std::size_t>> &ops_wires,
           const std::vector<bool> &ops_inverses,
           const std::vector<ArrayComplexT> &ops_matrices,
           const std::vector<std::vector<std::size_t>> &ops_controlled_wires,
           const std::vector<std::vector<bool>> &ops_controlled_values) {
            std::vector<std::vector<ComplexT>> conv_matrices =
                Pennylane::NanoBindings::Utils::convertMatrices<ComplexT,
                                                                PrecisionT>(
                    ops_matrices);
            return OpsData<StateVectorT>{ops_name,
                                         ops_params,
                                         ops_wires,
                                         ops_inverses,
                                         conv_matrices,
                                         ops_controlled_wires,
                                         ops_controlled_values};
        },
        "Create a list of operations from data.");

    //***********************************************************************//
    //                            Adjoint Jacobian MPI
    //***********************************************************************//
    class_name = "AdjointJacobianMPIC" + bitsize;
    nb::class_<AdjointJacobianMPI<StateVectorT>>(m, class_name.c_str())
        .def(nb::init<>())
        .def("__call__", &registerAdjointJacobianMPI<StateVectorT>,
             "Adjoint Jacobian method.")
        .def(
            "batched",
            [](AdjointJacobianMPI<StateVectorT> &self, const StateVectorT &sv,
               const std::vector<std::shared_ptr<Observable<StateVectorT>>>
                   &observables,
               const OpsData<StateVectorT> &operations,
               const std::vector<std::size_t> &trainableParams) {
                using PrecisionT = typename StateVectorT::PrecisionT;
                std::vector<PrecisionT> jac(observables.size() *
                                                trainableParams.size(),
                                            PrecisionT{0.0});
#if _ENABLE_PLGPU == 1
                const JacobianDataMPI<StateVectorT> jd{
                    operations.getTotalNumParams(), sv, observables, operations,
                    trainableParams};
                self.adjointJacobian_serial(std::span{jac}, jd);
#elif _ENABLE_PLKOKKOS == 1
                const JacobianData<StateVectorT> jd{
                    operations.getTotalNumParams(),
                    sv.getLength(),
                    sv.getData(),
                    observables,
                    operations,
                    trainableParams};
                self.adjointJacobian(std::span{jac}, jd, sv);
#endif
                return createNumpyArrayFromVector<PrecisionT>(std::move(jac));
            },
            "Batch Adjoint Jacobian method.");
}

/**
 * @brief Register backend-agnostic MPI.
 *
 * @param m Nanobind module
 */
inline void registerInfoMPI(nb::module_ &m) {
    // This function is now empty - MPI manager registration moved to
    // backend-specific
}

/**
 * @brief Templated class to build lightning MPI class bindings.
 *
 * @tparam StateVectorT State vector type
 * @param m Nanobind module.
 */
template <class StateVectorT> void lightningClassBindingsMPI(nb::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::is_same_v<PrecisionT, float> ? "64" : "128";

    //***********************************************************************//
    //                              StateVector
    //***********************************************************************//
    std::string class_name = "StateVectorMPIC" + bitsize;
    auto pyclass = nb::class_<StateVectorT>(m, class_name.c_str());

    // Register backend specific bindings
    registerBackendSpecificStateVectorMethodsMPI<StateVectorT>(pyclass);

    //***********************************************************************//
    //                              Observables
    //***********************************************************************//

    nb::module_ obs_submodule =
        m.def_submodule("observablesMPI", "Submodule for observables classes.");
    registerObservablesMPI<StateVectorT>(obs_submodule);

    //***********************************************************************//
    //                              Measurements
    //***********************************************************************//

    class_name = "MeasurementsMPIC" + bitsize;
    auto pyclass_measurements =
        nb::class_<MeasurementsMPI<StateVectorT>>(m, class_name.c_str());

    pyclass_measurements.def(nb::init<StateVectorT &>());
    registerBackendAgnosticMeasurementsMPI<StateVectorT>(pyclass_measurements);
    registerBackendSpecificMeasurementsMPI<StateVectorT>(pyclass_measurements);

    //***********************************************************************//
    //                              Algorithms
    //***********************************************************************//

    nb::module_ alg_submodule = m.def_submodule(
        "algorithmsMPI", "Submodule for the algorithms functionality.");
    registerBackendAgnosticAlgorithmsMPI<StateVectorT>(alg_submodule);
    registerBackendSpecificAlgorithmsMPI<StateVectorT>(alg_submodule);
}

template <typename TypeList>
void registerLightningClassBindingsMPI(nb::module_ &m) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        lightningClassBindingsMPI<StateVectorT>(m);
        registerLightningClassBindingsMPI<typename TypeList::Next>(m);
    }
}

} // namespace Pennylane::NanoBindings
