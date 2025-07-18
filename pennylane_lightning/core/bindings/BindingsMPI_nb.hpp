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
 * @file BindingsMPI_nb.hpp
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

#include "BindingsUtils_nb.hpp"
#include "JacobianData.hpp"

#if _ENABLE_PLGPU == 1

#include "AdjointJacobianGPUMPI.hpp"
#include "JacobianDataMPI.hpp"
#include "LGPUBindingsMPI.hpp"
#include "MeasurementsGPUMPI.hpp"
#include "ObservablesGPUMPI.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::Algorithms;
using namespace Pennylane::LightningGPU::Observables;
using namespace Pennylane::LightningGPU::Measures;
} // namespace
/// @endcond

#elif _ENABLE_PLKOKKOS == 1

#include "AdjointJacobianKokkosMPI.hpp"
#include "LKokkosBindingsMPI_nb.hpp"
#include "MeasurementsKokkosMPI.hpp"
#include "ObservablesKokkosMPI.hpp"

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::LightningKokkos::Measures;
} // namespace
/// @endcond

#else

static_assert(false, "Backend not found.");

#endif

namespace nb = nanobind;
namespace Pennylane::NanoBindings {

/**
 * @brief Register observable classes for MPI.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT> void registerObservablesMPI(nb::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision.
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type.

    const std::string bitsize =
        std::is_same_v<PrecisionT, float> ? "64" : "128";

    using arr_c = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;
    using ObservableT = Observable<StateVectorT>;
    using ObsPtr = std::shared_ptr<ObservableT>;

    std::string class_name;

    // Register Observable base class
    class_name = "ObservableMPIC" + bitsize;
    nb::class_<ObservableT>(m, class_name.c_str());

    // Register NamedObsMPI class
    class_name = "NamedObsMPIC" + bitsize;
    nb::class_<NamedObsMPI<StateVectorT>, ObservableT>(m, class_name.c_str())
        .def(nb::init<const std::string &, const std::vector<std::size_t> &>())
        .def("__repr__", &NamedObsMPI<StateVectorT>::getObsName)
        .def("get_wires", &NamedObsMPI<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const NamedObsMPI<StateVectorT> &self,
               nb::handle other) -> bool {
                if (!nb::isinstance<NamedObsMPI<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = nb::cast<NamedObsMPI<StateVectorT>>(other);
                return self == other_cast;
            },
            "Compare two observables");

    // Register HermitianObsMPI class
    class_name = "HermitianObsMPIC" + bitsize;
    nb::class_<HermitianObsMPI<StateVectorT>, ObservableT>(m,
                                                           class_name.c_str())
        .def(nb::init<const std::vector<ComplexT> &,
                      const std::vector<std::size_t> &>())
        .def("__repr__", &HermitianObsMPI<StateVectorT>::getObsName)
        .def("get_wires", &HermitianObsMPI<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const HermitianObsMPI<StateVectorT> &self,
               nb::handle other) -> bool {
                if (!nb::isinstance<HermitianObsMPI<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast =
                    nb::cast<HermitianObsMPI<StateVectorT>>(other);
                return self == other_cast;
            },
            "Compare two observables");

    // Register TensorProdObsMPI class
    class_name = "TensorProdObsMPIC" + bitsize;
    nb::class_<TensorProdObsMPI<StateVectorT>, ObservableT>(m,
                                                            class_name.c_str())
        .def(nb::init<const std::vector<ObsPtr> &>())
        .def("__repr__", &TensorProdObsMPI<StateVectorT>::getObsName)
        .def("get_wires", &TensorProdObsMPI<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const TensorProdObsMPI<StateVectorT> &self,
               nb::handle other) -> bool {
                if (!nb::isinstance<TensorProdObsMPI<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast =
                    nb::cast<TensorProdObsMPI<StateVectorT>>(other);
                return self == other_cast;
            },
            "Compare two observables");

    // Register HamiltonianMPI class
    class_name = "HamiltonianMPIC" + bitsize;
    nb::class_<HamiltonianMPI<StateVectorT>, ObservableT>(m, class_name.c_str())
        .def(nb::init<const std::vector<PrecisionT> &,
                      const std::vector<ObsPtr> &>())
        .def("__repr__", &HamiltonianMPI<StateVectorT>::getObsName)
        .def("get_wires", &HamiltonianMPI<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const HamiltonianMPI<StateVectorT> &self,
               nb::handle other) -> bool {
                if (!nb::isinstance<HamiltonianMPI<StateVectorT>>(other)) {
                    return false;
                }
                auto other_cast = nb::cast<HamiltonianMPI<StateVectorT>>(other);
                return self == other_cast;
            },
            "Compare two observables");

#if _ENABLE_PLGPU == 1
    using SparseIndexT =
        typename std::conditional<std::is_same<PrecisionT, float>::value,
                                  int32_t, int64_t>::type;

    // Register SparseHamiltonianMPI class
    class_name = "SparseHamiltonianMPIC" + bitsize;
    nb::class_<SparseHamiltonianMPI<StateVectorT>, ObservableT>(
        m, class_name.c_str())
        .def(nb::init<const std::vector<ComplexT> &,
                      const std::vector<SparseIndexT> &,
                      const std::vector<SparseIndexT> &,
                      const std::vector<std::size_t> &>())
        .def("__repr__", &SparseHamiltonianMPI<StateVectorT>::getObsName)
        .def("get_wires", &SparseHamiltonianMPI<StateVectorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const SparseHamiltonianMPI<StateVectorT> &self,
               nb::handle other) -> bool {
                if (!nb::isinstance<SparseHamiltonianMPI<StateVectorT>>(
                        other)) {
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
             });
}

/**
 * @brief Register the adjoint Jacobian method.
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

    using arr_c = nb::ndarray<std::complex<PrecisionT>, nb::c_contig>;

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
           const std::vector<arr_c> &ops_matrices,
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
    nb::class_<MPIManager>(m, "MPIManager")
        .def(nb::init<>())
        .def(nb::init<MPIManager &>())
        .def("Barrier", &MPIManager::Barrier)
        .def("getRank", &MPIManager::getRank)
        .def("getSize", &MPIManager::getSize)
        .def("getSizeNode", &MPIManager::getSizeNode)
        .def("getTime", &MPIManager::getTime)
        .def("getVendor", &MPIManager::getVendor)
        .def("getVersion", &MPIManager::getVersion)
        // Template version with explicit type constraints
        .def(
            "Scatter",
            []<typename PrecisionT>(
                MPIManager &mpi_manager,
                nb::ndarray<std::complex<PrecisionT>, nb::c_contig> &sendBuf,
                nb::ndarray<std::complex<PrecisionT>, nb::c_contig> &recvBuf,
                int root) {
                auto send_ptr = sendBuf.data();
                auto recv_ptr = recvBuf.data();
                mpi_manager.template Scatter<std::complex<PrecisionT>>(
                    send_ptr, recv_ptr, recvBuf.size(), root);
            },
            "MPI Scatter for complex arrays.");
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
    registerBackendAgnosticStateVectorMethods<StateVectorT>(pyclass);
    registerBackendSpecificStateVectorMethods<StateVectorT>(pyclass);

    // Register backend specific bindings
    registerBackendClassSpecificBindingsMPI<StateVectorT>(pyclass);

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

    pyclass_measurements.def(nb::init<const StateVectorT &>());
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
        nb::exception<Pennylane::Util::LightningException>(
            m, "LightningExceptionMPI");
    }
}

} // namespace Pennylane::NanoBindings
