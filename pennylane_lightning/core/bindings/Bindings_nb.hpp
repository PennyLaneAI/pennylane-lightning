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
 * @file Bindings_nb.hpp
 * Defines device-agnostic operations to export to Python and other utility
 * functions interfacing with Nanobind.
 */

#pragma once
#include <complex>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "BindingsUtils_nb.hpp"
#include "CPUMemoryModel.hpp" // CPUMemoryModel, bestCPUMemoryModel
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "GateOperation.hpp"
#include "JacobianData.hpp"
#include "Macros.hpp" // CPUArch
#include "Memory.hpp" // alignedAlloc
#include "Observables.hpp"
#include "Util.hpp" // for_each_enum, PL_reinterpret_cast

// Include backend-specific headers and define macros based on compile flags
#ifdef _ENABLE_PLQUBIT
#include "AdjointJacobianLQubit.hpp"
#include "LQubitBindings_nb.hpp"
#include "MeasurementsLQubit.hpp"
#include "ObservablesLQubit.hpp"

#define LIGHTNING_MODULE_NAME lightning_qubit_nb

/// @cond DEV
namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Algorithms;
using namespace Pennylane::LightningQubit::Observables;
using namespace Pennylane::LightningQubit::Measures;
using namespace Pennylane::LightningQubit::NanoBindings;
} // namespace
/// @endcond

#elif _ENABLE_PLKOKKOS == 1
#include "AdjointJacobianKokkos.hpp"
#include "LKokkosBindings_nb.hpp"
#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"

#define LIGHTNING_MODULE_NAME lightning_kokkos_nb

/// @cond DEV
namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::NanoBindings;
} // namespace
/// @endcond

#elif _ENABLE_PLGPU == 1
#include "AdjointJacobianGPU.hpp"
#include "BindingsCudaUtils_nb.hpp"
#include "LGPUBindings_nb.hpp"
#include "MeasurementsGPU.hpp"
#include "ObservablesGPU.hpp"

#define LIGHTNING_MODULE_NAME lightning_gpu_nb

/// @cond DEV
namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::Algorithms;
using namespace Pennylane::LightningGPU::Observables;
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::NanoBindings;
} // namespace
/// @endcond

#elif _ENABLE_PLTENSOR == 1
#include "LTensorTNCudaBindings_nb.hpp"
#include "MeasurementsTNCuda.hpp"
#include "ObservablesTNCuda.hpp"

#define LIGHTNING_TENSOR_MODULE_NAME lightning_tensor_nb

/// @cond DEV
namespace {
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using namespace Pennylane::LightningTensor::TNCuda::Measures;
using namespace Pennylane::LightningTensor::TNCuda::NanoBindings;
} // namespace
/// @endcond

#else
static_assert(false, "Backend not found.");
#endif

namespace nb = nanobind;

/// @cond DEV
namespace {
using Pennylane::NanoBindings::Utils::createNumpyArrayFromVector;
using Pennylane::Util::bestCPUMemoryModel;
using Pennylane::Util::CPUMemoryModel;
using Pennylane::Util::PL_reinterpret_cast;
} // namespace
/// @endcond

namespace Pennylane::NanoBindings {
/**
 * @brief Register applyMatrix
 */
template <class StateReprT>
void applyMatrix(
    StateReprT &st,
    const nb::ndarray<const std::complex<typename StateReprT::PrecisionT>,
                      nb::c_contig> &matrix,
    const std::vector<std::size_t> &wires, bool inverse = false) {
    using ComplexT = typename StateReprT::ComplexT;

    PL_ASSERT(matrix.size() == Util::exp2(2 * wires.size()));

    // Cast to raw pointer
    auto *data_ptr = PL_reinterpret_cast<const ComplexT>(matrix.data());
    st.applyMatrix(data_ptr, wires, inverse);
}

/**
 * @brief Register controlled matrix kernel.
 */
template <class StateReprT>
void applyControlledMatrix(
    StateReprT &st,
    const nb::ndarray<const std::complex<typename StateReprT::PrecisionT>,
                      nb::c_contig> &matrix,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &wires, bool inverse = false) {
    using ComplexT = typename StateReprT::ComplexT;
#ifdef _ENABLE_PLTENSOR
    std::vector<ComplexT> conv_matrix;
    if (matrix.size()) {
        conv_matrix =
            std::vector<ComplexT>{matrix.data(), matrix.data() + matrix.size()};
    }

    st.applyControlledOperation("applyControlledMatrix", controlled_wires,
                                controlled_values, wires, inverse, {},
                                conv_matrix);
#else
    st.applyControlledMatrix(PL_reinterpret_cast<const ComplexT>(matrix.data()),
                             controlled_wires, controlled_values, wires,
                             inverse);
#endif
}

/**
 * @brief Register gates for a given backend.
 *
 * @tparam StateReprT The type used to represent the state (statevector,
 * tensornet, etc.)
 * @tparam PyClass Nanobind's class object type
 *
 * @param pyclass Nanobind's class object to bind statevector
 */
template <class StateReprT, class PyClass>
void registerGates(PyClass &pyclass) {
    using PrecisionT = typename StateReprT::PrecisionT; // State's precision
    using ParamT = PrecisionT; // Parameter's data precision

    using Pennylane::Gates::GateOperation;
    using Pennylane::Util::for_each_enum;
    namespace Constant = Pennylane::Gates::Constant;

    pyclass.def("applyMatrix", &applyMatrix<StateReprT>,
                "Apply a given matrix to wires.");
    pyclass.def("applyControlledMatrix", &applyControlledMatrix<StateReprT>,
                "Apply controlled operation");

    for_each_enum<GateOperation>([&pyclass](GateOperation gate_op) {
        using Pennylane::Util::lookup;
        const auto gate_name =
            std::string(lookup(Constant::gate_names, gate_op));
        const std::string doc = "Apply the " + gate_name + " gate.";
        auto func =
            [gate_name](StateReprT &st, const std::vector<std::size_t> &wires,
                        bool inverse, const std::vector<ParamT> &params) {
                st.applyOperation(gate_name, wires, inverse, params);
            };
        pyclass.def(gate_name.c_str(), func, doc.c_str());
    });
}

/**
 * @brief Register controlled gate operations for a statevector
 *
 * @tparam StateReprT State vector type
 * @tparam PyClass Nanobind class type
 * @param pyclass Nanobind class to bind methods to
 */
template <class StateReprT, class PyClass>
void registerControlledGates(PyClass &pyclass) {
    using PrecisionT = typename StateReprT::PrecisionT;
    using ParamT = PrecisionT;

    using Pennylane::Gates::ControlledGateOperation;
    using Pennylane::Util::for_each_enum;
    namespace Constant = Pennylane::Gates::Constant;

    for_each_enum<ControlledGateOperation>(
        [&pyclass](ControlledGateOperation gate_op) {
            using Pennylane::Util::lookup;
            const auto gate_name =
                std::string(lookup(Constant::controlled_gate_names, gate_op));
            const std::string doc = "Apply the " + gate_name + " gate.";
            auto func = [gate_name = gate_name](
                            StateReprT &st,
                            const std::vector<std::size_t> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<std::size_t> &wires, bool inverse,
                            const std::vector<ParamT> &params) {
                st.applyOperation(gate_name, controlled_wires,
                                  controlled_values, wires, inverse, params);
            };
            pyclass.def(gate_name.c_str(), func, doc.c_str(),
                        nb::arg("controlled_wires"),
                        nb::arg("controlled_values"), nb::arg("wires"),
                        nb::arg("inverse") = false,
                        nb::arg("params") = std::vector<ParamT>{});
        });
}

/**
 * @brief Return basic information of runtime environment
 */
nb::dict getRuntimeInfo() {
    using Pennylane::Util::RuntimeInfo;

    nb::dict info;
    info["binding_type"] = "nanobind";
    info["AVX"] = RuntimeInfo::AVX();
    info["AVX2"] = RuntimeInfo::AVX2();
    info["AVX512F"] = RuntimeInfo::AVX512F();

    return info;
}

/**
 * @brief Get compile information as a dictionary
 *
 * @return Dictionary with compile information
 */
nb::dict getCompileInfo() {
    using namespace Pennylane::Util;

    // Convert string_view to std::string
    std::string cpu_arch_str;
    switch (cpu_arch) {
    case CPUArch::X86_64:
        cpu_arch_str = "x86_64";
        break;
    case CPUArch::PPC64:
        cpu_arch_str = "PPC64";
        break;
    case CPUArch::ARM:
        cpu_arch_str = "ARM";
        break;
    default:
        cpu_arch_str = "Unknown";
        break;
    }

    std::string compiler_name_str;
    switch (compiler) {
    case Compiler::GCC:
        compiler_name_str = "GCC";
        break;
    case Compiler::Clang:
        compiler_name_str = "Clang";
        break;
    case Compiler::MSVC:
        compiler_name_str = "MSVC";
        break;
    case Compiler::NVCC:
        compiler_name_str = "NVCC";
        break;
    case Compiler::NVHPC:
        compiler_name_str = "NVHPC";
        break;
    default:
        compiler_name_str = "Unknown";
        break;
    }

    std::string compiler_version_str =
        std::string(getCompilerVersion<compiler>());

    nb::dict info;
    info["cpu.arch"] = cpu_arch_str;
    info["compiler.name"] = compiler_name_str;
    info["compiler.version"] = compiler_version_str;
    info["AVX2"] = use_avx2;
    info["AVX512F"] = use_avx512f;

    return info;
}

#ifndef _ENABLE_PLTENSOR
// These functions are used solely by the statevector simulators

/**
 * @brief Create an aligned array for a given type, memory model and array size
 *
 * @tparam VectorT Datatype of array to create
 * @param memory_model Memory model to use
 * @param size Size of the array to create
 * @return nb::ndarray<VectorT, nb::numpy, nb::c_contig>
 */
template <typename VectorT>
auto alignedArray(CPUMemoryModel memory_model, std::size_t size, bool zeroInit)
    -> nb::ndarray<VectorT, nb::numpy, nb::c_contig> {
    using Pennylane::Util::alignedAlloc;
    using Pennylane::Util::getAlignment;

    // Allocate aligned memory
    void *ptr = alignedAlloc(getAlignment<VectorT>(memory_model),
                             sizeof(VectorT) * size, zeroInit);

    // Create capsule with custom deleter
    auto capsule =
        nb::capsule(ptr, [](void *p) noexcept { Util::alignedFree(p); });

    std::vector<size_t> shape{size};

    // Return ndarray with custom allocated memory
    return nb::ndarray<VectorT, nb::numpy, nb::c_contig>(ptr, shape.size(),
                                                         shape.data(), capsule);
}

/**
 * @brief Allocate aligned array with specified dtype
 *
 * @param size Size of the array to create
 * @param dtype Python dtype object to create the array with
 * @param zeroInit Whether to initialize the array with zeros
 * @return nb::object a general nanobind object that can assume any of the types
 * in the method.
 */
auto allocateAlignedArray(std::size_t size, const nb::object &dtype,
                          bool zeroInit = false) -> nb::object {
    auto memory_model = bestCPUMemoryModel();

    // Convert dtype to string representation
    std::string dtype_str = nb::cast<std::string>(dtype.attr("name"));

    if (dtype_str == "complex64") {
        return nb::cast(
            alignedArray<std::complex<float>>(memory_model, size, zeroInit));
    } else if (dtype_str == "complex128") {
        return nb::cast(
            alignedArray<std::complex<double>>(memory_model, size, zeroInit));
    } else if (dtype_str == "float32") {
        return nb::cast(alignedArray<float>(memory_model, size, zeroInit));
    } else if (dtype_str == "float64") {
        return nb::cast(alignedArray<double>(memory_model, size, zeroInit));
    }

    throw std::runtime_error("Unsupported dtype: " + dtype_str);
}

/**
 * @brief Register array alignment functionality
 *
 * @param m Nanobind module
 */
void registerArrayAlignmentBindings(nb::module_ &m) {
    // Add allocate_aligned_array function
    m.def("allocate_aligned_array", &allocateAlignedArray,
          "Allocate aligned array with specified dtype", nb::arg("size"),
          nb::arg("dtype"), nb::arg("zero_init") = false);
}
#endif // ifndef _ENABLE_PLTENSOR

/**
 * @brief Register bindings for general info
 *
 * @param m Nanobind module
 */
void registerInfo(nb::module_ &m) {
    // Add compile info
    m.def("compile_info", &getCompileInfo, "Compiled binary information.");

    // Add runtime info
    m.def("runtime_info", &getRuntimeInfo, "Runtime information.");
}

/**
 * @brief Register backend-agnostic observables
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendAgnosticObservables(nb::module_ &m) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using ParamT = PrecisionT;

    using nd_arr_c = nb::ndarray<const std::complex<ParamT>, nb::c_contig>;

    const std::string bitsize =
        std::is_same_v<PrecisionT, float> ? "64" : "128";

#ifdef _ENABLE_PLTENSOR
    using ObservableT = ObservableTNCuda<StateVectorT>;
    using NamedObsT = NamedObsTNCuda<StateVectorT>;
    using HermitianObsT = HermitianObsTNCuda<StateVectorT>;
    using TensorProdObsT = TensorProdObsTNCuda<StateVectorT>;
    using HamiltonianT = HamiltonianTNCuda<StateVectorT>;
#else
    using ObservableT = Observable<StateVectorT>;
    using NamedObsT = NamedObs<StateVectorT>;
    using HermitianObsT = HermitianObs<StateVectorT>;
    using TensorProdObsT = TensorProdObs<StateVectorT>;
    using HamiltonianT = Hamiltonian<StateVectorT>;
#endif

    using ObsPtr = std::shared_ptr<ObservableT>;

    std::string class_name;

    // Register Observable base class
    class_name = "ObservableC" + bitsize;
    auto obs_class = nb::class_<ObservableT>(m, class_name.c_str());
    obs_class.def("get_wires", &ObservableT::getWires,
                  "Get wires the observable acts on.");

    // Register NamedObs class
    class_name = "NamedObsC" + bitsize;
    nb::class_<NamedObsT>(m, class_name.c_str(), obs_class)
        .def(nb::init<const std::string &, const std::vector<std::size_t> &>())
        .def("__repr__", &NamedObsT::getObsName)
        .def("get_wires", &NamedObsT::getWires, "Get wires of observables")
        .def(
            "__eq__",
            [](const NamedObsT &self, const NamedObsT &other) -> bool {
                return self == other;
            },
            "Compare two observables");

    // Register HermitianObs class
    class_name = "HermitianObsC" + bitsize;
    nb::class_<HermitianObsT, ObservableT>(m, class_name.c_str())
        .def("__init__",
             [](HermitianObsT *self, const nd_arr_c &matrix,
                const std::vector<std::size_t> &wires) {
                 const auto ptr = matrix.data();
                 new (self) HermitianObsT(
                     std::vector<ComplexT>(ptr, ptr + matrix.size()), wires);
             })
        .def("__repr__", &HermitianObsT::getObsName)
        .def("get_wires", &HermitianObsT::getWires, "Get wires of observables")
        .def("get_matrix", &HermitianObsT::getMatrix,
             "Get matrix representation of Hermitian operator")
        .def(
            "__eq__",
            [](const HermitianObsT &self, const HermitianObsT &other) -> bool {
                return self == other;
            },
            "Compare two observables");

    // Register TensorProdObs class
    class_name = "TensorProdObsC" + bitsize;
    nb::class_<TensorProdObsT, ObservableT>(m, class_name.c_str())
        .def(nb::init<const std::vector<ObsPtr> &>())
        .def("__repr__", &TensorProdObsT::getObsName)
        .def("get_wires", &TensorProdObsT::getWires, "Get wires of observables")
        .def("get_ops", &TensorProdObsT::getObs, "Get operations list")
        .def(
            "__eq__",
            [](const TensorProdObsT &self,
               const TensorProdObsT &other) -> bool { return self == other; },
            "Compare two observables");

    // Register Hamiltonian class
    class_name = "HamiltonianC" + bitsize;
    nb::class_<HamiltonianT, ObservableT>(m, class_name.c_str())
        .def(nb::init<const std::vector<ParamT> &,
                      const std::vector<ObsPtr> &>())
        .def("__init__",
             [](HamiltonianT *self,
                const nb::ndarray<ParamT, nb::c_contig> &coeffs,
                const std::vector<ObsPtr> &obs) {
                 const auto ptr = coeffs.data();
                 new (self) HamiltonianT(
                     std::vector<ParamT>(ptr, ptr + coeffs.size()), obs);
             })
        .def("__repr__", &HamiltonianT::getObsName)
        .def("get_wires", &HamiltonianT::getWires, "Get wires of observables")
        .def("get_coeffs", &HamiltonianT::getCoeffs, "Get coefficients")
        .def("get_ops", &HamiltonianT::getObs, "Get operations list")
        .def(
            "__eq__",
            [](const HamiltonianT &self, const HamiltonianT &other) -> bool {
                return self == other;
            },
            "Compare two observables");
}

/**
 * @brief Register probs method for specific wires with proper data ownership
 *
 * @tparam MeasurementsT Representation of Measurements type
 * @param M Measurements object
 * @param wires Vector of wire indices
 * @return nb::ndarray<PrecisionT, nb::numpy, nb::c_contig> NumPy array with
 * probabilities
 */
template <class MeasurementsT>
nb::ndarray<typename MeasurementsT::PrecisionT, nb::numpy, nb::c_contig>
probsForWires(MeasurementsT &M, const std::vector<std::size_t> &wires) {
    using PrecisionT = typename MeasurementsT::PrecisionT;
    return createNumpyArrayFromVector<PrecisionT>(M.probs(wires));
}

/**
 * @brief Register probs method for all wires with proper data ownership
 *
 * @tparam MeasurementsT Representation of Measurements type
 * @param M Measurements object
 * @return nb::ndarray<PrecisionT, nb::numpy, nb::c_contig> NumPy array with
 * probabilities
 */
template <class MeasurementsT>
nb::ndarray<typename MeasurementsT::PrecisionT, nb::numpy, nb::c_contig>
probsForAllWires(MeasurementsT &M) {
    using PrecisionT = typename MeasurementsT::PrecisionT;
    return createNumpyArrayFromVector<PrecisionT>(M.probs());
}

/**
 * @brief Generate samples with proper data ownership
 *
 * @tparam MeasurementsT Representation of Measurements type
 * @param M Measurements object
 * @param num_wires Number of wires
 * @param num_shots Number of shots
 * @return nb::ndarray<std::size_t, nb::numpy, nb::c_contig> NumPy 2D array with
 * samples
 */
template <class MeasurementsT>
nb::ndarray<std::size_t, nb::numpy, nb::c_contig>
generateSamples(MeasurementsT &M, std::size_t num_wires,
                std::size_t num_shots) {
    return createNumpyArrayFromVector<std::size_t>(
        M.generate_samples(num_shots), num_shots, num_wires);
}

/**
 * @brief Register backend-agnostic measurement class functionalities.
 *
 * @tparam MeasurementsT Representation of Measurements type
 * @tparam ObservableT Representation of Observable type
 * @tparam Pyclass Nanobind's class object type
 * @param pyclass Nanobind's class object to bind measurements
 */
template <class MeasurementsT, class ObservableT, class PyClass>
void registerBackendAgnosticMeasurements(PyClass &pyclass) {
    // These functions are common to all *statevector* simulators
#ifndef _ENABLE_PLTENSOR
    // Set random seed
    pyclass.def("set_random_seed",
                [](MeasurementsT &M, std::size_t seed) { M.setSeed(seed); });

    // Add probs method for all wires
    pyclass.def("probs", &probsForAllWires<MeasurementsT>,
                "Calculate probabilities for all wires.");

    // Add generate_samples method
    pyclass.def("generate_samples", &generateSamples<MeasurementsT>,
                "Generate samples for all wires.");
#endif

    // Add probs method for specific wires
    pyclass.def("probs", &probsForWires<MeasurementsT>,
                "Calculate probabilities for specific wires.");

    // Add expval method for observable
    pyclass.def(
        "expval",
        [](MeasurementsT &M, const std::shared_ptr<ObservableT> &ob) {
            return M.expval(*ob);
        },
        "Expected value of an observable object.");

    // Add var method for observable
    pyclass.def(
        "var",
        [](MeasurementsT &M, const std::shared_ptr<ObservableT> &ob) {
            return M.var(*ob);
        },
        "Calculate variance for an observable.");

    // TODO: generate_samples method for specific wires for ltensor
}

#ifndef _ENABLE_PLTENSOR
/**
 * @brief Register AdjointJacobian class.
 *
 * @tparam StateVectorT State vector class.
 * @param m Nanobind module.
 */
template <class StateVectorT> void registerAdjointJacobian(nb::module_ &m) {
    using PrecisionT = typename StateVectorT::PrecisionT;

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    std::string class_name = "AdjointJacobianC" + bitsize;
    auto adjoint_jacobian_class =
        nb::class_<AdjointJacobian<StateVectorT>>(m, class_name.c_str());
    adjoint_jacobian_class.def(nb::init<>());

    // Add the __call__ method with proper binding to match pybind11 behavior
    adjoint_jacobian_class.def(
        "__call__",
        [](AdjointJacobian<StateVectorT> &adj, const StateVectorT &sv,
           const std::vector<std::shared_ptr<Observable<StateVectorT>>>
               &observables,
           const OpsData<StateVectorT> &operations,
           const std::vector<size_t> &trainableParams) {
            using PrecisionT = typename StateVectorT::PrecisionT;
            std::vector<PrecisionT> jac(
                observables.size() * trainableParams.size(), PrecisionT{0.0});
            const JacobianData<StateVectorT> jd{operations.getTotalNumParams(),
                                                sv.getLength(),
                                                sv.getData(),
                                                observables,
                                                operations,
                                                trainableParams};
            adj.adjointJacobian(std::span{jac}, jd, sv);
            return createNumpyArrayFromVector<PrecisionT>(std::move(jac));
        },
        "Calculate the Jacobian using the adjoint method.");

#ifdef _ENABLE_PLGPU
    // lightning.gpu supports an additional batched adjoint jacobian
    adjoint_jacobian_class.def(
        "batched",
        [](AdjointJacobian<StateVectorT> &adjoint_jacobian,
           const StateVectorT &sv,
           const std::vector<std::shared_ptr<Observable<StateVectorT>>>
               &observables,
           const OpsData<StateVectorT> &operations,
           const std::vector<std::size_t> &trainableParams) {
            using PrecisionT = typename StateVectorT::PrecisionT;
            std::vector<PrecisionT> jac(
                observables.size() * trainableParams.size(), PrecisionT{0.0});
            const JacobianData<StateVectorT> jd{operations.getTotalNumParams(),
                                                sv.getLength(),
                                                sv.getData(),
                                                observables,
                                                operations,
                                                trainableParams};
            adjoint_jacobian.batchAdjointJacobian(std::span{jac}, jd);
            return createNumpyArrayFromVector<PrecisionT>(std::move(jac));
        },
        "Batch Adjoint Jacobian method.");
#endif
}

/**
 * @brief Create operations list from data
 *
 * @tparam StateVectorT State vector type
 * @param ops_name Operation names
 * @param ops_params Operation parameters
 * @param ops_wires Operation wires
 * @param ops_inverses Operation inverse flags
 * @param ops_matrices Operation matrices
 * @param ops_controlled_wires Operation controlled wires
 * @param ops_controlled_values Operation controlled values
 * @return OpsData<StateVectorT> Operations data
 */
template <class StateVectorT>
OpsData<StateVectorT> createOpsList(
    const std::vector<std::string> &ops_name,
    const std::vector<std::vector<typename StateVectorT::PrecisionT>>
        &ops_params,
    const std::vector<std::vector<std::size_t>> &ops_wires,
    const std::vector<bool> &ops_inverses,
    const std::vector<nb::ndarray<
        const std::complex<typename StateVectorT::PrecisionT>, nb::c_contig>>
        &ops_matrices,
    const std::vector<std::vector<std::size_t>> &ops_controlled_wires,
    const std::vector<std::vector<bool>> &ops_controlled_values) {
    using ComplexT = typename StateVectorT::ComplexT;
    using PrecisionT = typename StateVectorT::PrecisionT;

    // Convert ops_matrices to std::vector<std::vector<ComplexT>>
    std::vector<std::vector<ComplexT>> conv_matrices =
        Pennylane::NanoBindings::Utils::convertMatrices<ComplexT, PrecisionT>(
            ops_matrices);

    return OpsData<StateVectorT>{ops_name,
                                 ops_params,
                                 ops_wires,
                                 ops_inverses,
                                 conv_matrices,
                                 ops_controlled_wires,
                                 ops_controlled_values};
}

/**
 * @brief Register agnostic algorithms for statevector simulators.
 *
 * @tparam StateVectorT
 * @param m Nanobind module
 */
template <class StateVectorT>
void registerBackendAgnosticAlgorithms(nb::module_ &m) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ComplexT =
        typename StateVectorT::ComplexT; // Statevector's complex type
    using ParamT = PrecisionT;           // Parameter's data precision

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    std::string class_name;

    //***********************************************************************//
    //                              Operations
    //***********************************************************************//

    class_name = "OpsStructC" + bitsize;
    auto ops_class = nb::class_<OpsData<StateVectorT>>(m, class_name.c_str());

    ops_class.def(nb::init<const std::vector<std::string> &,
                           const std::vector<std::vector<ParamT>> &,
                           const std::vector<std::vector<std::size_t>> &,
                           const std::vector<bool> &,
                           const std::vector<std::vector<ComplexT>> &>());

    ops_class.def(nb::init<const std::vector<std::string> &,
                           const std::vector<std::vector<ParamT>> &,
                           const std::vector<std::vector<std::size_t>> &,
                           const std::vector<bool> &,
                           const std::vector<std::vector<ComplexT>> &,
                           const std::vector<std::vector<std::size_t>> &,
                           const std::vector<std::vector<bool>> &>());

    ops_class.def("__repr__", [](const OpsData<StateVectorT> &ops) {
        return Pennylane::NanoBindings::Utils::opsDataToString(ops, true);
    });

    /**
     * Create operation list.
     */
    std::string function_name = "create_ops_listC" + bitsize;
    m.def(function_name.c_str(), &createOpsList<StateVectorT>,
          "Create a list of operations from data.");

    //***********************************************************************//
    //                            Adjoint Jacobian
    //***********************************************************************//
    // Register the AdjointJacobian class using the dedicated function
    registerAdjointJacobian<StateVectorT>(m);
}

/**
 * @brief Update state vector data from an array
 *
 * This function accepts any array-like object that follows the buffer protocol,
 * including NumPy arrays and JAX arrays (for example).
 *
 * Example with JAX:
 * ```python
 * import jax.numpy as jnp
 * import pennylane_lightning.lightning_qubit_nb as plq
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
 * @brief Register backend agnostic state vector methods.
 *
 * @tparam StateVectorT
 * @tparam PyClass
 * @param pyclass Nanobind's state vector class to bind methods.
 */
template <class StateVectorT, class PyClass>
void registerBackendAgnosticStateVectorMethods(PyClass &pyclass) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;

    // Initialize with number of qubits
    pyclass.def(nb::init<size_t>());

    pyclass.def("updateData", &updateStateVectorData<StateVectorT>,
                "Update the state vector data from an array.",
                nb::arg("state"));
    pyclass.def("__len__", &StateVectorT::getLength,
                "Get the size of the statevector.");
    pyclass.def("size", &StateVectorT::getLength);

    // Reset state vector - common across all backends
    pyclass.def("resetStateVector", &StateVectorT::resetStateVector,
                "Reset the state vector to |0...0>.");

    // Set basis state - with conditional for async parameter (LGPU)
    pyclass.def(
        "setBasisState",
        [](StateVectorT &sv, const std::vector<std::size_t> &state,
           const std::vector<std::size_t> &wires, const bool async) {
            if constexpr (requires { sv.setBasisState(state, wires, async); }) {
                sv.setBasisState(state, wires, async);
            } else {
                sv.setBasisState(state, wires);
            }
        },
        "Set the state vector to a computational basis state.",
        nb::arg("state") = nb::none(), nb::arg("wires") = nb::none(),
        nb::arg("async") = false);

    // Set state vector - with conditional for async and size parameters (LGPU)
    pyclass.def(
        "setStateVector",
        [](StateVectorT &sv,
           const nb::ndarray<const std::complex<PrecisionT>, nb::c_contig>
               &state,
           const std::vector<std::size_t> &wires, const bool async) {
            const auto *data_ptr =
                PL_reinterpret_cast<const ComplexT>(state.data());
            std::size_t size = state.shape(0);

            if constexpr (requires {
                              sv.setStateVector(data_ptr, size, wires, async);
                          }) {
                sv.setStateVector(data_ptr, size, wires, async);
            } else {
                sv.setStateVector(data_ptr, wires);
            }
        },
        "Set the state vector to the data contained in 'state'.",
        nb::arg("state"), nb::arg("wires"), nb::arg("async") = false);
}
#endif

/**
 * @brief Templated class to build lightning class bindings.
 *
 * @tparam StateReprT State representation type (e.g., a StateVector, TensorNet)
 * @param m Nanobind module.
 */
template <class StateReprT> void lightningClassBindings(nb::module_ &m) {
    using PrecisionT = typename StateReprT::PrecisionT;

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

#ifdef _ENABLE_PLTENSOR
    std::string class_name = "TensorNetC" + bitsize;
    auto pyclass = nb::class_<StateReprT>(m, class_name.c_str());
#else
    // StateVector class
    std::string class_name = "StateVectorC" + bitsize;
    auto pyclass = nb::class_<StateReprT>(m, class_name.c_str());
    registerBackendAgnosticStateVectorMethods<StateReprT>(pyclass);
    registerBackendSpecificStateVectorMethods<StateReprT>(pyclass);
#endif

    // Register gates
    registerGates<StateReprT>(pyclass);
    registerControlledGates<StateReprT>(pyclass);

    // Register backend specific bindings
    registerBackendClassSpecificBindings<StateReprT>(pyclass);

    //***********************************************************************//
    //                              Observables
    //***********************************************************************//

    /* Observables submodule */
    nb::module_ obs_submodule =
        m.def_submodule("observables", "Submodule for observables classes.");

#ifndef _ENABLE_PLTENSOR
    registerBackendAgnosticObservables<StateReprT>(obs_submodule);
#endif
    registerBackendSpecificObservables<StateReprT>(obs_submodule);

    //***********************************************************************//
    //                              Measurements
    //***********************************************************************//

#ifdef _ENABLE_PLTENSOR
    using MeasurementsT = MeasurementsTNCuda<StateReprT>;
    using ObservableT = ObservableTNCuda<StateReprT>;
#else
    using MeasurementsT = Measurements<StateReprT>;
    using ObservableT = Observable<StateReprT>;
#endif

    /* Measurements class */
    class_name = "MeasurementsC" + bitsize;
    auto pyclass_measurements =
        nb::class_<MeasurementsT>(m, class_name.c_str());

#ifdef _ENABLE_PLGPU
    // TODO: Find if getting `const` to work with GPU state vector is an easy
    // lift
    pyclass_measurements.def(nb::init<StateReprT &>());
#else
    pyclass_measurements.def(nb::init<const StateReprT &>());
#endif
    registerBackendAgnosticMeasurements<MeasurementsT, ObservableT>(
        pyclass_measurements);
    registerBackendSpecificMeasurements<StateReprT>(pyclass_measurements);

    //***********************************************************************//
    //                              Algorithms
    //***********************************************************************//

    /* Algorithms submodule */
    nb::module_ alg_submodule = m.def_submodule(
        "algorithms", "Submodule for the algorithms functionality.");
#ifndef _ENABLE_PLTENSOR
    registerBackendAgnosticAlgorithms<StateReprT>(alg_submodule);
#endif
    registerBackendSpecificAlgorithms<StateReprT>(alg_submodule);
}

/**
 * @brief Register lightning class bindings for all backends.
 *
 * @tparam TypeList List of backend types
 * @param m Nanobind module
 */
template <typename TypeList>
void registerLightningClassBindings(nb::module_ &m) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateReprT = typename TypeList::Type;
        lightningClassBindings<StateReprT>(m);
        registerLightningClassBindings<typename TypeList::Next>(m);
    }
}

} // namespace Pennylane::NanoBindings
