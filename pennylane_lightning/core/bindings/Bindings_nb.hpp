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
#include <variant>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "CPUMemoryModel.hpp" // CPUMemoryModel, getMemoryModel, bestCPUMemoryModel, getAlignment
#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "GateOperation.hpp"
#include "JacobianData.hpp"
#include "Macros.hpp" // CPUArch
#include "Memory.hpp" // alignedAlloc
#include "Observables.hpp"
#include "Util.hpp" // for_each_enum

// Include backend-specific headers and define macros based on compile flags
#ifdef _ENABLE_PLQUBIT
#include "AdjointJacobianLQubit.hpp"
#include "LQubitBindings_nb.hpp"
#include "MeasurementsLQubit.hpp"
#include "ObservablesLQubit.hpp"

#define LIGHTNING_MODULE_NAME lightning_qubit_nb

namespace {
using namespace Pennylane::LightningQubit;
using namespace Pennylane::LightningQubit::Algorithms;
using namespace Pennylane::LightningQubit::Observables;
using namespace Pennylane::LightningQubit::Measures;
using namespace Pennylane::LightningQubit::NanoBindings;
} // namespace

#elif _ENABLE_PLKOKKOS == 1
#include "AdjointJacobianKokkos.hpp"
#include "LKokkosBindings_nb.hpp"
#include "MeasurementsKokkos.hpp"
#include "ObservablesKokkos.hpp"

#define LIGHTNING_MODULE_NAME lightning_kokkos_nb

namespace {
using namespace Pennylane::LightningKokkos;
using namespace Pennylane::LightningKokkos::Algorithms;
using namespace Pennylane::LightningKokkos::Observables;
using namespace Pennylane::LightningKokkos::Measures;
using namespace Pennylane::LightningKokkos::NanoBindings;
} // namespace

#elif _ENABLE_PLGPU == 1
#include "AdjointJacobianGPU.hpp"
#include "BindingsCudaUtils_nb.hpp"
#include "LGPUBindings_nb.hpp"
#include "MeasurementsGPU.hpp"
#include "ObservablesGPU.hpp"

#define LIGHTNING_MODULE_NAME lightning_gpu_nb

namespace {
using namespace Pennylane::LightningGPU;
using namespace Pennylane::LightningGPU::Algorithms;
using namespace Pennylane::LightningGPU::Observables;
using namespace Pennylane::LightningGPU::Measures;
using namespace Pennylane::LightningGPU::NanoBindings;
} // namespace

#elif _ENABLE_PLTENSOR == 1
#include "AdjointJacobianTNCuda.hpp"
#include "MeasurementsTNCuda.hpp"
#include "ObservablesTNCuda.hpp"

#define LIGHTNING_TENSOR_MODULE_NAME lightning_tensor_nb

namespace {
using namespace Pennylane::LightningTensor::TNCuda;
using namespace Pennylane::LightningTensor::TNCuda::Observables;
using namespace Pennylane::LightningTensor::TNCuda::Measures;
using namespace Pennylane::LightningTensor::TNCuda::NanoBindings;
} // namespace

#else
static_assert(false, "Backend not found.");
#endif

namespace nb = nanobind;
namespace Pennylane::NanoBindings {

/**
 * @brief Register applyMatrix.
 */
template <class StateVectorT>
void registerMatrix(StateVectorT &st,
                    const nb::ndarray<typename StateVectorT::ComplexT> &matrix,
                    const std::vector<std::size_t> &wires,
                    bool inverse = false) {
    using ComplexT = typename StateVectorT::ComplexT;

    // Get data pointer from ndarray
    const ComplexT *data_ptr =
        reinterpret_cast<const ComplexT *>(matrix.data());
    st.applyMatrix(data_ptr, wires, inverse);
}

/**
 * @brief Register StateVector class.
 *
 * @tparam StateVectorT Statevector type to register
 * @tparam PyClass Nanobind's class object type
 *
 * @param pyclass Nanobind's class object to bind statevector
 */
template <class StateVectorT, class PyClass>
void registerGatesForStateVector(PyClass &pyclass) {
    using PrecisionT =
        typename StateVectorT::PrecisionT; // Statevector's precision
    using ParamT = PrecisionT;             // Parameter's data precision

    using Pennylane::Gates::GateOperation;
    using Pennylane::Util::for_each_enum;
    namespace Constant = Pennylane::Gates::Constant;

    pyclass.def("applyMatrix", &registerMatrix<StateVectorT>,
                "Apply a given matrix to wires.");

    for_each_enum<GateOperation>([&pyclass](GateOperation gate_op) {
        using Pennylane::Util::lookup;
        const auto gate_name =
            std::string(lookup(Constant::gate_names, gate_op));
        const std::string doc = "Apply the " + gate_name + " gate.";
        auto func =
            [gate_name](StateVectorT &sv, const std::vector<std::size_t> &wires,
                        bool inverse, const std::vector<ParamT> &params) {
                sv.applyOperation(gate_name, wires, inverse, params);
            };
        pyclass.def(gate_name.c_str(), func, doc.c_str());
    });
}

/**
 * @brief Create an aligned array for a given type, memory model and array size.
 *
 * @tparam T Datatype of array to create
 * @param memory_model Memory model to use
 * @param size Size of the array to create
 * @return Nanobind ndarray
 */
template <typename T>
auto alignedArray(Util::CPUMemoryModel memory_model, std::size_t size,
                  bool zeroInit) -> nb::ndarray<T> {
    using Pennylane::Util::alignedAlloc;
    using Pennylane::Util::getAlignment;

    // Allocate aligned memory
    void *ptr =
        alignedAlloc(getAlignment<T>(memory_model), sizeof(T) * size, zeroInit);

    // Create capsule with custom deleter
    auto capsule =
        nb::capsule(ptr, [](void *p) noexcept { Util::alignedFree(p); });

    std::vector<size_t> shape{size};

    // Return ndarray with custom allocated memory
    return nb::ndarray<T>(ptr, 1, shape.data(), capsule);
}

/**
 * @brief Allocate aligned array with specified dtype.
 *
 * @param size Size of the array to create
 * @param dtype Data type as string ("complex64", "complex128", "float32",
 * "float64")
 * @param zeroInit Whether to initialize the array with zeros
 * @return Nanobind ndarray
 */
auto allocateAlignedArray(std::size_t size, const std::string &dtype,
                          bool zeroInit) -> nb::object {
    auto memory_model = Pennylane::Util::bestCPUMemoryModel();

    if (dtype == "complex64") {
        return nb::cast(
            alignedArray<std::complex<float>>(memory_model, size, zeroInit));
    } else if (dtype == "complex128") {
        return nb::cast(
            alignedArray<std::complex<double>>(memory_model, size, zeroInit));
    } else if (dtype == "float32") {
        return nb::cast(alignedArray<float>(memory_model, size, zeroInit));
    } else if (dtype == "float64") {
        return nb::cast(alignedArray<double>(memory_model, size, zeroInit));
    }

    throw std::runtime_error("Unsupported dtype: " + dtype);
}

/**
 * @brief Register array alignment functionality.
 *
 * @param m Nanobind module
 */
void registerArrayAlignmentBindings(nb::module_ &m) {
    using Pennylane::Util::bestCPUMemoryModel;
    using Pennylane::Util::CPUMemoryModel;

    // Register CPUMemoryModel enum
    nb::enum_<CPUMemoryModel>(m, "CPUMemoryModel")
        .value("Unaligned", CPUMemoryModel::Unaligned)
        .value("Aligned256", CPUMemoryModel::Aligned256)
        .value("Aligned512", CPUMemoryModel::Aligned512);

    // Register utility functions
    m.def("best_alignment", &bestCPUMemoryModel,
          "Best memory alignment for the simulator.");

    // Add allocate_aligned_array function
    m.def("allocate_aligned_array", &allocateAlignedArray,
          "Allocate aligned array with specified dtype", nb::arg("size"),
          nb::arg("dtype"), nb::arg("zero_init") = false);
}

/**
 * @brief Return basic information of runtime environment.
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
 * @brief Get compile information as a dictionary.
 *
 * @return Dictionary with compile information.
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

/**
 * @brief Register bindings for general info.
 *
 * @param m Nanobind module.
 */
void registerInfo(nb::module_ &m) {
    // Add compile info
    m.def("compile_info", &getCompileInfo, "Compiled binary information.");

    // Add runtime info
    m.def("runtime_info", &getRuntimeInfo, "Runtime information.");
}

/**
 * @brief Register backend-agnostic observables.
 *
 * @tparam LightningBackendT
 * @param m Nanobind module
 */
template <class LightningBackendT>
void registerBackendAgnosticObservables(nb::module_ &m) {
    using PrecisionT = typename LightningBackendT::PrecisionT;
    using ComplexT = typename LightningBackendT::ComplexT;
    using ParamT = PrecisionT;

    using nd_arr_c = nb::ndarray<std::complex<ParamT>>;

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

#ifdef _ENABLE_PLTENSOR
    using ObservableT = ObservableTNCuda<LightningBackendT>;
    using NamedObsT = NamedObsTNCuda<LightningBackendT>;
    using HermitianObsT = HermitianObsTNCuda<LightningBackendT>;
    using TensorProdObsT = TensorProdObsTNCuda<LightningBackendT>;
    using HamiltonianT = HamiltonianTNCuda<LightningBackendT>;
#else
    using ObservableT = Observable<LightningBackendT>;
    using NamedObsT = NamedObs<LightningBackendT>;
    using HermitianObsT = HermitianObs<LightningBackendT>;
    using TensorProdObsT = TensorProdObs<LightningBackendT>;
    using HamiltonianT = Hamiltonian<LightningBackendT>;
#endif

    std::string class_name;

    // Register Observable base class
    class_name = "ObservableC" + bitsize;
    auto observable = nb::class_<ObservableT>(m, class_name.c_str());
    observable.def("get_wires", &ObservableT::getWires,
                   "Get wires the observable acts on.");

    // Register NamedObs class
    class_name = "NamedObsC" + bitsize;
    auto named_obs = nb::class_<NamedObsT, ObservableT>(m, class_name.c_str());
    named_obs
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
    auto hermitian_obs =
        nb::class_<HermitianObsT, ObservableT>(m, class_name.c_str());
    hermitian_obs
        .def("__init__",
             [](HermitianObsT *self, const nd_arr_c &matrix,
                const std::vector<std::size_t> &wires) {
                 const auto ptr = static_cast<const ComplexT *>(matrix.data());
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
    auto tensor_prod_obs =
        nb::class_<TensorProdObsT, ObservableT>(m, class_name.c_str());
    tensor_prod_obs
        .def(nb::init<const std::vector<std::shared_ptr<ObservableT>> &>())
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
    using ObsPtr = std::shared_ptr<ObservableT>;
    auto hamiltonian =
        nb::class_<HamiltonianT, ObservableT>(m, class_name.c_str());
    hamiltonian
        .def(nb::init<const std::vector<ParamT> &,
                      const std::vector<ObsPtr> &>())
        .def("__init__",
             [](HamiltonianT *self, const nb::ndarray<ParamT> &coeffs,
                const std::vector<ObsPtr> &obs) {
                 const auto ptr = static_cast<const ParamT *>(coeffs.data());
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
 * @brief Create an array from a vector of data with proper ownership transfer
 *
 * @tparam T Data type of the vector elements
 * @param data Vector containing the data to transfer
 * @return nb::ndarray<T, nb::numpy> Array with copied data in numpy format
 */
template <typename T>
nb::ndarray<T, nb::numpy> createArrayFromVector(const std::vector<T> &data) {
    const std::size_t size = data.size();

    // Create a new array with the right size
    std::vector<size_t> shape{size};

    // Allocate new memory and copy the data
    T *new_data = new T[size];
    std::memcpy(new_data, data.data(), size * sizeof(T));

    // Create a capsule to manage memory
    auto capsule = nb::capsule(
        new_data, [](void *p) noexcept { delete[] static_cast<T *>(p); });

    // Create and return the ndarray with numpy format
    return nb::ndarray<T, nb::numpy>(new_data, 1, shape.data(), capsule);
}

/**
 * @brief Create a 2D array from a vector of data with proper ownership transfer
 *
 * @tparam T Data type of the vector elements
 * @param data Vector containing the data to transfer
 * @param rows Number of rows in the resulting 2D array
 * @param cols Number of columns in the resulting 2D array
 * @return nb::ndarray<T, nb::numpy> 2D array with copied data in numpy format
 */
template <typename T>
nb::ndarray<T, nb::numpy> create2DArrayFromVector(const std::vector<T> &data,
                                                  std::size_t rows,
                                                  std::size_t cols) {
    // Create a new array with the right size
    std::vector<size_t> shape{rows, cols};

    // Allocate new memory and copy the data
    T *new_data = new T[rows * cols];
    std::memcpy(new_data, data.data(), data.size() * sizeof(T));

    // Create a capsule to manage memory
    auto capsule = nb::capsule(
        new_data, [](void *p) noexcept { delete[] static_cast<T *>(p); });

    // Create and return the ndarray with numpy format
    return nb::ndarray<T, nb::numpy>(new_data, 2, shape.data(), capsule);
}

/**
 * @brief Register probs method for specific wires with proper data ownership
 *
 * @tparam StateVectorT State vector type
 * @param M Measurements object
 * @param wires Vector of wire indices
 * @return nb::ndarray<typename StateVectorT::PrecisionT, nb::numpy> Array with
 * probabilities in numpy format
 */
template <class StateVectorT>
nb::ndarray<typename StateVectorT::PrecisionT, nb::numpy>
probsForWires(Measurements<StateVectorT> &M,
              const std::vector<std::size_t> &wires) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    auto probs_vec = M.probs(wires);
    return createArrayFromVector<PrecisionT>(probs_vec);
}

/**
 * @brief Register probs method for all wires with proper data ownership
 *
 * @tparam StateVectorT State vector type
 * @param M Measurements object
 * @return nb::ndarray<typename StateVectorT::PrecisionT, nb::numpy> Array with
 * probabilities in numpy format
 */
template <class StateVectorT>
nb::ndarray<typename StateVectorT::PrecisionT, nb::numpy>
probsForAllWires(Measurements<StateVectorT> &M) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    auto probs_vec = M.probs();
    return createArrayFromVector<PrecisionT>(probs_vec);
}

/**
 * @brief Generate samples with proper data ownership
 *
 * @tparam StateVectorT State vector type
 * @param M Measurements object
 * @param num_wires Number of wires
 * @param num_shots Number of shots
 * @return nb::ndarray<std::size_t, nb::numpy> 2D array with samples in numpy
 * format
 */
template <class StateVectorT>
nb::ndarray<std::size_t, nb::numpy>
generateSamples(Measurements<StateVectorT> &M, std::size_t num_wires,
                std::size_t num_shots) {
    auto result = M.generate_samples(num_shots);
    return create2DArrayFromVector<std::size_t>(result, num_shots, num_wires);
}

/**
 * @brief Register backend-agnostic measurement class functionalities.
 *
 * @tparam StateVectorT
 * @param pyclass Nanobind's class object to bind measurements
 */
template <class StateVectorT, class PyClass>
void registerBackendAgnosticMeasurements(PyClass &pyclass) {
    // Add probs method for specific wires
    pyclass.def("probs", &probsForWires<StateVectorT>,
                "Calculate probabilities for specific wires.");

    // Add probs method for all wires
    pyclass.def("probs", &probsForAllWires<StateVectorT>,
                "Calculate probabilities for all wires.");

    // Add expval method for observable
    pyclass.def(
        "expval",
        [](Measurements<StateVectorT> &M, const Observable<StateVectorT> &obs) {
            return M.expval(obs);
        },
        "Calculate expectation value for an observable.");

    // Add var method for observable
    pyclass.def(
        "var",
        [](Measurements<StateVectorT> &M, const Observable<StateVectorT> &obs) {
            return M.var(obs);
        },
        "Calculate variance for an observable.");

    // Add generate_samples method
    pyclass.def("generate_samples", &generateSamples<StateVectorT>,
                "Generate samples for all wires.");
}

/**
 * @brief Register agnostic algorithms.
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

    using arr_c = nb::ndarray<std::complex<ParamT>, nb::c_contig>;

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    std::string class_name;

    //***********************************************************************//
    //                              Operations
    //***********************************************************************//

    class_name = "OpsStructC" + bitsize;
    nb::class_<OpsData<StateVectorT>>(m, class_name.c_str())
        .def(nb::init<const std::vector<std::string> &,
                      const std::vector<std::vector<ParamT>> &,
                      const std::vector<std::vector<std::size_t>> &,
                      const std::vector<bool> &,
                      const std::vector<std::vector<ComplexT>> &>())
        .def(nb::init<const std::vector<std::string> &,
                      const std::vector<std::vector<ParamT>> &,
                      const std::vector<std::vector<std::size_t>> &,
                      const std::vector<bool> &,
                      const std::vector<std::vector<ComplexT>> &,
                      const std::vector<std::vector<std::size_t>> &,
                      const std::vector<std::vector<bool>> &>())
        .def("__repr__", [](const OpsData<StateVectorT> &ops) {
            using namespace Pennylane::Util;
            std::ostringstream ops_stream;
            for (std::size_t op = 0; op < ops.getSize(); op++) {
                ops_stream << "{'name': " << ops.getOpsName()[op];
                ops_stream << ", 'params': " << ops.getOpsParams()[op];
                ops_stream << ", 'inv': " << ops.getOpsInverses()[op];
                ops_stream << ", 'controlled_wires': "
                           << ops.getOpsControlledWires()[op];
                ops_stream << ", 'controlled_values': "
                           << ops.getOpsControlledValues()[op];
                ops_stream << ", 'wires': " << ops.getOpsWires()[op];
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
    std::string function_name = "create_ops_listC" + bitsize;
    m.def(
        function_name.c_str(),
        [](const std::vector<std::string> &ops_name,
           const std::vector<std::vector<PrecisionT>> &ops_params,
           const std::vector<std::vector<std::size_t>> &ops_wires,
           const std::vector<bool> &ops_inverses,
           const std::vector<arr_c> &ops_matrices,
           const std::vector<std::vector<std::size_t>> &ops_controlled_wires,
           const std::vector<std::vector<bool>> &ops_controlled_values) {
            std::vector<std::vector<ComplexT>> conv_matrices(
                ops_matrices.size());
            for (std::size_t op = 0; op < ops_name.size(); op++) {
                if (ops_matrices[op].size() > 0) {
                    const auto *m_ptr = ops_matrices[op].data();
                    const auto m_size = ops_matrices[op].size();
                    conv_matrices[op] =
                        std::vector<ComplexT>(m_ptr, m_ptr + m_size);
                }
            }
            return OpsData<StateVectorT>{ops_name,
                                         ops_params,
                                         ops_wires,
                                         ops_inverses,
                                         conv_matrices,
                                         ops_controlled_wires,
                                         ops_controlled_values};
        },
        "Create a list of operations from data.");
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
    using ParamT = PrecisionT;

    // Initialize with number of qubits
    pyclass.def(nb::init<size_t>());

    pyclass.def("__len__", &StateVectorT::getLength,
                "Get the size of the statevector.");
    pyclass.def("size", &StateVectorT::getLength);
}

/**
 * @brief Templated class to build lightning class bindings.
 *
 * @tparam StateVectorT State vector type
 * @param m Nanobind module.
 */
template <class StateVectorT> void lightningClassBindings(nb::module_ &m) {
    using PrecisionT = typename StateVectorT::PrecisionT;
    using ComplexT = typename StateVectorT::ComplexT;
    using ParamT = PrecisionT;

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    // StateVector class
    std::string class_name = "StateVectorC" + bitsize;
    auto pyclass = nb::class_<StateVectorT>(m, class_name.c_str());
    registerBackendAgnosticStateVectorMethods<StateVectorT>(pyclass);
    registerBackendSpecificStateVectorMethods<StateVectorT>(pyclass);

    // Register gates for StateVector
    registerGatesForStateVector<StateVectorT>(pyclass);

    // Register backend specific bindings
    registerBackendClassSpecificBindings<StateVectorT>(pyclass);

    //***********************************************************************//
    //                              Observables
    //***********************************************************************//

    /* Observables submodule */
    nb::module_ obs_submodule =
        m.def_submodule("observables", "Submodule for observables classes.");
    registerBackendAgnosticObservables<StateVectorT>(obs_submodule);
    registerBackendSpecificObservables<StateVectorT>(obs_submodule);

    //***********************************************************************//
    //                              Measurements
    //***********************************************************************//

    /* Measurements class */
    class_name = "MeasurementsC" + bitsize;
    auto pyclass_measurements =
        nb::class_<Measurements<StateVectorT>>(m, class_name.c_str());

#ifdef _ENABLE_PLGPU
    // TODO: Find if getting `const` to work with GPU state vector is an easy
    // lift
    pyclass_measurements.def(nb::init<StateVectorT &>());
#else
    pyclass_measurements.def(nb::init<const StateVectorT &>());
#endif

    registerBackendAgnosticMeasurements<StateVectorT>(pyclass_measurements);
    registerBackendSpecificMeasurements<StateVectorT>(pyclass_measurements);

    //***********************************************************************//
    //                              Algorithms
    //***********************************************************************//

    /* Algorithms submodule */
    nb::module_ alg_submodule = m.def_submodule(
        "algorithms", "Submodule for the algorithms functionality.");
    registerBackendAgnosticAlgorithms<StateVectorT>(alg_submodule);
    registerBackendSpecificAlgorithms<StateVectorT>(alg_submodule);
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
        using StateVectorT = typename TypeList::Type;
        lightningClassBindings<StateVectorT>(m);
        registerLightningClassBindings<typename TypeList::Next>(m);
    }
}

} // namespace Pennylane::NanoBindings
