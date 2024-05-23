// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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

#include <pybind11/pybind11.h>

#include "Constant.hpp"
#include "ConstantUtil.hpp" // lookup
#include "GateOperation.hpp"

namespace py = pybind11;

namespace Pennylane::Bindings {
/**
 * @brief Register matrix.
 */
template <class StateTensorT>
void registerMatrix(
    StateTensorT &state_tensor,
    const py::array_t<std::complex<typename StateTensorT::PrecisionT>,
                      py::array::c_style | py::array::forcecast> &matrix,
    const std::vector<std::size_t> &wires, bool inverse = false) {
    using ComplexT = typename StateTensorT::ComplexT;
    const auto m_buffer = matrix.request();
    std::vector<ComplexT> conv_matrix;
    if (m_buffer.size) {
        const auto m_ptr = static_cast<const ComplexT *>(m_buffer.ptr);
        conv_matrix = std::vector<ComplexT>{
                        m_ptr, m_ptr + m_buffer.size};
    }
    state_tensor.applyOperation("applyMatrix", wires, inverse, {}, conv_matrix);
}

/**
 * @brief Register StateTensor class to pybind.
 *
 * @tparam StateTensorT Statetensor type to register
 * @tparam Pyclass Pybind11's class object type
 *
 * @param pyclass Pybind11's class object to bind statetensor
 */
template <class StateTensorT, class PyClass>
void registerGatesForStateTensor(PyClass &pyclass) {
    using PrecisionT =
        typename StateTensorT::PrecisionT; // Statetensor's precision
    using ParamT = PrecisionT;             // Parameter's data precision

    using Pennylane::Gates::GateOperation;
    using Pennylane::Util::for_each_enum;
    namespace Constant = Pennylane::Gates::Constant;

    pyclass.def("applyMatrix", &registerMatrix<StateTensorT>,
                "Apply a given matrix to wires.");

    for_each_enum<GateOperation>([&pyclass](GateOperation gate_op) {
        using Pennylane::Util::lookup;
        const auto gate_name =
            std::string(lookup(Constant::gate_names, gate_op));
        const std::string doc = "Apply the " + gate_name + " gate.";
        auto func = [gate_name = gate_name](
                        StateTensorT &state_tensor, const std::vector<std::size_t> &wires,
                        bool inverse, const std::vector<ParamT> &params) {
            state_tensor.applyOperation(gate_name, wires, inverse, params);
        };
        pyclass.def(gate_name.c_str(), func, doc.c_str());
    });
}
}

namespace Pennylane {
/**
 * @brief Get memory alignment of a given numpy array.
 *
 * @param numpyArray Pybind11's numpy array type.
 * @return CPUMemoryModel Memory model describing alignment
 */
auto getNumpyArrayAlignment(const py::array &numpyArray) -> CPUMemoryModel {
    return getMemoryModel(numpyArray.request().ptr);
}

/**
 * @brief Create an aligned numpy array for a given type, memory model and array
 * size.
 *
 * @tparam T Datatype of numpy array to create
 * @param memory_model Memory model to use
 * @param size Size of the array to create
 * @return Numpy array
 */
template <typename T>
auto alignedNumpyArray(CPUMemoryModel memory_model, std::size_t size,
                       bool zeroInit = false) -> py::array {
    using Pennylane::Util::alignedAlloc;
    if (getAlignment<T>(memory_model) > alignof(std::max_align_t)) {
        void *ptr = alignedAlloc(getAlignment<T>(memory_model),
                                 sizeof(T) * size, zeroInit);
        auto capsule = py::capsule(ptr, &Util::alignedFree);
        return py::array{py::dtype::of<T>(), {size}, {sizeof(T)}, ptr, capsule};
    }
    void *ptr = static_cast<void *>(new T[size]);
    auto capsule =
        py::capsule(ptr, [](void *p) { delete static_cast<T *>(p); });
    return py::array{py::dtype::of<T>(), {size}, {sizeof(T)}, ptr, capsule};
}
/**
 * @brief Create a numpy array whose underlying data is allocated by
 * lightning.
 *
 * See https://github.com/pybind/pybind11/issues/1042#issuecomment-325941022
 * for capsule usage.
 *
 * @param size Size of the array to create
 * @param dt Pybind11's datatype object
 */
auto allocateAlignedArray(size_t size, const py::dtype &dt,
                          bool zeroInit = false) -> py::array {
    // TODO: Move memset operations to here to reduce zeroInit pass-throughs.
    auto memory_model = bestCPUMemoryModel();

    if (dt.is(py::dtype::of<float>())) {
        return alignedNumpyArray<float>(memory_model, size, zeroInit);
    }
    if (dt.is(py::dtype::of<double>())) {
        return alignedNumpyArray<double>(memory_model, size, zeroInit);
    }
    if (dt.is(py::dtype::of<std::complex<float>>())) {
        return alignedNumpyArray<std::complex<float>>(memory_model, size,
                                                      zeroInit);
    }
    if (dt.is(py::dtype::of<std::complex<double>>())) {
        return alignedNumpyArray<std::complex<double>>(memory_model, size,
                                                       zeroInit);
    }
    throw py::type_error("Unsupported datatype.");
}

/**
 * @brief Register functionality for numpy array memory alignment.
 *
 * @param m Pybind module
 */
void registerArrayAlignmentBindings(py::module_ &m) {
    /* Add CPUMemoryModel enum class */
    py::enum_<CPUMemoryModel>(m, "CPUMemoryModel", py::module_local())
        .value("Unaligned", CPUMemoryModel::Unaligned)
        .value("Aligned256", CPUMemoryModel::Aligned256)
        .value("Aligned512", CPUMemoryModel::Aligned512);

    /* Add array alignment functionality */
    m.def("get_alignment", &getNumpyArrayAlignment,
          "Get alignment of an underlying data for a numpy array.");
    m.def("allocate_aligned_array", &allocateAlignedArray,
          "Get numpy array whose underlying data is aligned.");
    m.def("best_alignment", &bestCPUMemoryModel,
          "Best memory alignment. for the simulator.");
}

/**
 * @brief Return basic information of the compiled binary.
 */
auto getCompileInfo() -> py::dict {
    using namespace Pennylane::Util;
    using namespace py::literals;

    const std::string_view cpu_arch_str = [] {
        switch (cpu_arch) {
        case CPUArch::X86_64:
            return "x86_64";
        case CPUArch::PPC64:
            return "PPC64";
        case CPUArch::ARM:
            return "ARM";
        default:
            return "Unknown";
        }
    }();

    const std::string_view compiler_name_str = [] {
        switch (compiler) {
        case Compiler::GCC:
            return "GCC";
        case Compiler::Clang:
            return "Clang";
        case Compiler::MSVC:
            return "MSVC";
        case Compiler::NVCC:
            return "NVCC";
        case Compiler::NVHPC:
            return "NVHPC";
        default:
            return "Unknown";
        }
    }();

    const auto compiler_version_str = getCompilerVersion<compiler>();

    return py::dict("cpu.arch"_a = cpu_arch_str,
                    "compiler.name"_a = compiler_name_str,
                    "compiler.version"_a = compiler_version_str,
                    "AVX2"_a = use_avx2, "AVX512F"_a = use_avx512f);
}

/**
 * @brief Return basic information of runtime environment.
 */
auto getRuntimeInfo() -> py::dict {
    using Pennylane::Util::RuntimeInfo;
    using namespace py::literals;

    return py::dict("AVX"_a = RuntimeInfo::AVX(),
                    "AVX2"_a = RuntimeInfo::AVX2(),
                    "AVX512F"_a = RuntimeInfo::AVX512F());
}

/**
 * @brief Register bindings for general info.
 *
 * @param m Pybind11 module.
 */
void registerInfo(py::module_ &m) {
    /* Add compile info */
    m.def("compile_info", &getCompileInfo, "Compiled binary information.");

    /* Add runtime info */
    m.def("runtime_info", &getRuntimeInfo, "Runtime information.");
}

/**
 * @brief Register observable classes.
 *
 * @tparam StateTensorT
 * @param m Pybind module
 */
template <class StateTensorT>
void registerBackendAgnosticObservables(py::module_ &m) {
    using PrecisionT =
        typename StateTensorT::PrecisionT; // Statevector's precision.
    using ComplexT =
        typename StateTensorT::ComplexT; // Statevector's complex type.
    using ParamT = PrecisionT;           // Parameter's data precision

    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;
    using np_arr_r = py::array_t<ParamT, py::array::c_style>;

    std::string class_name;

    class_name = "ObservableC" + bitsize;
    py::class_<Observable<StateTensorT>,
               std::shared_ptr<Observable<StateTensorT>>>(m, class_name.c_str(),
                                                          py::module_local());

    class_name = "NamedObsC" + bitsize;
    py::class_<NamedObs<StateTensorT>, std::shared_ptr<NamedObs<StateTensorT>>,
               Observable<StateTensorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const std::string &name, const std::vector<std::size_t> &wires) {
                return NamedObs<StateTensorT>(name, wires);
            }))
        .def("__repr__", &NamedObs<StateTensorT>::getObsName)
        .def("get_wires", &NamedObs<StateTensorT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const NamedObs<StateTensorT> &self, py::handle other) -> bool {
                if (!py::isinstance<NamedObs<StateTensorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<NamedObs<StateTensorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HermitianObsC" + bitsize;
    py::class_<HermitianObs<StateTensorT>,
               std::shared_ptr<HermitianObs<StateTensorT>>,
               Observable<StateTensorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const np_arr_c &matrix, const std::vector<std::size_t> &wires) {
                auto buffer = matrix.request();
                const auto *ptr = static_cast<ComplexT *>(buffer.ptr);
                return HermitianObs<StateTensorT>(
                    std::vector<ComplexT>(ptr, ptr + buffer.size), wires);
            }))
        .def("__repr__", &HermitianObs<StateTensorT>::getObsName)
        .def("get_wires", &HermitianObs<StateTensorT>::getWires,
             "Get wires of observables")
        .def("get_matrix", &HermitianObs<StateTensorT>::getMatrix,
             "Get matrix representation of Hermitian operator")
        .def(
            "__eq__",
            [](const HermitianObs<StateTensorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<HermitianObs<StateTensorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<HermitianObs<StateTensorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "TensorProdObsC" + bitsize;
    py::class_<TensorProdObs<StateTensorT>,
               std::shared_ptr<TensorProdObs<StateTensorT>>,
               Observable<StateTensorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const std::vector<std::shared_ptr<Observable<StateTensorT>>>
                   &obs) { return TensorProdObs<StateTensorT>(obs); }))
        .def("__repr__", &TensorProdObs<StateTensorT>::getObsName)
        .def("get_wires", &TensorProdObs<StateTensorT>::getWires,
             "Get wires of observables")
        .def("get_ops", &TensorProdObs<StateTensorT>::getObs,
             "Get operations list")
        .def(
            "__eq__",
            [](const TensorProdObs<StateTensorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<TensorProdObs<StateTensorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<TensorProdObs<StateTensorT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HamiltonianC" + bitsize;
    using ObsPtr = std::shared_ptr<Observable<StateTensorT>>;
    py::class_<Hamiltonian<StateTensorT>,
               std::shared_ptr<Hamiltonian<StateTensorT>>,
               Observable<StateTensorT>>(m, class_name.c_str(),
                                         py::module_local())
        .def(py::init(
            [](const np_arr_r &coeffs, const std::vector<ObsPtr> &obs) {
                auto buffer = coeffs.request();
                const auto ptr = static_cast<const ParamT *>(buffer.ptr);
                return Hamiltonian<StateTensorT>{
                    std::vector(ptr, ptr + buffer.size), obs};
            }))
        .def("__repr__", &Hamiltonian<StateTensorT>::getObsName)
        .def("get_wires", &Hamiltonian<StateTensorT>::getWires,
             "Get wires of observables")
        .def("get_ops", &Hamiltonian<StateTensorT>::getObs,
             "Get operations contained by Hamiltonian")
        .def("get_coeffs", &Hamiltonian<StateTensorT>::getCoeffs,
             "Get Hamiltonian coefficients")
        .def(
            "__eq__",
            [](const Hamiltonian<StateTensorT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<Hamiltonian<StateTensorT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<Hamiltonian<StateTensorT>>();
                return self == other_cast;
            },
            "Compare two observables");
}

/**
 * @brief Register agnostic measurements class functionalities.
 *
 * @tparam StateTensorT
 * @tparam PyClass
 * @param pyclass Pybind11's measurements class to bind methods.
 */
template <class StateTensorT, class PyClass>
void registerBackendAgnosticMeasurements(PyClass &pyclass) {
    using PrecisionT =
        typename StateTensorT::PrecisionT; // StateTensor's precision.
    using ParamT = PrecisionT;             // Parameter's data precision

    pyclass
        .def(
            "expval",
            [](Measurements<StateTensorT> &M,
               const std::shared_ptr<Observable<StateTensorT>> &ob) {
                return M.expval(*ob);
            },
            "Expected value of an observable object.");
}

/**
 * @brief Templated class to build lightning class bindings.
 *
 * @tparam StateVectorT State vector type
 * @param m Pybind11 module.
 */
template <class StateTensorT> void lightningClassBindings(py::module_ &m) {
    using PrecisionT =
        typename StateTensorT::PrecisionT; // StateTensor's precision.
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    //***********************************************************************//
    //                              StateTensor
    //***********************************************************************//
    std::string class_name = "StateTensorC" + bitsize;
    auto pyclass =
        py::class_<StateTensor>(m, class_name.c_str(), py::module_local());

    registerBackendClassSpecificBindings<StateTensorT>(pyclass);

    //***********************************************************************//
    //                              Observables
    //***********************************************************************//
    /* Observables submodule */
    py::module_ obs_submodule =
        m.def_submodule("observables", "Submodule for observables classes.");
    registerBackendAgnosticObservables<StateTensorT>(obs_submodule);
    registerBackendSpecificObservables<StateTensorT>(obs_submodule);

    //***********************************************************************//
    //                             Measurements
    //***********************************************************************//
    class_name = "MeasurementsC" + bitsize;
    auto pyclass_measurements = py::class_<Measurements<StateTensorT>>(
        m, class_name.c_str(), py::module_local());

    pyclass_measurements.def(py::init<StateTensorT &>());
    registerBackendAgnosticMeasurements<StateTensorT>(pyclass_measurements);
}

template <typename TypeList>
void registerLightningClassBindings(py::module_ &m) {
    if constexpr (!std::is_same_v<TypeList, void>) {
        using StateVectorT = typename TypeList::Type;
        lightningClassBindings<StateVectorT>(m);
        registerLightningClassBindings<typename TypeList::Next>(m);
        py::register_local_exception<Pennylane::Util::LightningException>(
            m, "LightningException");
    }
}
} // namespace Pennylane
