// Copyright 2021 Xanadu Quantum Technologies Inc.

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
 * @file Bindings.cpp
 * Export C++ functions to Python using Pybind.
 */
#include "Bindings.hpp"

#include "GateUtil.hpp"
#include "Measures.hpp"
#include "StateVecAdjDiff.hpp"
#include "StateVectorManagedCPU.hpp"

#include "pybind11/pybind11.h"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Util;
using namespace Pennylane::Simulators;
using namespace Pennylane::Algorithms;
using namespace Pennylane::Gates;

using Pennylane::StateVectorRawCPU;

using std::complex;
using std::string;
using std::vector;
} // namespace
/// @endcond

namespace py = pybind11;

/**
 * @brief Templated class to build all required precisions for Python module.
 *
 * @tparam PrecisionT Precision of the state-vector data.
 * @tparam ParamT Precision of the parameter data.
 * @param m Pybind11 module.
 */
template <class PrecisionT, class ParamT>
void lightning_class_bindings(py::module_ &m) {
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;
    using sparse_index_type =
        long int; // Kokkos Kernels needs signed int as Ordinal type.
    using np_arr_sparse_ind =
        py::array_t<sparse_index_type,
                    py::array::c_style | py::array::forcecast>;

    //***********************************************************************//
    //                              StateVector
    //***********************************************************************//
    //
    std::string class_name = "StateVectorC" + bitsize;
    auto pyclass = py::class_<StateVectorRawCPU<PrecisionT>>(
        m, class_name.c_str(), py::module_local());
    pyclass.def(py::init(&createRaw<PrecisionT>));

    registerGatesForStateVector<PrecisionT, ParamT,
                                StateVectorRawCPU<PrecisionT>>(pyclass);

    pyclass.def("kernel_map", &svKernelMap<PrecisionT>,
                "Get internal kernels for operations");

    //***********************************************************************//
    //                              Measures
    //***********************************************************************//

    class_name = "MeasuresC" + bitsize;
    py::class_<Measures<PrecisionT>>(m, class_name.c_str(), py::module_local())
        .def(py::init<const StateVectorRawCPU<PrecisionT> &>())
        .def("probs",
             [](Measures<PrecisionT> &M, const std::vector<size_t> &wires) {
                 return py::array_t<ParamT>(py::cast(M.probs(wires)));
             })
        .def("probs",
             [](Measures<PrecisionT> &M) {
                 return py::array_t<ParamT>(py::cast(M.probs()));
             })
        .def("expval",
             static_cast<PrecisionT (Measures<PrecisionT>::*)(
                 const std::string &, const std::vector<size_t> &)>(
                 &Measures<PrecisionT>::expval),
             "Expected value of an operation by name.")
        .def(
            "expval",
            [](Measures<PrecisionT> &M,
               const std::shared_ptr<Observable<PrecisionT>> &ob) {
                return M.expval(*ob);
            },
            "Expected value of an operation object.")
        .def(
            "expval",
            [](Measures<PrecisionT> &M, const np_arr_sparse_ind row_map,
               const np_arr_sparse_ind entries, const np_arr_c values) {
                return M.expval(
                    static_cast<sparse_index_type *>(row_map.request().ptr),
                    static_cast<sparse_index_type>(row_map.request().size),
                    static_cast<sparse_index_type *>(entries.request().ptr),
                    static_cast<std::complex<PrecisionT> *>(
                        values.request().ptr),
                    static_cast<sparse_index_type>(values.request().size));
            },
            "Expected value of a sparse Hamiltonian.")
        .def("generate_samples",
             [](Measures<PrecisionT> &M, size_t num_wires, size_t num_shots) {
                 auto &&result = M.generate_samples(num_shots);
                 const size_t ndim = 2;
                 const std::vector<size_t> shape{num_shots, num_wires};
                 constexpr auto sz = sizeof(size_t);
                 const std::vector<size_t> strides{sz * num_wires, sz};
                 // return 2-D NumPy array
                 return py::array(py::buffer_info(
                     result.data(), /* data as contiguous array  */
                     sz,            /* size of one scalar        */
                     py::format_descriptor<size_t>::format(), /* data type */
                     ndim,   /* number of dimensions      */
                     shape,  /* shape of the matrix       */
                     strides /* strides for each axis     */
                     ));
             })
        .def("var",
             [](Measures<PrecisionT> &M, const std::string &operation,
                const std::vector<size_t> &wires) {
                 return M.var(operation, wires);
             })
        .def("var",
             static_cast<PrecisionT (Measures<PrecisionT>::*)(
                 const std::string &, const std::vector<size_t> &)>(
                 &Measures<PrecisionT>::var),
             "Variance of an operation by name.")
        .def(
            "var",
            [](Measures<PrecisionT> &M,
               const std::shared_ptr<Observable<PrecisionT>> &ob) {
                return M.var(*ob);
            },
            "Variance of an operation object.")
        .def(
            "var",
            [](Measures<PrecisionT> &M, const np_arr_sparse_ind row_map,
               const np_arr_sparse_ind entries, const np_arr_c values) {
                return M.var(
                    static_cast<sparse_index_type *>(row_map.request().ptr),
                    static_cast<sparse_index_type>(row_map.request().size),
                    static_cast<sparse_index_type *>(entries.request().ptr),
                    static_cast<std::complex<PrecisionT> *>(
                        values.request().ptr),
                    static_cast<sparse_index_type>(values.request().size));
            },
            "Expected value of a sparse Hamiltonian.");
}

template <class PrecisionT, class ParamT>
void registerAlgorithms(py::module_ &m) {
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    //***********************************************************************//
    //                              Observable
    //***********************************************************************//

    using np_arr_c = py::array_t<std::complex<ParamT>, py::array::c_style>;
    using np_arr_r = py::array_t<ParamT, py::array::c_style>;

    std::string class_name;

    class_name = "ObservableC" + bitsize;
    py::class_<Observable<PrecisionT>, std::shared_ptr<Observable<PrecisionT>>>(
        m, class_name.c_str(), py::module_local());

    class_name = "NamedObsC" + bitsize;
    py::class_<NamedObs<PrecisionT>, std::shared_ptr<NamedObs<PrecisionT>>,
               Observable<PrecisionT>>(m, class_name.c_str(),
                                       py::module_local())
        .def(py::init(
            [](const std::string &name, const std::vector<size_t> &wires) {
                return NamedObs<PrecisionT>(name, wires);
            }))
        .def("__repr__", &NamedObs<PrecisionT>::getObsName)
        .def("get_wires", &NamedObs<PrecisionT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const NamedObs<PrecisionT> &self, py::handle other) -> bool {
                if (!py::isinstance<NamedObs<PrecisionT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<NamedObs<PrecisionT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HermitianObsC" + bitsize;
    py::class_<HermitianObs<PrecisionT>,
               std::shared_ptr<HermitianObs<PrecisionT>>,
               Observable<PrecisionT>>(m, class_name.c_str(),
                                       py::module_local())
        .def(py::init([](const np_arr_c &matrix,
                         const std::vector<size_t> &wires) {
            auto buffer = matrix.request();
            const auto *ptr =
                static_cast<std::complex<PrecisionT> *>(buffer.ptr);
            return HermitianObs<PrecisionT>(
                std::vector<std::complex<PrecisionT>>(ptr, ptr + buffer.size),
                wires);
        }))
        .def("__repr__", &HermitianObs<PrecisionT>::getObsName)
        .def("get_wires", &HermitianObs<PrecisionT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const HermitianObs<PrecisionT> &self, py::handle other) -> bool {
                if (!py::isinstance<HermitianObs<PrecisionT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<HermitianObs<PrecisionT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "TensorProdObsC" + bitsize;
    py::class_<TensorProdObs<PrecisionT>,
               std::shared_ptr<TensorProdObs<PrecisionT>>,
               Observable<PrecisionT>>(m, class_name.c_str(),
                                       py::module_local())
        .def(py::init(
            [](const std::vector<std::shared_ptr<Observable<PrecisionT>>>
                   &obs) { return TensorProdObs<PrecisionT>(obs); }))
        .def("__repr__", &TensorProdObs<PrecisionT>::getObsName)
        .def("get_wires", &TensorProdObs<PrecisionT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const TensorProdObs<PrecisionT> &self,
               py::handle other) -> bool {
                if (!py::isinstance<TensorProdObs<PrecisionT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<TensorProdObs<PrecisionT>>();
                return self == other_cast;
            },
            "Compare two observables");

    class_name = "HamiltonianC" + bitsize;
    using ObsPtr = std::shared_ptr<Observable<PrecisionT>>;
    py::class_<Hamiltonian<PrecisionT>,
               std::shared_ptr<Hamiltonian<PrecisionT>>,
               Observable<PrecisionT>>(m, class_name.c_str(),
                                       py::module_local())
        .def(py::init([](const np_arr_r &coeffs,
                         const std::vector<ObsPtr> &obs) {
            auto buffer = coeffs.request();
            const auto ptr = static_cast<const ParamT *>(buffer.ptr);
            return Hamiltonian<PrecisionT>{std::vector(ptr, ptr + buffer.size),
                                           obs};
        }))
        .def("__repr__", &Hamiltonian<PrecisionT>::getObsName)
        .def("get_wires", &Hamiltonian<PrecisionT>::getWires,
             "Get wires of observables")
        .def(
            "__eq__",
            [](const Hamiltonian<PrecisionT> &self, py::handle other) -> bool {
                if (!py::isinstance<Hamiltonian<PrecisionT>>(other)) {
                    return false;
                }
                auto other_cast = other.cast<Hamiltonian<PrecisionT>>();
                return self == other_cast;
            },
            "Compare two observables");

    //***********************************************************************//
    //                              Operations
    //***********************************************************************//

    class_name = "OpsStructC" + bitsize;
    py::class_<OpsData<PrecisionT>>(m, class_name.c_str(), py::module_local())
        .def(py::init<
             const std::vector<std::string> &,
             const std::vector<std::vector<ParamT>> &,
             const std::vector<std::vector<size_t>> &,
             const std::vector<bool> &,
             const std::vector<std::vector<std::complex<PrecisionT>>> &>())
        .def("__repr__", [](const OpsData<PrecisionT> &ops) {
            using namespace Pennylane::Util;
            std::ostringstream ops_stream;
            for (size_t op = 0; op < ops.getSize(); op++) {
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
     * Create operation list
     * */
    std::string function_name = "create_ops_list_C" + bitsize;
    m.def(
        function_name.c_str(),
        [](const std::vector<std::string> &ops_name,
           const std::vector<std::vector<PrecisionT>> &ops_params,
           const std::vector<std::vector<size_t>> &ops_wires,
           const std::vector<bool> &ops_inverses,
           const std::vector<np_arr_c> &ops_matrices) {
            std::vector<std::vector<std::complex<PrecisionT>>> conv_matrices(
                ops_matrices.size());
            for (size_t op = 0; op < ops_name.size(); op++) {
                const auto m_buffer = ops_matrices[op].request();
                if (m_buffer.size) {
                    const auto m_ptr =
                        static_cast<const std::complex<ParamT> *>(m_buffer.ptr);
                    conv_matrices[op] = std::vector<std::complex<ParamT>>{
                        m_ptr, m_ptr + m_buffer.size};
                }
            }
            return OpsData<PrecisionT>{ops_name, ops_params, ops_wires,
                                       ops_inverses, conv_matrices};
        },
        "Create a list of operations from data.");
    m.def(
        "adjoint_jacobian",
        [](const StateVectorRawCPU<PrecisionT> &sv,
           const std::vector<std::shared_ptr<Observable<PrecisionT>>>
               &observables,
           const OpsData<PrecisionT> &operations,
           const std::vector<size_t> &trainableParams) {
            std::vector<PrecisionT> jac(
                observables.size() * trainableParams.size(), PrecisionT{0.0});

            const JacobianData<PrecisionT> jd{operations.getTotalNumParams(),
                                              sv.getLength(),
                                              sv.getData(),
                                              observables,
                                              operations,
                                              trainableParams};

            adjointJacobian(std::span{jac}, jd);

            return py::array_t<ParamT>(py::cast(jac));
        },
        "Compute jacobian of the circuit using the adjoint method.");

    m.def(
        "statevector_vjp",
        /* Do not cast non-conforming array. Argument trainableParams should
         * only contain indices for operations.
         */
        [](const StateVectorRawCPU<PrecisionT> &sv,
           const OpsData<PrecisionT> &operations, const np_arr_c &dy,
           const std::vector<size_t> &trainableParams) {
            std::vector<std::complex<PrecisionT>> vjp(
                trainableParams.size(), std::complex<PrecisionT>{});

            const JacobianData<PrecisionT> jd{operations.getTotalNumParams(),
                                              sv.getLength(),
                                              sv.getData(),
                                              {},
                                              operations,
                                              trainableParams};
            const auto buffer = dy.request();

            statevectorVJP<PrecisionT>(
                std::span{vjp}, jd,
                std::span{
                    static_cast<const std::complex<PrecisionT> *>(buffer.ptr),
                    static_cast<size_t>(buffer.size)});

            return py::array_t<std::complex<PrecisionT>>(py::cast(vjp));
        },
        "Compute jacobian of the circuit using the adjoint method.");
}

/**
 * @brief Add C++ classes, methods and functions to Python module.
 */
PYBIND11_MODULE(lightning_qubit_ops, // NOLINT: No control over Pybind internals
                m) {
    // Suppress doxygen autogenerated signatures

    py::options options;
    options.disable_function_signatures();

    m.doc() = "lightning.qubit apply() method";
    m.def(
        "apply",
        py::overload_cast<py::array_t<complex<double>> &,
                          const vector<string> &,
                          const vector<vector<size_t>> &, const vector<bool> &,
                          const vector<vector<double>> &>(apply<double>),
        "lightning.qubit apply() method");
    m.def(
        "apply",
        py::overload_cast<py::array_t<complex<float>> &, const vector<string> &,
                          const vector<vector<size_t>> &, const vector<bool> &,
                          const vector<vector<float>> &>(apply<float>),
        "lightning.qubit apply() method");

    m.def("generateBitPatterns",
          py::overload_cast<const vector<size_t> &, size_t>(
              &Gates::generateBitPatterns),
          "Get statevector indices for gate application");
    m.def("getIndicesAfterExclusion",
          py::overload_cast<const vector<size_t> &, size_t>(
              &Gates::getIndicesAfterExclusion),
          "Get statevector indices for gate application");

    /* Algorithms submodule */
    py::module_ alg_submodule = m.def_submodule(
        "adjoint_diff", "A submodule for adjoint differentiation method.");

    registerAlgorithms<float, float>(alg_submodule);
    registerAlgorithms<double, double>(alg_submodule);

    /* Add CPUMemoryModel enum class */
    py::enum_<CPUMemoryModel>(m, "CPUMemoryModel")
        .value("Unaligned", CPUMemoryModel::Unaligned)
        .value("Aligned256", CPUMemoryModel::Aligned256)
        .value("Aligned512", CPUMemoryModel::Aligned512);

    /* Add array */
    m.def("allocate_aligned_array", &allocateAlignedArray,
          "Get numpy array whose underlying data is aligned.");
    m.def("get_alignment", &getNumpyArrayAlignment,
          "Get alignment of an underlying data for a numpy array.");
    m.def("best_alignment", &bestCPUMemoryModel,
          "Best memory alignment. for the simulator.");

    /* Add compile info */
    m.def("compile_info", &getCompileInfo, "Compiled binary information.");

    /* Add runtime info */
    m.def("runtime_info", &getRuntimeInfo, "Runtime information.");

    /* Add Kokkos and Kokkos Kernels info */
    m.def("Kokkos_info", &getKokkosInfo,
          "Kokkos and Kokkos Kernels information.");

    lightning_class_bindings<float, float>(m);
    lightning_class_bindings<double, double>(m);
}
