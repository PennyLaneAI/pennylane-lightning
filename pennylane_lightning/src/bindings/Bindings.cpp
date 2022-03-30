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

#include "StateVecAdjDiff.hpp"
#include "GateUtil.hpp"
#include "SelectKernel.hpp"

#include "pybind11/pybind11.h"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Util;
using namespace Pennylane::Algorithms;
using namespace Pennylane::Gates;

using Pennylane::StateVectorRaw;

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
void lightning_class_bindings(py::module &m) {
    // Enable module name to be based on size of complex datatype
    const std::string bitsize =
        std::to_string(sizeof(std::complex<PrecisionT>) * 8);

    //***********************************************************************//
    //                              StateVector
    //***********************************************************************//

    std::string class_name = "StateVectorC" + bitsize;
    auto pyclass = py::class_<StateVectorRaw<PrecisionT>>(m, class_name.c_str(),
                                                          py::module_local());
    pyclass.def(py::init(&create<PrecisionT>));

    registerKernelsToPyexport<PrecisionT, ParamT>(pyclass);

    //***********************************************************************//
    //                              Measures
    //***********************************************************************//

    class_name = "MeasuresC" + bitsize;
    py::class_<Measures<PrecisionT>>(m, class_name.c_str(), py::module_local())
        .def(py::init<const StateVectorRaw<PrecisionT> &>())
        .def("probs",
             [](Measures<PrecisionT> &M, const std::vector<size_t> &wires) {
                 if (wires.empty()) {
                     return py::array_t<ParamT>(py::cast(M.probs()));
                 }
                 return py::array_t<ParamT>(py::cast(M.probs(wires)));
             })
        .def("expval",
             [](Measures<PrecisionT> &M, const std::string &operation,
                const std::vector<size_t> &wires) {
                 return M.expval(operation, wires);
             })
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
        .def("var", [](Measures<PrecisionT> &M, const std::string &operation,
                       const std::vector<size_t> &wires) {
            return M.var(operation, wires);
        });
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

    class_name = "ObsTermC" + bitsize;
    py::class_<ObsTerm<PrecisionT>, std::shared_ptr<ObsTerm<PrecisionT>>,
               Observable<PrecisionT>>(m, class_name.c_str(),
                                       py::module_local())
        .def(py::init([](const std::vector<std::string> &names,
                         const std::vector<np_arr_c> &params,
                         const std::vector<std::vector<size_t>> &wires) {
            std::vector<typename ObsTerm<PrecisionT>::MatrixT> conv_params;
            conv_params.reserve(params.size());
            for (size_t p_idx = 0; p_idx < params.size(); p_idx++) {
                auto buffer = params[p_idx].request();
                auto ptr = static_cast<std::complex<ParamT> *>(buffer.ptr);
                if (buffer.size) {
                    conv_params.emplace_back(ptr, ptr + buffer.size);
                }
            }
            return ObsTerm<PrecisionT>(names, conv_params, wires);
        }))
        .def("__repr__", &ObsTerm<PrecisionT>::toString)
        .def("get_name", &ObsTerm<PrecisionT>::getObsName,
             "Get names of observables")
        .def("get_wires", &ObsTerm<PrecisionT>::getObsWires,
             "Get wires of observables");

    class_name = "HamiltonianC" + bitsize;
    using ObsTermPtr = std::shared_ptr<ObsTerm<PrecisionT>>;
    py::class_<Hamiltonian<PrecisionT>,
               std::shared_ptr<Hamiltonian<PrecisionT>>,
               Observable<PrecisionT>>(m, class_name.c_str(),
                                       py::module_local())
        .def(py::init(
            [](const np_arr_r &coeffs, const std::vector<ObsTermPtr> &terms) {
                auto buffer = coeffs.request();
                const auto ptr = static_cast<const ParamT *>(buffer.ptr);
                return Hamiltonian{{ptr, ptr + buffer.size}, terms};
            }))
        .def("__repr__", &Hamiltonian<PrecisionT>::toString);

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
        [](const StateVectorRaw<PrecisionT> &sv,
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

            adjointJacobian(jac, jd);

            return py::array_t<ParamT>(py::cast(jac));
        },
        "Compute jacobian of the circuit using the adjoint method.");

    m.def(
        "statevector_vjp",
        [](const StateVectorRaw<PrecisionT> &sv,
           const OpsData<PrecisionT> &operations,
           /* Do not cast non-conforming array */
           const py::array_t<PrecisionT, py::array::c_style> dy,
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

            statevectorVJP<PrecisionT>(vjp, jd,
                    static_cast<std::complex<PrecisionT>*>(buffer.ptr), buffer.itemsize);

            return py::array_t<ParamT>(py::cast(vjp));
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

    /* Add compile info */
    m.def("compile_info", &getCompileInfo, "Compiled binary information.");

    /* Add runtime info */
    m.def("runtime_info", &getRuntimeInfo, "Runtime information.");

    /* Add EXPORTED_KERNELS */
    std::vector<std::pair<std::string, std::string>> exported_kernel_ops;

    for (const auto kernel : kernels_to_pyexport) {
        const auto kernel_name = lookup(kernel_id_name_pairs, kernel);
        const auto implemented_gates = implementedGatesForKernel(kernel);
        for (const auto gate_op : implemented_gates) {
            const auto gate_name =
                std::string(lookup(Gates::Constant::gate_names, gate_op));
            exported_kernel_ops.emplace_back(kernel_name, gate_name);
        }
    }

    m.attr("EXPORTED_KERNEL_OPS") = py::cast(exported_kernel_ops);

    /* Add DEFAULT_KERNEL_FOR_OPS */
    namespace GateConstant = Gates::Constant;
    std::map<std::string, std::string> default_kernel_ops_map;
    for (const auto &[gate_op, name] : GateConstant::gate_names) {
        const auto kernel =
            lookup(GateConstant::default_kernel_for_gates, gate_op);
        const auto kernel_name = lookup(kernel_id_name_pairs, kernel);
        default_kernel_ops_map.emplace(std::string(name), kernel_name);
    }
    m.attr("DEFAULT_KERNEL_FOR_OPS") = py::cast(default_kernel_ops_map);

    lightning_class_bindings<float, float>(m);
    lightning_class_bindings<double, double>(m);
}
