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
#include "SelectKernel.hpp"

#include "pybind11/pybind11.h"

/// @cond DEV
namespace {
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
    //                              Observable
    //***********************************************************************//

    class_name = "ObsStructC" + bitsize;
    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;
    using np_arr_r =
        py::array_t<ParamT, py::array::c_style | py::array::forcecast>;

    using obs_data_var = std::variant<std::monostate, np_arr_r, np_arr_c>;
    py::class_<ObsDatum<PrecisionT>>(m, class_name.c_str(), py::module_local())
        .def(py::init([](const std::vector<std::string> &names,
                         const std::vector<obs_data_var> &params,
                         const std::vector<std::vector<size_t>> &wires) {
            std::vector<typename ObsDatum<PrecisionT>::param_var_t> conv_params(
                params.size());
            for (size_t p_idx = 0; p_idx < params.size(); p_idx++) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        if constexpr (std::is_same_v<p_t, np_arr_c>) {
                            auto buffer = param.request();
                            auto ptr =
                                static_cast<std::complex<ParamT> *>(buffer.ptr);
                            if (buffer.size) {
                                conv_params[p_idx] =
                                    std::vector<std::complex<ParamT>>{
                                        ptr, ptr + buffer.size};
                            }
                        } else if constexpr (std::is_same_v<p_t, np_arr_r>) {
                            auto buffer = param.request();

                            auto *ptr = static_cast<ParamT *>(buffer.ptr);
                            if (buffer.size) {
                                conv_params[p_idx] =
                                    std::vector<ParamT>{ptr, ptr + buffer.size};
                            }
                        } else {
                            PL_ABORT(
                                "Parameter datatype not current supported");
                        }
                    },
                    params[p_idx]);
            }
            return ObsDatum<PrecisionT>(names, conv_params, wires);
        }))
        .def("__repr__",
             [](const ObsDatum<PrecisionT> &obs) {
                 using namespace Pennylane::Util;
                 std::ostringstream obs_stream;
                 std::string obs_name = obs.getObsName()[0];
                 for (size_t o = 1; o < obs.getObsName().size(); o++) {
                     if (o < obs.getObsName().size()) {
                         obs_name += " @ ";
                     }
                     obs_name += obs.getObsName()[o];
                 }
                 obs_stream << "'wires' : " << obs.getObsWires();
                 return "Observable: { 'name' : " + obs_name + ", " +
                        obs_stream.str() + " }";
             })
        .def("get_name",
             [](const ObsDatum<PrecisionT> &obs) { return obs.getObsName(); })
        .def("get_wires",
             [](const ObsDatum<PrecisionT> &obs) { return obs.getObsWires(); })
        .def("get_params", [](const ObsDatum<PrecisionT> &obs) {
            py::list params;
            for (size_t i = 0; i < obs.getObsParams().size(); i++) {
                std::visit(
                    [&](const auto &param) {
                        using p_t = std::decay_t<decltype(param)>;
                        if constexpr (std::is_same_v<
                                          p_t,
                                          std::vector<std::complex<ParamT>>>) {
                            params.append(py::array_t<std::complex<ParamT>>(
                                py::cast(param)));
                        } else if constexpr (std::is_same_v<
                                                 p_t, std::vector<ParamT>>) {
                            params.append(py::array_t<ParamT>(py::cast(param)));
                        } else if constexpr (std::is_same_v<p_t,
                                                            std::monostate>) {
                            params.append(py::list{});
                        } else {
                            throw("Unsupported data type");
                        }
                    },
                    obs.getObsParams()[i]);
            }
            return params;
        });

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

    //***********************************************************************//
    //                              Adjoint Jacobian
    //***********************************************************************//

    class_name = "AdjointJacobianC" + bitsize;
    py::class_<AdjointJacobian<PrecisionT>>(m, class_name.c_str(),
                                            py::module_local())
        .def(py::init<>())
        .def("create_ops_list",
             [](AdjointJacobian<PrecisionT> &adj,
                const std::vector<std::string> &ops_name,
                const std::vector<np_arr_r> &ops_params,
                const std::vector<std::vector<size_t>> &ops_wires,
                const std::vector<bool> &ops_inverses,
                const std::vector<np_arr_c> &ops_matrices) {
                 std::vector<std::vector<PrecisionT>> conv_params(
                     ops_params.size());
                 std::vector<std::vector<std::complex<PrecisionT>>>
                     conv_matrices(ops_matrices.size());
                 static_cast<void>(adj);
                 for (size_t op = 0; op < ops_name.size(); op++) {
                     const auto p_buffer = ops_params[op].request();
                     const auto m_buffer = ops_matrices[op].request();
                     if (p_buffer.size) {
                         const auto *const p_ptr =
                             static_cast<const ParamT *>(p_buffer.ptr);
                         conv_params[op] =
                             std::vector<ParamT>{p_ptr, p_ptr + p_buffer.size};
                     }
                     if (m_buffer.size) {
                         const auto m_ptr =
                             static_cast<const std::complex<ParamT> *>(
                                 m_buffer.ptr);
                         conv_matrices[op] = std::vector<std::complex<ParamT>>{
                             m_ptr, m_ptr + m_buffer.size};
                     }
                 }
                 return OpsData<PrecisionT>{ops_name, conv_params, ops_wires,
                                            ops_inverses, conv_matrices};
             })
        .def("adjoint_jacobian", &AdjointJacobian<PrecisionT>::adjointJacobian)
        .def("adjoint_jacobian",
             [](AdjointJacobian<PrecisionT> &adj,
                const StateVectorRaw<PrecisionT> &sv,
                const std::vector<ObsDatum<PrecisionT>> &observables,
                const OpsData<PrecisionT> &operations,
                const std::vector<size_t> &trainableParams, size_t num_params) {
                 std::vector<PrecisionT> jac(observables.size() * num_params,
                                             0);

                 const JacobianData<PrecisionT> jd{
                     num_params,  sv.getLength(), sv.getData(),
                     observables, operations,     trainableParams};

                 adj.adjointJacobian(jac, jd);

                 return py::array_t<ParamT>(py::cast(jac));
             });

    //***********************************************************************//
    //                              VJP
    //***********************************************************************//

    class_name = "VectorJacobianProductC" + bitsize;
    py::class_<VectorJacobianProduct<PrecisionT>>(m, class_name.c_str(),
                                                  py::module_local())
        .def(py::init<>())
        .def("create_ops_list",
             [](VectorJacobianProduct<PrecisionT> &v,
                const std::vector<std::string> &ops_name,
                const std::vector<np_arr_r> &ops_params,
                const std::vector<std::vector<size_t>> &ops_wires,
                const std::vector<bool> &ops_inverses,
                const std::vector<np_arr_c> &ops_matrices) {
                 std::vector<std::vector<PrecisionT>> conv_params(
                     ops_params.size());
                 std::vector<std::vector<std::complex<PrecisionT>>>
                     conv_matrices(ops_matrices.size());
                 static_cast<void>(v);
                 for (size_t op = 0; op < ops_name.size(); op++) {
                     const auto p_buffer = ops_params[op].request();
                     const auto m_buffer = ops_matrices[op].request();
                     if (p_buffer.size) {
                         const auto *const p_ptr =
                             static_cast<const ParamT *>(p_buffer.ptr);
                         conv_params[op] =
                             std::vector<ParamT>{p_ptr, p_ptr + p_buffer.size};
                     }
                     if (m_buffer.size) {
                         const auto m_ptr =
                             static_cast<const std::complex<ParamT> *>(
                                 m_buffer.ptr);
                         conv_matrices[op] = std::vector<std::complex<ParamT>>{
                             m_ptr, m_ptr + m_buffer.size};
                     }
                 }
                 return OpsData<PrecisionT>{ops_name, conv_params, ops_wires,
                                            ops_inverses, conv_matrices};
             })
        .def("compute_vjp_from_jac",
             &VectorJacobianProduct<PrecisionT>::computeVJP)
        .def("compute_vjp_from_jac",
             [](VectorJacobianProduct<PrecisionT> &v,
                const std::vector<PrecisionT> &jac,
                const std::vector<PrecisionT> &dy_row, size_t m, size_t n) {
                 std::vector<PrecisionT> vjp_res(n);
                 v.computeVJP(vjp_res, jac, dy_row, m, n);
                 return py::array_t<ParamT>(py::cast(vjp_res));
             })
        .def("vjp_fn",
             [](VectorJacobianProduct<PrecisionT> &v,
                const std::vector<PrecisionT> &dy, size_t num_params) {
                 auto fn = v.vectorJacobianProduct(dy, num_params);
                 return py::cpp_function(
                     [fn, num_params](
                         const StateVectorRaw<PrecisionT> &sv,
                         const std::vector<ObsDatum<PrecisionT>> &observables,
                         const OpsData<PrecisionT> &operations,
                         const std::vector<size_t> &trainableParams) {
                         const JacobianData<PrecisionT> jd{
                             num_params,  sv.getLength(), sv.getData(),
                             observables, operations,     trainableParams};
                         return py::array_t<ParamT>(py::cast(fn(jd)));
                     });
             });

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
        .def("var", [](Measures<PrecisionT> &M, const std::string &operation,
                       const std::vector<size_t> &wires) {
            return M.var(operation, wires);
        });
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

    /* Add EXPORTED_KERNELS */
    std::vector<std::pair<std::string, std::string>> exported_kernel_ops;

    for (const auto kernel : kernels_to_pyexport) {
        const auto kernel_name = lookup(kernel_id_name_pairs, kernel);
        const auto implemented_gates = implementedGatesForKernel(kernel);
        for (const auto gate_op : implemented_gates) {
            const auto gate_name =
                std::string(lookup(Constant::gate_names, gate_op));
            exported_kernel_ops.emplace_back(kernel_name, gate_name);
        }
    }

    m.attr("EXPORTED_KERNEL_OPS") = py::cast(exported_kernel_ops);

    /* Add DEFAULT_KERNEL_FOR_OPS */
    std::map<std::string, std::string> default_kernel_ops_map;
    for (const auto &[gate_op, name] : Constant::gate_names) {
        const auto kernel = lookup(Constant::default_kernel_for_gates, gate_op);
        const auto kernel_name = Util::lookup(kernel_id_name_pairs, kernel);
        default_kernel_ops_map.emplace(std::string(name), kernel_name);
    }
    m.attr("DEFAULT_KERNEL_FOR_OPS") = py::cast(default_kernel_ops_map);

    lightning_class_bindings<float, float>(m);
    lightning_class_bindings<double, double>(m);
}
