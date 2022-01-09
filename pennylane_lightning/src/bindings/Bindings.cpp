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

#include <set>
#include <tuple>
#include <vector>

#include <iostream>

#include "AdjointDiff.hpp"
#include "IndicesUtil.hpp"
#include "JacobianProd.hpp"
#include "StateVectorBase.hpp"
#include "StateVectorRaw.hpp"
#include "pybind11/complex.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

/// @cond DEV
namespace {
using namespace Pennylane::Algorithms;
using Pennylane::StateVectorBase;
using Pennylane::StateVectorRaw;

using Pennylane::Internal::callGateOps;
using Pennylane::Internal::implementedGatesForKernel;

using Pennylane::Internal::GateOpsFuncPtrPairs;

using std::complex;
using std::set;
using std::string;
using std::vector;
} // namespace
/// @endcond

namespace py = pybind11;

/**
 * @brief Create a `%StateVector` object from a 1D numpy complex data array.
 *
 * @tparam PrecisionT Precision data type
 * @param numpyArray Numpy data array.
 * @return StateVector<PrecisionT> `%StateVector` object.
 */
template <class PrecisionT = double>
static auto create(py::array_t<complex<PrecisionT>> &numpyArray)
    -> StateVectorRaw<PrecisionT> {
    py::buffer_info numpyArrayInfo = numpyArray.request();

    if (numpyArrayInfo.ndim != 1) {
        throw std::invalid_argument(
            "NumPy array must be a 1-dimensional array");
    }
    if (numpyArrayInfo.itemsize != sizeof(complex<PrecisionT>)) {
        throw std::invalid_argument(
            "NumPy array must be of type np.complex64 or np.complex128");
    }
    auto *data_ptr = static_cast<complex<PrecisionT> *>(numpyArrayInfo.ptr);
    return StateVectorRaw<PrecisionT>(
        {data_ptr, static_cast<size_t>(numpyArrayInfo.shape[0])});
}

/**
 * @brief Apply given list of operations to Numpy data array using C++
 * `%StateVector` class.
 *
 * @tparam PrecisionT Precision data type
 * @param stateNumpyArray Complex numpy data array representing statevector.
 * @param ops Operations to apply to the statevector using the C++ backend.
 * @param wires Wires on which to apply each operation from `ops`.
 * @param inverse Indicate whether a given operation is an inverse.
 * @param params Parameters for each given operation in `ops`.
 */
template <class PrecisionT = double>
void apply(py::array_t<complex<PrecisionT>> &stateNumpyArray,
           const vector<string> &ops, const vector<vector<size_t>> &wires,
           const vector<bool> &inverse,
           const vector<vector<PrecisionT>> &params) {
    auto state = create<PrecisionT>(stateNumpyArray);
    state.applyOperations(ops, wires, inverse, params);
}

/**
 * @brief Return a specific lambda function to return 
 *
 * We do not expect template paramters kernel and gate_op can be
 * function paramters as we want the lambe function to be a stateless.
 */
template <class PrecisionT, class ParamT, KernelType kernel,
          GateOperations gate_op>
constexpr auto getLambdaForKernelGateOp () {
    static_assert(
        array_has_elt(SelectGateOps<PrecisionT, kernel>::implemented_gates,
                      gate_op),
        "The operator to register must be implemented.");

    if constexpr (gate_op != GateOperations::Matrix) {
        constexpr size_t num_params =
            static_lookup<gate_op>(Constant::gate_num_params);

        return [](StateVectorRaw<PrecisionT> &st,
                                        const std::vector<size_t> &wires,
                                        bool inverse,
                                        const std::vector<ParamT> &params) {
            auto func_ptr = static_lookup<gate_op>(
                GateOpsFuncPtrPairs<PrecisionT, ParamT, kernel, num_params>::value);
            callGateOps(func_ptr, st.getData(), st.getNumQubits(), wires, inverse,
                        params);
        };
    } else {
        return [](StateVectorRaw<PrecisionT> &st,
                  const py::array_t<std::complex<PrecisionT>,
                             py::array::c_style | py::array::forcecast> &matrix,
                  const std::vector<size_t> &wires, bool inverse = false) {
            st.template applyMatrix_<kernel>(
                static_cast<std::complex<PrecisionT> *>(matrix.request().ptr),
                wires, inverse);
        };
    }
};

/// @cond DEV
template <class PrecisionT, class ParamT, KernelType kernel, size_t gate_idx>
constexpr auto getGateOpLambdaPairsIter() {
    if constexpr (gate_idx < SelectGateOps<PrecisionT, kernel>::implemented_gates.size()) {
        constexpr auto gate_op = SelectGateOps<PrecisionT, kernel>::implemented_gates[gate_idx];
        return prepend_to_tuple(
                std::pair{gate_op, getLambdaForKernelGateOp<PrecisionT, ParamT, kernel, gate_op>()},
                getGateOpLambdaPairsIter<PrecisionT, ParamT, kernel, gate_idx+1>());
    } else {
        return std::tuple{};
    }
}
/// @endcond

/**
 * @brief Create a tuple of lambda functions to bind
 */
template <class PrecisionT, class ParamT, KernelType kernel>
constexpr auto getGateOpLambdaPairs() {
    return getGateOpLambdaPairsIter<PrecisionT, ParamT, kernel, 0>();
}

/**
 * @brief For given kernel, register all implemented gate operations and apply
 * matrix.
 *
 * @tparam PrecisionT type for state-vector precision
 * @tparam ParamT type for parameters for the gate operation
 * @tparam kernel Kernel to register
 * @tparam PyClass pybind11 class type
 */
template <class PrecisionT, class ParamT, KernelType kernel, class PyClass>
void registerImplementedGatesForKernel(PyClass &pyclass) {
    const auto kernel_name =
        std::string(lookup(Constant::available_kernels, kernel));

    constexpr auto gate_op_lambda_pairs = getGateOpLambdaPairs<PrecisionT, ParamT,
              kernel>();

    auto registerToPyclass = [&pyclass, &kernel_name] (auto&& gate_op_lambda_pair) {
        const auto& [gate_op, func] = gate_op_lambda_pair;
        if (gate_op == GateOperations::Matrix) {
            const std::string name = "applyMatrix_" + kernel_name;
            const std::string doc = "Apply a given matrix to wires.";
            pyclass.def(name.c_str(), func, doc.c_str());
        } else {
            const auto gate_name =
                std::string(lookup(Constant::gate_names, gate_op));
            // auto func = lookup(gate_op_pairs, gate_op);
            const std::string name = gate_name + "_" + kernel_name;
            const std::string doc = "Apply the " + gate_name + " gate using " +
                                    kernel_name + " kernel.";
            pyclass.def(name.c_str(), func, doc.c_str());
        }
        return gate_op;
    };

    const auto registeredGateOps = 
        std::apply([&registerToPyclass](auto ...x){
            std::make_tuple(registerToPyclass(x)...);
        }, gate_op_lambda_pairs);

    assert(tuple_to_array<GateOperations>(registeredGateOps) == 
           SelectGateOps<fp_t, kernel>::implemented_gates); // double check in debug mode
}

template <class PrecisionT, class ParamT, size_t kernel_idx, class PyClass>
void registerKernelsToPyexportIter(PyClass &pyclass) {
    if constexpr (kernel_idx < Constant::kernels_to_pyexport.size()) {
        constexpr auto kernel = Constant::kernels_to_pyexport[kernel_idx];
        registerImplementedGatesForKernel<PrecisionT, ParamT, kernel>(pyclass);
        registerKernelsToPyexportIter<PrecisionT, ParamT, kernel_idx + 1>(
            pyclass);
    }
}

/**
 * @brief register gates for each kernel in kernels_to_pyexport
 */
template <class PrecisionT, class ParamT, class PyClass>
void registerKernelsToPyexport(PyClass &pyclass) {
    registerKernelsToPyexportIter<PrecisionT, ParamT, 0>(pyclass);
}

/**
 * @brief Templated class to build all required precisions for Python module.
 *
 * @tparam PrecisionT Precision of the statevector data.
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
    auto pyclass =
        py::class_<StateVectorRaw<PrecisionT>>(m, class_name.c_str());
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
    py::class_<ObsDatum<PrecisionT>>(m, class_name.c_str())
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
    py::class_<OpsData<PrecisionT>>(m, class_name.c_str())
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

    class_name = "AdjointJacobianC" + bitsize;
    py::class_<AdjointJacobian<PrecisionT>>(m, class_name.c_str())
        .def(py::init<>())
        .def("create_ops_list", &AdjointJacobian<PrecisionT>::createOpsData)
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
                 std::vector<std::vector<PrecisionT>> jac(
                     observables.size(),
                     std::vector<PrecisionT>(num_params, 0));
                 adj.adjointJacobian(sv.getData(), sv.getLength(), jac,
                                     observables, operations, trainableParams);
                 return py::array_t<ParamT>(py::cast(jac));
             });

    class_name = "VectorJacobianProductC" + bitsize;
    py::class_<VectorJacobianProduct<PrecisionT>>(m, class_name.c_str())
        .def(py::init<>())
        .def("create_ops_list",
             &VectorJacobianProduct<PrecisionT>::createOpsData)
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
                 v._computeVJP(vjp_res, jac, dy_row, m, n);
                 return py::array_t<ParamT>(py::cast(vjp_res));
             })
        .def("vjp", &VectorJacobianProduct<PrecisionT>::vectorJacobianProduct)
        .def("vjp", [](VectorJacobianProduct<PrecisionT> &v,
                       const std::vector<PrecisionT> &dy,
                       const StateVectorRaw<PrecisionT> &sv,
                       const std::vector<ObsDatum<PrecisionT>> &observables,
                       const OpsData<PrecisionT> &operations,
                       const std::vector<size_t> &trainableParams,
                       size_t num_params) {
            std::vector<std::vector<PrecisionT>> jac(
                observables.size(), std::vector<PrecisionT>(num_params, 0));
            std::vector<PrecisionT> vjp_res(num_params);
            v.vectorJacobianProduct(vjp_res, jac, dy, sv.getData(),
                                    sv.getLength(), observables, operations,
                                    trainableParams);
            return py::make_tuple(py::array_t<ParamT>(py::cast(jac)),
                                  py::array_t<ParamT>(py::cast(vjp_res)));
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
              &IndicesUtil::generateBitPatterns),
          "Get statevector indices for gate application");
    m.def("getIndicesAfterExclusion",
          py::overload_cast<const vector<size_t> &, size_t>(
              &IndicesUtil::getIndicesAfterExclusion),
          "Get statevector indices for gate application");

    /* Add EXPORTED_KERNELS */
    std::vector<std::pair<std::string, std::string>> exported_kernel_ops;

    std::set<GateOperations> gates_to_pyexport(
        std::begin(Constant::gates_to_pyexport),
        std::end(Constant::gates_to_pyexport));
    for (auto kernel : Constant::kernels_to_pyexport) {
        auto kernel_name =
            std::string(lookup(Constant::available_kernels, kernel));
        auto implemeted_gates = implementedGatesForKernel<float>(kernel);
        for (auto gate_op : implemeted_gates) {
            if (gates_to_pyexport.count(gate_op) != 0) {
                auto gate_name =
                    std::string(lookup(Constant::gate_names, gate_op));
                exported_kernel_ops.emplace_back(kernel_name, gate_name);
            }
        }
    }

    m.attr("EXPORTED_KERNEL_OPS") = py::cast(exported_kernel_ops);

    /* Add DEFAULT_KERNEL_FOR_OPS */
    std::map<std::string, std::string> default_kernel_ops_map;
    for (const auto &[gate_op, name] : Constant::gate_names) {
        auto kernel = lookup(Constant::default_kernel_for_ops, gate_op);
        auto kernel_name =
            std::string(lookup(Constant::available_kernels, kernel));
        default_kernel_ops_map.emplace(std::string(name), kernel_name);
    }
    m.attr("DEFAULT_KERNEL_FOR_OPS") = py::cast(default_kernel_ops_map);

    lightning_class_bindings<float, float>(m);
    lightning_class_bindings<double, double>(m);
}

#ifndef NDEBUG // if debug

/// @cond DEV
template <typename PrecisionT, typename ParamT, KernelType kernel, size_t idx>
constexpr void testBinderGateOpPairsForKernelIter() {
    if constexpr (idx < Constant::gates_to_pyexport.size()) {
        constexpr auto op_pairs =
            AllBinderGateOpPairs<PrecisionT, ParamT, kernel>::value;
        constexpr auto gate_op = Constant::gates_to_pyexport[idx];
        static_assert(array_has_elt(Util::first_elts_of(op_pairs), gate_op) ||
                          gate_op == GateOperations::Matrix,
                      "AllBinderGateOpPairs should have elementes for all gate "
                      "operations to pyexport.");
        testBinderGateOpPairsForKernelIter<PrecisionT, ParamT, kernel,
                                           idx + 1>();
    }
}
/// @endcond
/**
 * @brief Test whether BinderGateOpPairs are defined for all gates to export Python
 */
template <typename PrecisionT, typename ParamT, KernelType kernel>
constexpr void testBinderGateOpPairsForKernel() {
    testBinderGateOpPairsForKernelIter<PrecisionT, ParamT, kernel, 0>();
}
/// @cond DEV
template <typename PrecisionT, typename ParamT, size_t idx>
constexpr void testBinderGateOpPairsIter() {
    if constexpr (idx < Constant::kernels_to_pyexport.size()) {
        testBinderGateOpPairsForKernel<PrecisionT, ParamT,
                                       Constant::kernels_to_pyexport[idx]>();
        testBinderGateOpPairsIter<PrecisionT, ParamT, idx + 1>();
    }
}
/// @endcond

/**
 * @brief Test for all kernels to export Python
 */
template <typename PrecisionT, typename ParamT>
constexpr bool testBinderGateOpPairs() {
    testBinderGateOpPairsIter<PrecisionT, ParamT, 0>();
    return true;
}

static_assert(
    testBinderGateOpPairs<float, float>(),
    "AllBinderGateOpPairs should be well defined for all kernels to pyexport.");

static_assert(
    testBinderGateOpPairs<double, double>(),
    "AllBinderGateOpPairs should be well defined for all kernels to pyexport.");

#endif
