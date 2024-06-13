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
template <class StateVectorT>
void registerMatrix(
    StateVectorT &st,
    const py::array_t<std::complex<typename StateVectorT::PrecisionT>,
                      py::array::c_style | py::array::forcecast> &matrix,
    const std::vector<std::size_t> &wires, bool inverse = false) {
    using ComplexT = typename StateVectorT::ComplexT;
    st.applyMatrix(static_cast<const ComplexT *>(matrix.request().ptr), wires,
                   inverse);
}

/**
 * @brief Register StateVector class to pybind.
 *
 * @tparam StateVectorT Statevector type to register
 * @tparam Pyclass Pybind11's class object type
 *
 * @param pyclass Pybind11's class object to bind statevector
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
        auto func = [gate_name = gate_name](
                        StateVectorT &sv, const std::vector<std::size_t> &wires,
                        bool inverse, const std::vector<ParamT> &params) {
            sv.applyOperation(gate_name, wires, inverse, params);
        };
        pyclass.def(gate_name.c_str(), func, doc.c_str());
    });
}

// TODO: Unify registerTensor and registerGatesForStateVector
/**
 * @brief Register matrix.
 */
template <class TensorNetT>
void registerTensor(
    TensorNetT &tensor_network,
    const py::array_t<std::complex<typename TensorNetT::PrecisionT>,
                      py::array::c_style | py::array::forcecast> &matrix,
    const std::vector<std::size_t> &wires, bool inverse = false) {
    using ComplexT = typename TensorNetT::ComplexT;
    const auto m_buffer = matrix.request();
    std::vector<ComplexT> conv_matrix;
    if (m_buffer.size) {
        const auto m_ptr = static_cast<const ComplexT *>(m_buffer.ptr);
        conv_matrix = std::vector<ComplexT>{m_ptr, m_ptr + m_buffer.size};
    }
    tensor_network.applyOperation("applyMatrix", wires, inverse, {},
                                  conv_matrix);
}

// TODO: Unify registerGatesForTensorNet and registerMatrix
/**
 * @brief Register TensorNet class to pybind.
 *
 * @tparam TensorNetT Tensor network type to register
 * @tparam Pyclass Pybind11's class object type
 *
 * @param pyclass Pybind11's class object to bind tensor network
 */
template <class TensorNetT, class PyClass>
void registerGatesForTensorNet(PyClass &pyclass) {
    using PrecisionT = typename TensorNetT::PrecisionT; // TensorNet's precision
    using ParamT = PrecisionT; // Parameter's data precision

    using Pennylane::Gates::GateOperation;
    using Pennylane::Util::for_each_enum;
    namespace Constant = Pennylane::Gates::Constant;

    pyclass.def("applyMatrix", &registerTensor<TensorNetT>,
                "Apply a given matrix to wires.");

    for_each_enum<GateOperation>([&pyclass](GateOperation gate_op) {
        using Pennylane::Util::lookup;
        const auto gate_name =
            std::string(lookup(Constant::gate_names, gate_op));
        const std::string doc = "Apply the " + gate_name + " gate.";
        auto func = [gate_name = gate_name](
                        TensorNetT &tensor_network,
                        const std::vector<std::size_t> &wires, bool inverse,
                        const std::vector<ParamT> &params) {
            tensor_network.applyOperation(gate_name, wires, inverse, params);
        };
        pyclass.def(gate_name.c_str(), func, doc.c_str());
    });
}
} // namespace Pennylane::Bindings
