// Copyright 2024 Xanadu Quantum Technologies Inc.

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
#include <memory>
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
#include "Util.hpp" // for_each_enum

namespace py = pybind11;

namespace Pennylane::Bindings {
/**
 * @brief Register matrix.
 */
template <class StateTensorT>
void registerTensor(
    StateTensorT &state_tensor,
    const py::array_t<std::complex<typename StateTensorT::PrecisionT>,
                      py::array::c_style | py::array::forcecast> &matrix,
    const std::vector<std::size_t> &wires, bool inverse = false) {
    using ComplexT = typename StateTensorT::ComplexT;
    const auto m_buffer = matrix.request();
    std::vector<ComplexT> conv_matrix;
    if (m_buffer.size) {
        const auto m_ptr = static_cast<const ComplexT *>(m_buffer.ptr);
        conv_matrix = std::vector<ComplexT>{m_ptr, m_ptr + m_buffer.size};
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

    pyclass.def("applyMatrix", &registerTensor<StateTensorT>,
                "Apply a given matrix to wires.");

    for_each_enum<GateOperation>([&pyclass](GateOperation gate_op) {
        using Pennylane::Util::lookup;
        const auto gate_name =
            std::string(lookup(Constant::gate_names, gate_op));
        const std::string doc = "Apply the " + gate_name + " gate.";
        auto func = [gate_name = gate_name](
                        StateTensorT &state_tensor,
                        const std::vector<std::size_t> &wires, bool inverse,
                        const std::vector<ParamT> &params) {
            state_tensor.applyOperation(gate_name, wires, inverse, params);
        };
        pyclass.def(gate_name.c_str(), func, doc.c_str());
    });
}
} // namespace Pennylane::Bindings