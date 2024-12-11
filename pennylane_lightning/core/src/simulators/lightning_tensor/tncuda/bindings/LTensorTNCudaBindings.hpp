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

/**
 * @file LTensorTNCudaBindings.hpp
 * Defines LightningTensor-specific operations to export to Python, other
 * utility functions interfacing with Pybind11 and support to agnostic bindings.
 */

#pragma once
#include <vector>

#include "cuda.h"

#include "BindingsBase.hpp"
#include "BindingsCudaUtils.hpp"
#include "DevTag.hpp"
#include "DevicePool.hpp"
#include "Error.hpp"
#include "MPSTNCuda.hpp"
#include "TypeList.hpp"
#include "Util.hpp"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
using namespace Pennylane;
using namespace Pennylane::Bindings;
using namespace Pennylane::LightningGPU::Util;
using Pennylane::LightningTensor::TNCuda::MPSTNCuda;
} // namespace
/// @endcond

namespace py = pybind11;

namespace Pennylane::LightningTensor::TNCuda {
using TensorNetBackends =
    Pennylane::Util::TypeList<MPSTNCuda<float>, MPSTNCuda<double>, void>;

/**
 * @brief Register controlled matrix kernel.
 */
template <class TensorNet>
void applyControlledMatrix(
    TensorNet &tensor_network,
    const py::array_t<std::complex<typename TensorNet::PrecisionT>,
                      py::array::c_style | py::array::forcecast> &matrix,
    const std::vector<std::size_t> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::size_t> &target_wires, bool inverse = false) {
    using ComplexT = typename TensorNet::ComplexT;
    const auto m_buffer = matrix.request();
    std::vector<ComplexT> conv_matrix;
    if (m_buffer.size) {
        const auto m_ptr = static_cast<const ComplexT *>(m_buffer.ptr);
        conv_matrix = std::vector<ComplexT>{m_ptr, m_ptr + m_buffer.size};
    }

    tensor_network.applyControlledOperation(
        "applexControlledMatrix", controlled_wires, controlled_values,
        target_wires, inverse, {}, conv_matrix);
}

template <class TensorNet, class PyClass>
void registerControlledGate(PyClass &pyclass) {
    using PrecisionT = typename TensorNet::PrecisionT; // TensorNet's precision
    using ParamT = PrecisionT; // Parameter's data precision

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
                            TensorNet &tensor_network,
                            const std::vector<std::size_t> &controlled_wires,
                            const std::vector<bool> &controlled_values,
                            const std::vector<std::size_t> &target_wires,
                            bool inverse, const std::vector<ParamT> &params) {
                tensor_network.applyControlledOperation(
                    gate_name, controlled_wires, controlled_values,
                    target_wires, inverse, params);
            };
            pyclass.def(gate_name.c_str(), func, doc.c_str());
        });
}

/**
 * @brief Get a gate kernel map for a tensor network.
 */
template <class TensorNet, class PyClass>
void registerBackendClassSpecificBindings(PyClass &pyclass) {
    registerGatesForTensorNet<TensorNet>(pyclass);
    registerControlledGate<TensorNet, PyClass>(pyclass);
    using PrecisionT = typename TensorNet::PrecisionT; // TensorNet's precision
    using ParamT = PrecisionT; // Parameter's data precision

    using np_arr_c = py::array_t<std::complex<ParamT>,
                                 py::array::c_style | py::array::forcecast>;

    pyclass
        .def(py::init<const std::size_t,
                      const std::size_t>()) // num_qubits, max_bond_dim
        .def(py::init<const std::size_t, const std::size_t,
                      DevTag<int>>()) // num_qubits, max_bond_dim, dev-tag
        .def(
            "getState",
            [](TensorNet &tensor_network, np_arr_c &state) {
                py::buffer_info numpyArrayInfo = state.request();
                auto *data_ptr =
                    static_cast<std::complex<PrecisionT> *>(numpyArrayInfo.ptr);

                tensor_network.getData(data_ptr, state.size());
            },
            "Copy StateVector data into a Numpy array.")
        .def("applyControlledMatrix", &applyControlledMatrix<TensorNet>,
             "Apply controlled operation")
        .def(
            "updateMPSSitesData",
            [](TensorNet &tensor_network, std::vector<np_arr_c> &tensors) {
                for (std::size_t idx = 0; idx < tensors.size(); idx++) {
                    py::buffer_info numpyArrayInfo = tensors[idx].request();
                    auto *data_ptr = static_cast<std::complex<PrecisionT> *>(
                        numpyArrayInfo.ptr);
                    tensor_network.updateMPSSiteData(idx, data_ptr,
                                                     tensors[idx].size());
                }
            },
            "Pass MPS site data to the C++ backend.")
        .def(
            "setBasisState",
            [](TensorNet &tensor_network,
               std::vector<std::size_t> &basisState) {
                tensor_network.setBasisState(basisState);
            },
            "Create Basis State on GPU.")
        .def(
            "applyMPOOperation",
            [](TensorNet &tensor_network, std::vector<np_arr_c> &tensors,
               std::vector<std::size_t> &wires, const std::size_t MPOBondDims) {
                using ComplexT = typename TensorNet::ComplexT;
                std::vector<std::vector<ComplexT>> conv_tensors;
                for (const auto &tensor : tensors) {
                    py::buffer_info numpyArrayInfo = tensor.request();
                    auto *m_ptr = static_cast<ComplexT *>(numpyArrayInfo.ptr);
                    conv_tensors.push_back(
                        std::vector<ComplexT>{m_ptr, m_ptr + tensor.size()});
                }
                tensor_network.applyMPOOperation(conv_tensors, wires,
                                                 MPOBondDims);
            },
            "Apply MPO to the tensor network graph.")
        .def(
            "appendMPSFinalState",
            [](TensorNet &tensor_network, double cutoff,
               std::string cutoff_mode) {
                tensor_network.append_mps_final_state(cutoff, cutoff_mode);
            },
            "Get the final state.")
        .def("reset", &TensorNet::reset, "Reset the statevector.");
}

/**
 * @brief Provide backend information.
 */
auto getBackendInfo() -> py::dict {
    using namespace py::literals;

    return py::dict("NAME"_a = "lightning.tensor");
}

/**
 * @brief Register bindings for backend-specific info.
 *
 * @param m Pybind11 module.
 */
void registerBackendSpecificInfo(py::module_ &m) {
    m.def("backend_info", &getBackendInfo, "Backend-specific information.");
    registerCudaUtils(m);
}

} // namespace Pennylane::LightningTensor::TNCuda
