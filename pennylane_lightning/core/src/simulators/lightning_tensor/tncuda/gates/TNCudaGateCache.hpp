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
 * @file TNCudaGateCache.hpp
 * Memory management for Gate caches.
 */

#pragma once

#include <cmath>
#include <complex>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <iostream>

#include "DataBuffer.hpp"
#include "DevTag.hpp"
#include "TensorCuda.hpp"
#include "cuGates_host.hpp"
#include "cuda.h"
#include "cuda_helpers.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace cuUtil;
using namespace Pennylane::LightningTensor::TNCuda;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor {

/**
 * @brief Memory management for gate tensor data on device and its id in the
 * graph.
 *
 * @tparam PrecisionT Floating point precision.
 */
template <class PrecisionT> class TNCudaGateCache {
  public:
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));
    using gate_key_info = std::pair<const std::string, std::vector<PrecisionT>>;
    using gate_info = std::pair<gate_key_info, TensorCuda<PrecisionT>>;
    TNCudaGateCache() = delete;
    TNCudaGateCache(const TNCudaGateCache &other) = delete;
    TNCudaGateCache(TNCudaGateCache &&other) = delete;
    TNCudaGateCache(int device_id = 0, cudaStream_t stream_id = 0)
        : device_tag_(device_id, stream_id), total_alloc_bytes_{0} {}
    TNCudaGateCache(const DevTag<int> device_tag)
        : device_tag_{device_tag}, total_alloc_bytes_{0} {}
    virtual ~TNCudaGateCache(){};

    /**
     * @brief Add gate numerical value to the cache, indexed by the id of gate
     * tensor operator in the graph and its name and parameter value are
     * recorded as well.
     *
     * @param gate_id The id of gate tensor operator in the computate graph.
     * @param gate_name String representing the name of the given gate.
     * @param gate_param Gate parameter value. `0.0` if non-parametric gate.
     */
    void add_gate(const size_t gate_id, const std::string &gate_name,
                  std::vector<PrecisionT> gate_param) {
        auto gate_key = std::make_pair(gate_name, gate_param);

        if (nonparametric_gates_.find(gate_name) !=
            nonparametric_gates_.end()) {
            auto gate_data_host = nonparametric_gates_.at(gate_name)();
            add_gate(gate_id, gate_key, gate_data_host);
        } else if (parametric_gates_.find(gate_name) !=
                   parametric_gates_.end()) {
            auto gate_data_host = parametric_gates_.at(gate_name)({gate_param});
            add_gate(gate_id, gate_key, gate_data_host);
        } else {
            throw std::runtime_error("Unsupported gate.");
        }
    }
    /**
     * @brief Add gate numerical value to the cache, indexed by the id of gate
     * tensor operator in the graph and its name and parameter value as well as
     * the gate data on host.
     *
     * @param gate_id The id of gate tensor operator in the computate graph.
     * @param gate_key String representing the name of the given gate as well as
     * its associated parameter value.
     * @param gate_data_host Vector of complex floating point values
     * representing the gate data on host.
     */

    void add_gate(const size_t gate_id, gate_key_info gate_key,
                  const std::vector<CFP_t> &gate_data_host) {
        size_t rank = Pennylane::Util::log2(gate_data_host.size());
        auto modes = std::vector<size_t>(rank, 0);
        auto extents = std::vector<size_t>(rank, 2);

        auto &&tensor =
            TensorCuda<PrecisionT>(rank, modes, extents, device_tag_);

        device_gates_.emplace(
            std::piecewise_construct, std::forward_as_tuple(gate_id),
            std::forward_as_tuple(gate_key, std::move(tensor)));

        device_gates_.at(gate_id).second.getDataBuffer().CopyHostDataToGpu(
            gate_data_host.data(), gate_data_host.size());

        total_alloc_bytes_ += (sizeof(CFP_t) * gate_data_host.size());
    }

    /**
     * @brief Returns a pointer to the GPU device memory where the gate is
     * stored.
     *
     * @param gate_id The id of gate tensor operator in the computate graph.
     * @return const CFP_t* Pointer to gate values on device.
     */
    CFP_t *get_gate_device_ptr(const size_t gate_id) {
        return device_gates_.at(gate_id).second.getDataBuffer().getData();
    }

  private:
    const DevTag<int> device_tag_;
    std::size_t total_alloc_bytes_;

    struct gate_info_hash {
        std::size_t operator()(std::size_t key_id) const {
            return std::hash<std::size_t>()(key_id);
        }
    };

    using ParamGateFunc =
        std::function<std::vector<CFP_t>(const std::vector<PrecisionT> &)>;
    using NonParamGateFunc = std::function<std::vector<CFP_t>()>;
    using ParamGateFuncMap = std::unordered_map<std::string, ParamGateFunc>;
    using NonParamGateFuncMap =
        std::unordered_map<std::string, NonParamGateFunc>;

    NonParamGateFuncMap nonparametric_gates_{
        {"Identity",
         [&]() -> std::vector<CFP_t> { return cuGates::getIdentity<CFP_t>(); }},
        {"PauliX",
         [&]() -> std::vector<CFP_t> { return cuGates::getPauliX<CFP_t>(); }},
        {"PauliY",
         [&]() -> std::vector<CFP_t> { return cuGates::getPauliY<CFP_t>(); }},
        {"PauliZ",
         [&]() -> std::vector<CFP_t> { return cuGates::getPauliZ<CFP_t>(); }},
        {"S", [&]() -> std::vector<CFP_t> { return cuGates::getS<CFP_t>(); }},
        {"Hadamard",
         [&]() -> std::vector<CFP_t> { return cuGates::getHadamard<CFP_t>(); }},
        {"T", [&]() -> std::vector<CFP_t> { return cuGates::getT<CFP_t>(); }},
        {"SWAP",
         [&]() -> std::vector<CFP_t> { return cuGates::getSWAP<CFP_t>(); }},
        {"CNOT",
         [&]() -> std::vector<CFP_t> { return cuGates::getCNOT<CFP_t>(); }},
        {"Toffoli",
         [&]() -> std::vector<CFP_t> { return cuGates::getToffoli<CFP_t>(); }},
        {"CY", [&]() -> std::vector<CFP_t> { return cuGates::getCY<CFP_t>(); }},
        {"CZ", [&]() -> std::vector<CFP_t> { return cuGates::getCZ<CFP_t>(); }},
        {"CSWAP",
         [&]() -> std::vector<CFP_t> { return cuGates::getCSWAP<CFP_t>(); }}};

    ParamGateFuncMap parametric_gates_{
        {"PhaseShift",
         [&](auto &&params) {
             return cuGates::getPhaseShift<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"RX",
         [&](auto &&params) {
             return cuGates::getRX<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"RY",
         [&](auto &&params) {
             return cuGates::getRY<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"RZ",
         [&](auto &&params) {
             return cuGates::getRZ<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"Rot",
         [&](auto &&params) {
             return cuGates::getRot<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]),
                 std::forward<decltype(params[1])>(params[1]),
                 std::forward<decltype(params[2])>(params[2]));
         }},
        {"CRX",
         [&](auto &&params) {
             return cuGates::getCRX<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRY",
         [&](auto &&params) {
             return cuGates::getCRY<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRZ",
         [&](auto &&params) {
             return cuGates::getCRZ<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"CRot",
         [&](auto &&params) {
             return cuGates::getCRot<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]),
                 std::forward<decltype(params[1])>(params[1]),
                 std::forward<decltype(params[2])>(params[2]));
         }},
        {"ControlledPhaseShift",
         [&](auto &&params) {
             return cuGates::getControlledPhaseShift<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingXX",
         [&](auto &&params) {
             return cuGates::getIsingXX<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingYY",
         [&](auto &&params) {
             return cuGates::getIsingYY<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingZZ",
         [&](auto &&params) {
             return cuGates::getIsingZZ<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"IsingXY",
         [&](auto &&params) {
             return cuGates::getIsingXY<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitation",
         [&](auto &&params) {
             return cuGates::getSingleExcitation<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationMinus",
         [&](auto &&params) {
             return cuGates::getSingleExcitationMinus<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"SingleExcitationPlus",
         [&](auto &&params) {
             return cuGates::getSingleExcitationPlus<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitation",
         [&](auto &&params) {
             return cuGates::getDoubleExcitation<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationMinus",
         [&](auto &&params) {
             return cuGates::getDoubleExcitationMinus<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }},
        {"DoubleExcitationPlus", [&](auto &&params) {
             return cuGates::getDoubleExcitationPlus<CFP_t>(
                 std::forward<decltype(params[0])>(params[0]));
         }}};

    std::unordered_map<std::size_t, gate_info, gate_info_hash> device_gates_;
};
} // namespace Pennylane::LightningTensor
