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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

} // namespace
/// @endcond

namespace Pennylane::LightningTensor {

/**
 * @brief Represents a cache for gate data to be accessible on the device.
 *
 * @tparam PrecisionT Floating point precision.
 */
template <class PrecisionT> class TNCudaGateCache {
  public:
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));
    using gate_id = std::pair<std::string, PrecisionT>;
    using graph_id = std::pair<std::size_t, bool>;
    TNCudaGateCache() = delete;
    TNCudaGateCache(const TNCudaGateCache &other) = delete;
    TNCudaGateCache(TNCudaGateCache &&other) = delete;
    TNCudaGateCache(bool populate, int device_id = 0,
                    cudaStream_t stream_id = 0)
        : device_tag_(device_id, stream_id), total_alloc_bytes_{0} {
        if (populate) {
            defaultPopulateCache();
        }
    }
    TNCudaGateCache(bool populate, const DevTag<int> &device_tag)
        : device_tag_{device_tag}, total_alloc_bytes_{0} {
        if (populate) {
            defaultPopulateCache();
        }
    }
    virtual ~TNCudaGateCache(){};

    /**
     * @brief Add a default gate-set to the given cache. Assumes
     * initializer-list evaluated gates for "PauliX", "PauliY", "PauliZ",
     * "Hadamard", "S", "T", "SWAP", with "CNOT" and "CZ" represented as their
     * single-qubit values.
     *
     */
    void defaultPopulateCache() {
        for (const auto &[gate_name, gate_func] : nonparametric_gates_) {
            auto gate = gate_func();
            add_gate(gate_name, 0.0, gate);
        }
    }

    /**
     * @brief Check for the existence of a given gate.
     *
     * @param gate_id std::pair of gate_name and given parameter value.
     * @return true Gate exists in cache.
     * @return false Gate does not exist in cache.
     */
    bool gateExists(const gate_id &gate) {
        return ((host_gates_.find(gate) != host_gates_.end()) &&
                (device_gates_.find(gate)) != device_gates_.end());
    }
    /**
     * @brief Check for the existence of a given gate.
     *
     * @param gate_name String of gate name.
     * @param gate_param Gate parameter value. `0.0` if non-parametric gate.
     * @return true Gate exists in cache.
     * @return false Gate does not exist in cache.
     */
    bool gateExists(const std::string &gate_name, PrecisionT gate_param) {
        return (host_gates_.find(std::make_pair(gate_name, gate_param)) !=
                host_gates_.end()) &&
               (device_gates_.find(std::make_pair(gate_name, gate_param)) !=
                device_gates_.end());
    }

    /**
     * @brief Add gate numerical value to the cache, indexed by the gate name
     * and parameter value.
     *
     * @param gate_name String representing the name of the given gate.
     * @param gate_param Gate parameter value. `0.0` if non-parametric gate.
     * @param host_data Vector of the gate values in row-major order.
     */
    void add_gate(const std::string &gate_name, PrecisionT gate_param,
                  std::vector<CFP_t> host_data) {
        const auto gate_key = std::make_pair(gate_name, gate_param);
        host_gates_[gate_key] = std::move(host_data);
        auto &gate = host_gates_[gate_key];

        std::size_t rank = Pennylane::Util::log2(gate.size());
        auto modes = std::vector<std::size_t>(rank, 0);
        auto extents = std::vector<std::size_t>(rank, 2);

        device_gates_.emplace(
            std::piecewise_construct, std::forward_as_tuple(gate_key),
            std::forward_as_tuple(rank, modes, extents, device_tag_));
        device_gates_.at(gate_key).getDataBuffer().CopyHostDataToGpu(
            gate.data(), gate.size());

        this->total_alloc_bytes_ += (sizeof(CFP_t) * gate.size());
    }

    /**
     * @brief see `void add_gate(const std::string &gate_name, PrecisionT
     gate_param, const std::vector<CFP_t> &host_data)`
     *
     * @param gate_key
     * @param host_data
     */
    void add_gate(const gate_id &gate_key, std::vector<CFP_t> host_data) {
        host_gates_[gate_key] = std::move(host_data);
        auto &gate = host_gates_[gate_key];

        size_t rank = Pennylane::Util::log2(gate.size());
        auto modes = std::vector<size_t>(rank, 0);
        auto extents = std::vector<size_t>(rank, 2);

        device_gates_.emplace(
            std::piecewise_construct, std::forward_as_tuple(gate_key),
            std::forward_as_tuple(rank, modes, extents, device_tag_));
        device_gates_.at(gate_key).getDataBuffer().CopyHostDataToGpu(
            gate.data(), gate.size());

        total_alloc_bytes_ += (sizeof(CFP_t) * gate.size());
    }

    void add_gate_id(const graph_id &graph_key, const gate_id &gate_key) {
        graph_gate_map_[graph_key] = gate_key;
    }

    /**
     * @brief Returns a pointer to the GPU device memory where the gate is
     * stored.
     *
     * @param gate_name String representing the name of the given gate.
     * @param gate_param Gate parameter value. `0.0` if non-parametric gate.
     * @return const CFP_t* Pointer to gate values on device.
     */
    CFP_t *get_gate_device_ptr(const std::string &gate_name,
                               PrecisionT gate_param) {
        return device_gates_.at(std::make_pair(gate_name, gate_param))
            .getDataBuffer()
            .getData();
    }

    CFP_t *get_gate_device_ptr(const gate_id &gate_key) {
        return device_gates_.at(gate_key).getDataBuffer().getData();
    }

    auto get_gate_host(const std::string &gate_name, PrecisionT gate_param) {
        return host_gates_.at(std::make_pair(gate_name, gate_param));
    }
    
    auto get_gate_host(const gate_id &gate_key) {
        return host_gates_.at(gate_key);
    }

  private:
    const DevTag<int> device_tag_;
    std::size_t total_alloc_bytes_;

    struct gate_id_hash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2> &pair) const {
            return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
        }
    };

    using ParamGateFunc = std::function<std::vector<Precision>(const std::vector<Precision> &)>;
    using NonParamGateFunc = std::function<std::vector<Precision>()>;
    using ParamGateFuncMap = std::unordered_map<std::string, ParamGateFunc>;
    using NonParamGateFuncMap = std::unordered_map<std::string, NonParamGateFunc>;

    NonParamGateFuncMap nonparametric_gates_{
        {"Identity", cuGates::getIdentity<CFP_t>},
        {"PauliX", cuGates::getPauliX<CFP_t>},
        {"PauliY", cuGates::getPauliY<CFP_t>},
        {"PauliZ", cuGates::getPauliZ<CFP_t>},
        {"Hadamard", cuGates::getHadamard<CFP_t>},
        {"S", cuGates::getS<CFP_t>},
        {"T", cuGates::getT<CFP_t>},
        {"SWAP", cuGates::getSWAP<CFP_t>},
        {"CNOT", cuGates::getCNOT<CFP_t>},
        {"Toffoli", cuGates::getToffoli<CFP_t>},
        {"CY", cuGates::getCY<CFP_t>},
        {"CZ", cuGates::getCZ<CFP_t>},
        {"CSWAP", cuGates::getCSWAP<CFP_t>}
    };  

    ParamGateFuncMap parametric_gates_{
        {"PhaseShift", cuGates::getPhaseShift<CFP_t>},
        {"RX", cuGates::getRX<CFP_t>},
        {"RY", cuGates::getRY<CFP_t>},
        {"RZ", cuGates::getRZ<CFP_t>},
        {"Rot", cuGates::getRot<CFP_t>},
        {"CRX", cuGates::getCRX<CFP_t>},
        {"CRY", cuGates::getCRY<CFP_t>},
        {"CRZ", cuGates::getCRZ<CFP_t>},
        {"CRot", cuGates::getCRot<CFP_t>},
        {"ControlledPhaseShift", cuGates::getControlledPhaseShift<CFP_t>},
        {"SingleExcitation", cuGates::getSingleExcitation<CFP_t>},
        {"SingleExcitationMinus", cuGates::getSingleExcitationMinus<CFP_t>},
        {"SingleExcitationPlus", cuGates::getSingleExcitationPlus<CFP_t>},
        {"DoubleExcitation", cuGates::getDoubleExcitation<CFP_t>},
        {"DoubleExcitationMinus", cuGates::getDoubleExcitationMinus<CFP_t>},
        {"DoubleExcitationPlus", cuGates::getDoubleExcitationPlus<CFP_t>}
    };

    std::unordered_map<graph_id, gate_id, gate_id_hash> graph_gate_map_;

    std::unordered_map<gate_id, TensorCuda<PrecisionT>, gate_id_hash>
        device_gates_;
    std::unordered_map<gate_id, std::vector<CFP_t>, gate_id_hash> host_gates_;
};
} // namespace Pennylane::LightningTensor
