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
 * Memory management for Gate tensor data affiliated to the tensor network
 * graph.
 */

#pragma once

#include <complex>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataBuffer.hpp"
#include "DevTag.hpp"
#include "TensorCuda.hpp"
#include "cuGates_host.hpp"

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace cuUtil;
using namespace Pennylane::LightningTensor::TNCuda;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::Gates {

/**
 * @brief Memory management for gate tensor data on device and its id in the
 * compute graph.
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

    ~TNCudaGateCache(){};

    /**
     * @brief Add gate numerical value to the cache, indexed by the id of gate
     * tensor operator in the graph and its name and parameter value are
     * recorded as well.
     *
     * @param gate_id The id of gate tensor operator in the computate graph.
     * @param gate_name String representing the name of the given gate.
     * @param gate_param Vector of parameter values. `{}` if non-parametric
     * gate.
     */
    void add_gate(const std::size_t gate_id, const std::string &gate_name,
                  [[maybe_unused]] std::vector<PrecisionT> gate_param = {}) {
        auto gate_key = std::make_pair(gate_name, gate_param);

        auto &gateMap =
            cuGates::DynamicGateDataAccess<PrecisionT>::getInstance();

        add_gate(gate_id, gate_key, gateMap.getGateData(gate_name, gate_param));
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

    void add_gate(const std::size_t gate_id, gate_key_info gate_key,
                  const std::vector<CFP_t> &gate_data_host) {
        const std::size_t rank = Pennylane::Util::log2(gate_data_host.size());
        auto modes = std::vector<std::size_t>(rank, 0);
        auto extents = std::vector<std::size_t>(rank, 2);

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
    CFP_t *get_gate_device_ptr(const std::size_t gate_id) {
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

    // device_gates_ is a map of id of gate tensor operator in the graph to the
    // gate_info and gate_info is a pair of gate_info_key, which contains both
    // gate name and parameter value, and the tensor data on device.
    std::unordered_map<std::size_t, gate_info, gate_info_hash> device_gates_;
};
} // namespace Pennylane::LightningTensor::TNCuda::Gates
