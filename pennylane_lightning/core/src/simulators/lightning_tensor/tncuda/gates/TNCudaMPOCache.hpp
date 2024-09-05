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
 * @file TNCudaMPOCache.hpp
 * Memory management for MPO tensor data affiliated to the tensor network
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
 * @brief Memory management for MPO tensor data on device and its id in the
 * compute graph.
 *
 * @tparam PrecisionT Floating point precision.
 */
template <class PrecisionT> class TNCudaMPOCache {
  private:
    TNCudaMPOCache() = default;

  public:
    using CFP_t = decltype(cuUtil::getCudaType(PrecisionT{}));
    using ComplexT = std::complex<PrecisionT>;
    using identity_mpo_info =
        std::tuple<const std::string, const std::size_t bondDimL,
                   const std::size_t bondDimR>;
    using mpo_info =
        std::tuple<const std::string, const std::vector<PrecisionT>,
                   const std::vector<std::size_t> wires_order,
                   const std::size_t hash_value>;
    using mpo_data = std::vector<std::share_ptr<TensorCuda<PrecisionT>>>;
    using identity_data = std::share_ptr<TensorCuda<PrecisionT>>;

    TNCudaMPOCache(TNCudaMPOCache &&) = delete;
    TNCudaMPOCache(const TNCudaMPOCache &) = delete;
    TNCudaMPOCache &operator=(const TNCudaMPOCache &) = delete;

    ~TNCudaMPOCache() = default;

  public:
    static TNCudaMPOCache &getInstance() {
        static TNCudaMPOCache instance;
        return instance;
    }

    // Add MPO tensor to the cache. This function is called by the python layer
    // to add the MPO tensor to the cache.
    void add_MPO(const std::string &opsName,
                 const std::vector<PrecisionT> &param,
                 const std::vector<std::size_t> &wire_order,
                 const std::vector<std::size_t> &extents,
                 const std::vector<std::vector<ComplexT>> &mpo_data,
                 const std::vector<ComplexT> &matrix_data = {}) {
        // auto tensor = std::make_shared<TensorCuda<PrecisionT>>(
        //     modes.size(), modes, extents, dev_tag, true);
        // tensor->setData(data);
        // mpoCache_[id] = tensor;
    }

    auto get_MPO_device_Ptr(const std::string &id) const
        -> std::shared_ptr<TensorCuda<PrecisionT>> {
        if (mpoCache_.find(id) == mpoCache_.end()) {
            throw TNCudaError("MPO tensor not found in cache.");
        }
        return mpoCache_.at(id);
    }

    auto get_Identity_MPO_device_Ptr(const std::size_t bondDimL,
                                     const std::size_t bondDimR) const
        -> std::shared_ptr<TensorCuda<PrecisionT>> {
        if (identity_mpo_cache_.find(id) == identity_mpo_cache_.end()) {
            add_Idenity_MPO_(bondDimL, bondDimR);
        }
        const identity_mpo_info identity_key =
            std::make_tuple("Identity", bondDimL, bondDimR);
        return identity_mpo_cache_.at(identity_key);
    }

    bool hasMPO(const std::string &opsName,
                const std::vector<PrecisionT> &param,
                const std::vector<std::size_t> &wire_order,
                const std::vector<std::size_t> &extents,
                const std::vector<ComplexT> &matrix_data = {}) const {
        std::size_t hash_value = 0;
        if (!matrix_data.empty()) {
            hash_value = std::hash<std::string>()(matrix_data);
        }

        const mpo_info mpo_key =
            std::make_tuple(opsName, param, wire_order, hash_value);

        return mpoCache_.find(id) != mpoCache_.end();
    }

  private:
    void add_Idenity_MPO_(const std::size_t bondDimL,
                          const std::size_t bondDimR) {
        const identity_mpo_info identity_key =
            std::make_tuple("Identity", bondDimL, bondDimR);

        const std::size_t tensor_rank = 4;
        const std::vector<std::size_t> extents{bondDimL, 2, bondDimR, 2};
        const std::vector<std::size_t> modes{0, 1, 2, 3};

        auto tensor = std::make_shared<TensorCuda<PrecisionT>>(
            tensor_rank, modes, extents, device_tag_, true);

        tensor->getDataBuffer().zeroInit();

        std::size_t length = tensor->getDataBuffer().getLength();
        std::vector<CFP_t> identity_tensor_host(length, CFP_t{0.0, 0.0});
        for (std::size_t idx = 0; idx < length;
             idx += 2 * bondDims_[i - 1] + 1) {
            identity_tensor_host[idx] =
                cuUtil::complexToCu<ComplexT>(ComplexT{1.0, 0.0});
        }

        tensor->getDataBuffer().CopyHostDataToGpu(identity_tensor_host.data(),
                                                  identity_tensor_host.size());

        identity_mpo_cache_.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(identity_key),
                                    tensor);
    }

  private:
    const DevTag<int> device_tag_;

    struct mpo_hash {
        std::size_t operator()(const mpo_info &key) const {
            return std::hash<std::size_t>()(key);
        }
    };

    struct mpo_equal {
        bool operator()(const mpo_info &first, const mpo_info &second) const {
            return lhs == rhs;
        }
    };

    struct identity_mpo_hash {
        std::size_t operator()(const identity_mpo_info &key) const {
            return std::hash<std::size_t>()(key_id);
        }
    };

    struct identity_mpo_equal {
        bool operator()(const identity_mpo_info &first,
                        const identity_mpo_info &second) const {
            return lhs == rhs;
        }
    };

    std::unordered_map<const mpo_info, mpo_data, mpo_hash, mpo_equal> mpoCache_;

    std::unordered_map<const identity_mpo_info, identity_data,
                       identity_mpo_hash, identity_mpo_equal>
        identity_mpo_cache_;
};
} // namespace Pennylane::LightningTensor::TNCuda::Gates
