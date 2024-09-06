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
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataBuffer.hpp"
#include "DevTag.hpp"
#include "TensorCuda.hpp"
#include "cuGates_host.hpp"
#include "cuda_helpers.hpp"

// Golden Ratio constant used for better hash. For more
// details, please check
// [here](https://softwareengineering.stackexchange.com/a/402543).
#define GOLDEN_RATIO 0x9e3779b9U

/// @cond DEV
namespace {
namespace cuUtil = Pennylane::LightningGPU::Util;
using namespace Pennylane::LightningGPU;
using namespace cuUtil;
using namespace Pennylane::LightningTensor::TNCuda;
} // namespace
/// @endcond

namespace Pennylane::LightningTensor::TNCuda::MPO {

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
    // OpsName(Identity), BondDimL, BondDimR
    // Note: No need to specify KeyEqual for std::unordered_map with std::tuple
    // objects as keys.
    using identity_mpo_info =
        std::tuple<const std::string, const std::size_t, const std::size_t>;
    // OpsName, Param, WireOrder, Extents, maxMPOBondDim, HashValue
    using mpo_info =
        std::tuple<const std::string, const std::vector<PrecisionT>,
                   const std::vector<std::size_t>, const std::size_t,
                   const std::size_t>;
    // MPO tensor data vector for each MPO sites
    using mpo_data = std::vector<std::shared_ptr<TensorCuda<PrecisionT>>>;
    // Identity MPO tensor data
    using identity_data = std::shared_ptr<TensorCuda<PrecisionT>>;

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
    /**
     * @brief Add MPO tensor to the cache.
     *
     * @param opsName Name of the gate operation.
     * @param param Parameters of the gate operation.
     * @param wire_order Wire order of the gate operation.
     * @param maxMPOBondDim Maximum bond dimension of the MPO tensor.
     * @param mpo_data MPO tensor data.
     * @param matrix_data Matrix data of the gate operation.
     */
    void add_MPO(const std::string &opsName,
                 const std::vector<PrecisionT> &param,
                 const std::vector<std::size_t> &wire_order,
                 const std::size_t maxMPOBondDim,
                 const std::vector<std::vector<std::size_t>> &extents,
                 const std::vector<std::vector<ComplexT>> &mpo_data,
                 const std::vector<ComplexT> &matrix_data = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t hash_val =
            matrix_data.empty() ? 0 : MatrixHasher()(matrix_data);
        const mpo_info mpo_key = std::make_tuple(opsName, param, wire_order,
                                                 maxMPOBondDim, hash_val);

        // Allocate memory on device and copy host data to device
        std::vector<std::shared_ptr<TensorCuda<PrecisionT>>> mpo_tensors;
        for (std::size_t i = 0; i < mpo_data.size(); ++i) {
            const std::size_t tensor_rank = extents[i].size();
            std::vector<std::size_t> modes(tensor_rank);
            std::iota(modes.begin(), modes.end(), 0);

            auto tensor = std::make_shared<TensorCuda<PrecisionT>>(
                tensor_rank, modes, extents[i], device_tag_, true);
            tensor->getDataBuffer().CopyHostDataToGpu(mpo_data[i].data(),
                                                      mpo_data[i].size());
            mpo_tensors.emplace_back(tensor);
        }

        mpo_cache_.emplace(std::piecewise_construct,
                           std::forward_as_tuple(mpo_key),
                           std::forward_as_tuple(mpo_tensors));
    }

    auto get_MPO_device_Ptr(const std::string &opsName,
                            const std::vector<PrecisionT> &param,
                            const std::vector<std::size_t> &wire_order,
                            const std::size_t maxMPOBondDim,
                            const std::vector<ComplexT> &matrix_data = {})
        -> std::vector<void *> {
        std::size_t hash_val =
            matrix_data.empty() ? 0 : MatrixHasher()(matrix_data);
        const mpo_info mpo_key = std::make_tuple(opsName, param, wire_order,
                                                 maxMPOBondDim, hash_val);

        PL_ABORT_IF(mpo_cache_.find(mpo_key) == mpo_cache_.end(),
                    "MPO not found in cache.");

        std::vector<void *> mpo_data_ptr;
        for (const auto &tensor : mpo_cache_.at(mpo_key)) {
            mpo_data_ptr.emplace_back(tensor->getDataBuffer().getData());
        }
        return mpo_data_ptr;
    }

    auto get_Identity_MPO_device_Ptr(const std::size_t bondDimL,
                                     const std::size_t bondDimR) -> void * {
        const identity_mpo_info identity_key =
            std::make_tuple("Identity", bondDimL, bondDimR);
        if (identity_mpo_cache_.find(identity_key) ==
            identity_mpo_cache_.end()) {
            add_Idenity_MPO_(bondDimL, bondDimR);
        }

        return static_cast<void *>(
            identity_mpo_cache_.at(identity_key)->getDataBuffer().getData());
    }

    bool
    is_gate_decomposed(const std::string &opsName,
                       const std::vector<PrecisionT> &param,
                       const std::vector<std::size_t> &wire_order,
                       const std::size_t maxMPOBondDim,
                       const std::vector<ComplexT> &matrix_data = {}) const {
        std::size_t hash_val =
            matrix_data.empty() ? 0 : MatrixHasher()(matrix_data);

        const mpo_info mpo_key = std::make_tuple(opsName, param, wire_order,
                                                 maxMPOBondDim, hash_val);

        return mpo_cache_.find(mpo_key) != mpo_cache_.end();
    }

  private:
    void add_Idenity_MPO_(const std::size_t bondDimL,
                          const std::size_t bondDimR) {
        std::lock_guard<std::mutex> lock(mutex_);
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
        for (std::size_t idx = 0; idx < length; idx += 2 * bondDimL + 1) {
            identity_tensor_host[idx] =
                cuUtil::complexToCu<ComplexT>(ComplexT{1.0, 0.0});
        }

        tensor->getDataBuffer().CopyHostDataToGpu(identity_tensor_host.data(),
                                                  identity_tensor_host.size());

        identity_mpo_cache_.emplace(std::piecewise_construct,
                                    std::forward_as_tuple(identity_key),
                                    std::forward_as_tuple(std::move(tensor)));
    }

  private:
    const DevTag<int> device_tag_{0, 0};
    static std::mutex mutex_;

    // Follow the boost::hash_combine pattern(as shown
    // [here](https://stackoverflow.com/a/2595226).
    struct mpo_hash {
        std::size_t operator()(const mpo_info &key) const {
            std::size_t seed = 0;
            seed ^= std::hash<std::string>()(std::get<0>(key)) + GOLDEN_RATIO +
                    (seed << 6) + (seed >> 2);
            for (const auto &param : std::get<1>(key)) {
                seed ^= std::hash<PrecisionT>()(param) + GOLDEN_RATIO +
                        (seed << 6) + (seed >> 2);
            }
            for (const auto &order : std::get<2>(key)) {
                seed ^= std::hash<std::size_t>()(order) + GOLDEN_RATIO +
                        (seed << 6) + (seed >> 2);
            }
            seed ^= std::hash<std::size_t>()(std::get<3>(key)) + GOLDEN_RATIO +
                    (seed << 6) + (seed >> 2);
            seed ^= std::hash<std::size_t>()(std::get<4>(key)) + GOLDEN_RATIO +
                    (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    struct identity_mpo_hash {
        std::size_t operator()(const identity_mpo_info &key) const {
            std::size_t seed = 0;
            seed ^= std::hash<std::string>()(std::get<0>(key)) + GOLDEN_RATIO +
                    (seed << 6) + (seed >> 2);
            seed ^= std::hash<std::size_t>()(std::get<1>(key)) + GOLDEN_RATIO +
                    (seed << 6) + (seed >> 2);
            seed ^= std::hash<std::size_t>()(std::get<2>(key)) + GOLDEN_RATIO +
                    (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    std::unordered_map<const mpo_info, mpo_data, mpo_hash> mpo_cache_;

    std::unordered_map<const identity_mpo_info, identity_data,
                       identity_mpo_hash>
        identity_mpo_cache_;
};
} // namespace Pennylane::LightningTensor::TNCuda::MPO
