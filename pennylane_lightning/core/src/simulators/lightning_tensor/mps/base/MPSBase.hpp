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
 * @file MPSBase.hpp
 * Base class for all MPS backend.
 */

#pragma once

#include <vector>

#include "Error.hpp"

namespace Pennylane::LightningTensor::MPS {
template <class Precision, class Derived> class MPSBase {
  private:
    size_t numQubits_;
    size_t maxExtent_;
    std::vector<size_t> qubitDims_;
    std::vector<std::vector<size_t>> sitesModes_;
    std::vector<std::vector<size_t>> sitesExtents_;

  public:
    MPSBase() = delete;

    explicit MPSBase(const size_t numQubits, const size_t maxExtent,
                     const std::vector<size_t> &qubitDims)
        : numQubits_(numQubits), maxExtent_(maxExtent), qubitDims_(qubitDims) {
        initHelper_();
    }

    virtual ~MPSBase() = default;

    /**
     * @brief Get dimension of each site
     *
     * @return std::vector<size_t> &
     */
    [[nodiscard]] auto getQubitDims() -> std::vector<size_t> & {
        return qubitDims_;
    };

    /**
     * @brief Get the max bond dimension.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getMaxExtent() const -> size_t { return maxExtent_; };

    /**
     * @brief Get the number of qubits represented by the MPS.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getNumQubits() const -> size_t { return numQubits_; };

    /**
     * @brief Get modes of each sites
     *
     * @return const std::vector<std::vector<size_t>> &
     */
    [[nodiscard]] auto getSitesModes() const
        -> const std::vector<std::vector<size_t>> & {
        return sitesModes_;
    };

    /**
     * @brief Get extents of each sites
     *
     * @return const std::vector<std::vector<size_t>> &
     */
    [[nodiscard]] auto getSitesExtents() const
        -> const std::vector<std::vector<size_t>> & {
        return sitesExtents_;
    };

  private:
    void initHelper_() {
        // Configure extents for each sites
        for (size_t i = 0; i < numQubits_; i++) {
            std::vector<size_t> localSiteModes;
            std::vector<size_t> localSiteExtents;
            if (i == 0) {
                // Leftmost site (state mode, shared mode)
                localSiteModes = std::vector<size_t>({i, i + numQubits_});
                localSiteExtents =
                    std::vector<size_t>({qubitDims_[i], maxExtent_});
            } else if (i == numQubits_ - 1) {
                // Rightmost site (shared mode, state mode)
                localSiteModes = std::vector<size_t>({i + numQubits_ - 1, i});
                localSiteExtents =
                    std::vector<size_t>({maxExtent_, qubitDims_[i]});
            } else {
                // Interior sites (state mode, state mode, shared mode)
                localSiteModes = std::vector<size_t>(
                    {i + numQubits_ - 1, i, i + numQubits_});
                localSiteExtents = std::vector<size_t>(
                    {maxExtent_, qubitDims_[i], maxExtent_});
            }
            sitesExtents_.push_back(localSiteExtents);
            sitesModes_.push_back(localSiteModes);
        }
    }
};
} // namespace Pennylane::LightningTensor::MPS
