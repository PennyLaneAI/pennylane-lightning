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

#include <complex>
#include <vector>

#include "Error.hpp"

namespace Pennylane::LightningTensor::MPS {
template <class Precision, class Derived> class MPSBase {
  public:
    using ComplexT = std::complex<Precision>;
    using PrecisionT = Precision;
    using BaseType = MPSBase<Precision, Derived>;

  private:
    size_t numQubits_;
    size_t maxExtent_;
    std::vector<size_t> qubitDims_;
    std::vector<std::vector<size_t>> sitesModes_;
    std::vector<std::vector<size_t>> sitesExtents_;

  public:
    explicit MPSBase(size_t numQubits, size_t maxExtent,
                     std::vector<size_t> qubitDims)
        : numQubits_(numQubits), maxExtent_(maxExtent), qubitDims_(qubitDims) {
        // Configure extents for each sites
        for (size_t i = 0; i < numQubits_; i++) {
            std::vector<size_t> localSiteModes;
            std::vector<size_t> localSiteExtents;
            if (i == 0) {
                // Leftmost site
                localSiteModes = std::vector<size_t>({i, i + numQubits_});
                localSiteExtents =
                    std::vector<size_t>({qubitDims_[i], maxExtent_});
            } else if (i == numQubits_ - 1) {
                // Rightmost site
                localSiteModes = std::vector<size_t>({i + numQubits_, i});
                localSiteExtents =
                    std::vector<size_t>({qubitDims_[i], maxExtent_});
            } else {
                // Interior sites
                localSiteModes = std::vector<size_t>(
                    {i + numQubits_ - 1, i, i + numQubits_});
                localSiteExtents = std::vector<size_t>(
                    {maxExtent_, qubitDims_[i], maxExtent_});
            }
            sitesExtents_.push_back(localSiteExtents);
            sitesModes_.push_back(localSiteModes);
        }
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
     * @brief Get modes of ith site
     *
     * @param index i-th site
     *
     * @return std::vector<size_t> &
     */
    [[nodiscard]] auto getIthSiteModes(size_t index) -> std::vector<size_t> & {
        PL_ABORT_IF(index >= numQubits_,
                    "The site index value should be less than qubit number.")
        return sitesModes_[index];
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

    /**
     * @brief Get extents of ith site
     *
     * @param index i-th site
     *
     * @return std::vector<size_t> &
     */
    [[nodiscard]] auto getIthSiteExtents(size_t index)
        -> std::vector<size_t> & {
        PL_ABORT_IF(index >= numQubits_,
                    "The site index value should be less than qubit number.")
        return sitesExtents_[index];
    };
};
} // namespace Pennylane::LightningTensor::MPS