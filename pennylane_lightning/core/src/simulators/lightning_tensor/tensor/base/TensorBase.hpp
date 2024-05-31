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
 * @file TensorBase.hpp
 * Tensor base class for all backends.
 */

#pragma once

#include <functional>
#include <vector>

#include "Error.hpp"

namespace Pennylane::LightningTensor {

/**
 * @brief CRTP-enabled base class for tensor.
 *
 * @tparam PrecisionT Floating point precision.
 * @tparam Derived Derived class to instantiate using CRTP.
 */
template <class PrecisionT, class Derived> class TensorBase {
  private:
    const std::size_t rank_;                 // A rank N tensor has N modes
    std::size_t length_;                     // Number of elements
    const std::vector<std::size_t> modes_;   // modes for contraction identify
    const std::vector<std::size_t> extents_; // Number of elements in each mode

  public:
    explicit TensorBase(std::size_t rank, const std::vector<std::size_t> &modes,
                        const std::vector<std::size_t> &extents)
        : rank_(rank), modes_(modes), extents_(extents) {
        PL_ABORT_IF_NOT(rank_ == extents_.size(),
                        "Please check if rank or extents are set correctly.");
        length_ = std::accumulate(extents.begin(), extents.end(),
                                  std::size_t{1}, std::multiplies<>());
    }
    /**
     * @brief Construct a tensor object with given extents.
     *
     * @param extents Extents of a tensor object.
     */
    explicit TensorBase(const std::vector<std::size_t> &extents)
        : rank_(extents.size()),
          modes_(std::move(std::vector(rank_, std::size_t{0}))),
          extents_(std::move(extents)) {
        length_ = std::accumulate(extents_.begin(), extents_.end(),
                                  std::size_t{1}, std::multiplies<>());
    }

    ~TensorBase() {}

    /**
     * @brief Return the rank of a tensor object.
     *
     * @return std::size_t Rank of a tensor object.
     */
    [[nodiscard]] auto getRank() const -> std::size_t { return rank_; }

    /**
     * @brief Return the extents of a tensor object.
     *
     * @return std::vector<std::size_t> Extents of a tensor object.
     */
    [[nodiscard]] auto getExtents() const -> const std::vector<std::size_t> & {
        return extents_;
    }

    /**
     * @brief Return the modes of a tensor object.
     *
     * @return std::vector<std::size_t> Modes of a tensor object.
     */
    [[nodiscard]] auto getModes() const -> const std::vector<std::size_t> & {
        return modes_;
    };

    /**
     * @brief Return the number of elements of a tensor object.
     *
     * @return std::size_t Number of elements of a tensor object.
     */
    [[nodiscard]] std::size_t getLength() const { return length_; }
};
} // namespace Pennylane::LightningTensor
