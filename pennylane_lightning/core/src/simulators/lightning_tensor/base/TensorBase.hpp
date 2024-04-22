// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

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
 */

#pragma once

#include <vector>

#include "Error.hpp"

namespace Pennylane::LightningTensor {

/**
 * @brief CRTP-enabled base class for tensor.
 *
 * @tparam Precision Floating point precision.
 * @tparam Derived Derived class to instantiate using CRTP.
 */
template <class PrecisionT, class Derived> class TensorBase {
  private:
    size_t rank_;                 // A rank N tensor has N modes
    size_t length_;               // Number of elements
    std::vector<size_t> modes_;   // modes for contraction identify
    std::vector<size_t> extents_; // Number of elements in each mode

  public:
    TensorBase(size_t rank, std::vector<size_t> &modes,
               std::vector<size_t> &extents)
        : rank_(rank), modes_(modes), extents_(extents) {
        PL_ABORT_IF(rank_ != extents_.size(),
                    "Please check if rank or extents are set correctly.");
        length_ = 1;
        for (auto extent : extents) {
            length_ *= extent;
        }
    };

    virtual ~TensorBase() {}

    /**
     * @brief Return the rank of a tensor object.
     *
     * @return size_t Rank of a tensor object.
     */
    [[nodiscard]] auto getRank() -> size_t { return rank_; }

    /**
     * @brief Return the extents of a tensor object.
     *
     * @return std::vector<size_t> Extents of a tensor object.
     */
    [[nodiscard]] auto getExtents() -> std::vector<size_t> { return extents_; }

    /**
     * @brief Return the modes of a tensor object.
     *
     * @return std::vector<size_t> Modes of a tensor object.
     */
    [[nodiscard]] auto getModes() const -> std::vector<size_t> {
        return modes_;
    };

    /**
     * @brief Return the number of elements of a tensor object.
     *
     * @return std::vector<size_t> Number of elements of a tensor object.
     */
    [[nodiscard]] size_t getLength() const { return length_; }

    /**
     * @brief Return a pointer to the tensor data.
     *
     * @return Complex pointer to the tensor data.
     */
    [[nodiscard]] auto getData() {
        return static_cast<Derived *>(this)->getData();
    }
};
} // namespace Pennylane::LightningTensor