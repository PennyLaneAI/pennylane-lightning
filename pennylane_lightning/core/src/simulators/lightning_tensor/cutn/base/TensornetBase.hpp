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
 * @file TensornetBase.hpp
 * Base class for all cutensornet backend.
 */

#pragma once

#include <vector>

#include "Error.hpp"

namespace Pennylane::LightningTensor::Cutn {
template <class Precision, class Derived> class TensornetBase {
  private:
    size_t numQubits_;
    std::vector<size_t> qubitDims_;

  public:
    TensornetBase() = delete;

    explicit TensornetBase(const size_t numQubits)
        : numQubits_(numQubits){
        qubitDims_.resize(numQubits);
        std::fill(qubitDims_.begin(), qubitDims_.end(), size_t{2});
    }

    virtual ~TensornetBase() = default;

    /**
     * @brief Get dimension of each qubit
     *
     * @return std::vector<size_t> &
     */
    [[nodiscard]] auto getQubitDims() -> std::vector<size_t> & {
        return qubitDims_;
    };

    /**
     * @brief Get the number of qubits of the simulated system.
     *
     * @return std::size_t
     */
    [[nodiscard]] auto getNumQubits() const -> size_t { return numQubits_; };
};
} // namespace Pennylane::LightningTensor::MPS
