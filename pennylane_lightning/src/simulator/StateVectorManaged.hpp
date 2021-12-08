// Copyright 2021 Xanadu Quantum Technologies Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "StateVector.hpp"

namespace Pennylane {

/**
 * @brief Managed memory version of StateVector class. Memory ownership resides
 * within class.
 *
 * @tparam fp_t
 */
template <class fp_t = double, std::size_t STACK_BYTES = 8192>
class StateVectorManaged : public StateVector<fp_t, STACK_BYTES> {
  private:
    using CFP_t = std::complex<fp_t>;

    std::vector<CFP_t> data_;

  public:
    StateVectorManaged() : StateVector<fp_t, STACK_BYTES>() {}
    StateVectorManaged(size_t num_qubits)
        : StateVector<fp_t, STACK_BYTES>(
              nullptr, static_cast<size_t>(Util::exp2(num_qubits))),
          data_(static_cast<size_t>(Util::exp2(num_qubits)), CFP_t{0, 0}) {
        StateVector<fp_t, STACK_BYTES>::setData(data_.data());
        data_[0] = {1, 0};
    }
    StateVectorManaged(const StateVector<fp_t, STACK_BYTES> &other)
        : StateVector<fp_t, STACK_BYTES>(nullptr, data_.size()),
          data_{other.getData(), other.getData() + other.getLength()} {
        StateVector<fp_t, STACK_BYTES>::setData(data_.data());
    }
    StateVectorManaged(const std::vector<CFP_t> &other_data)
        : StateVector<fp_t, STACK_BYTES>(nullptr, other_data.size()),
          data_{other_data} {
        StateVector<fp_t, STACK_BYTES>::setData(data_.data());
    }
    StateVectorManaged(const CFP_t *other_data, size_t other_size)
        : StateVector<fp_t, STACK_BYTES>(nullptr, other_size),
          data_{other_data, other_data + other_size} {
        StateVector<fp_t, STACK_BYTES>::setData(data_.data());
    }
    StateVectorManaged(const StateVectorManaged<fp_t> &other)
        : StateVector<fp_t>(nullptr, other.getDataVector().size()),
          data_{other.data_} {
        StateVector<fp_t, STACK_BYTES>::setData(data_.data());
    }

    auto operator=(const StateVectorManaged<fp_t> &other)
        -> StateVectorManaged & {
        if (this != &other) {
            if (data_.size() != other.getLength()) {
                data_.resize(other.getLength());
                StateVector<fp_t, STACK_BYTES>::setData(data_.data());
                StateVector<fp_t, STACK_BYTES>::setLength(other.getLength());
            }
            std::copy(other.data_.data(),
                      other.data_.data() + other.getLength(), data_.data());
        }
        return *this;
    }
    auto getDataVector() -> std::vector<CFP_t> & { return data_; }
    [[nodiscard]] auto getDataVector() const -> const std::vector<CFP_t> & {
        return data_;
    }

    template <typename VecType = std::pmr::vector<std::size_t>>
    auto getInternalIndices(const VecType &qubit_indices)
        -> std::pmr::vector<std::size_t> {
        return StateVector<fp_t, STACK_BYTES>::generateBitPatterns(
            qubit_indices);
    }

    template <typename VecType = std::pmr::vector<std::size_t>>
    auto getExternalIndices(const VecType &qubit_indices)
        -> std::pmr::vector<std::size_t> {
        auto externalWires =
            StateVector<fp_t, STACK_BYTES>::getIndicesAfterExclusion(
                qubit_indices);
        auto externalIndices =
            StateVector<fp_t, STACK_BYTES>::generateBitPatterns(externalWires);
        return externalIndices;
    }

    void updateData(const std::vector<CFP_t> &new_data) {
        PL_ABORT_IF_NOT(data_.size() == new_data.size(),
                        "New data must be the same size as old data.")
        std::copy(new_data.begin(), new_data.end(), data_.begin());
    }
};

} // namespace Pennylane