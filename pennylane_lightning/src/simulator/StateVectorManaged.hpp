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
template <class fp_t = double>
class StateVectorManaged : public StateVector<fp_t> {
  private:
    using CFP_t = std::complex<fp_t>;

    std::vector<CFP_t> data_;

  public:
    StateVectorManaged() : StateVector<fp_t>() {}
    StateVectorManaged(size_t num_qubits)
        : data_(static_cast<size_t>(Util::exp2(num_qubits)), CFP_t{0, 0}),
          StateVector<fp_t>(nullptr,
                            static_cast<size_t>(Util::exp2(num_qubits))) {
        StateVector<fp_t>::setData(data_.data());
        data_[0] = {1, 0};
    }
    StateVectorManaged(const StateVector<fp_t> &other)
        : data_{other.getData(), other.getData() + other.getLength()},
          StateVector<fp_t>(nullptr, data_.size()) {
        StateVector<fp_t>::setData(data_.data());
    }
    StateVectorManaged(const std::vector<CFP_t> &other_data)
        : data_{other_data}, StateVector<fp_t>(nullptr, other_data.size()) {
        StateVector<fp_t>::setData(data_.data());
    }
    StateVectorManaged(const CFP_t *other_data, size_t other_size)
        : data_{other_data, other_data + other_size}, StateVector<fp_t>(
                                                          nullptr, other_size) {
        StateVector<fp_t>::setData(data_.data());
    }
    StateVectorManaged(const StateVectorManaged<fp_t> &other)
        : data_{other.data_}, StateVector<fp_t>(nullptr,
                                                other.getDataVector().size()) {
        StateVector<fp_t>::setData(data_.data());
    }

    StateVectorManaged &operator=(const StateVectorManaged<fp_t> &other) {
        if (this != &other) {
            if (data_.size() != other.getLength()) {
                data_.resize(other.getLength());
                StateVector<fp_t>::setData(data_.data());
                StateVector<fp_t>::setLength(other.getLength());
            }
            std::copy(other.data_.data(),
                      other.data_.data() + other.getLength(), data_.data());
        }
        return *this;
    }
    std::vector<CFP_t> &getDataVector() { return data_; }
    const std::vector<CFP_t> &getDataVector() const { return data_; }

    std::vector<size_t>
    getInternalIndices(const std::vector<size_t> &qubit_indices) {
        return StateVector<fp_t>::generateBitPatterns(qubit_indices,
                                                      Util::log2(data_.size()));
    }
    std::vector<size_t>
    getExternalIndices(const std::vector<size_t> &qubit_indices) {
        std::vector<size_t> externalWires =
            StateVector<fp_t>::getIndicesAfterExclusion(
                qubit_indices, Util::log2(data_.size()));
        std::vector<size_t> externalIndices =
            StateVector<fp_t>::generateBitPatterns(externalWires,
                                                   Util::log2(data_.size()));
        return externalIndices;
    }
    void updateData(const std::vector<CFP_t> &new_data) {
        PL_ABORT_IF_NOT(data_.size() == new_data.size(),
                        "New data must be the same size as old data.")
        std::copy(new_data.begin(), new_data.end(), data_.begin());
    }
};

} // namespace Pennylane