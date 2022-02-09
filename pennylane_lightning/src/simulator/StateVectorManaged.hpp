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

#include "BitUtil.hpp"
#include "StateVectorBase.hpp"
#include "Util.hpp"

namespace Pennylane {

/**
 * @brief Managed memory version of StateVector class. Memory ownership resides
 * within class.
 *
 * This class is only internally used in C++ code.
 *
 * @tparam PrecisionT
 */
template <class PrecisionT = double>
class StateVectorManaged
    : public StateVectorBase<PrecisionT, StateVectorManaged<PrecisionT>> {
  public:
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    using BaseType = StateVectorBase<PrecisionT, StateVectorManaged>;

    std::vector<ComplexPrecisionT> data_;

  public:
    StateVectorManaged() : StateVectorBase<PrecisionT, StateVectorManaged>() {}

    explicit StateVectorManaged(size_t num_qubits)
        : BaseType(num_qubits),
          data_(static_cast<size_t>(Util::exp2(num_qubits)),
                ComplexPrecisionT{0, 0}) {
        data_[0] = {1, 0};
    }

    template <class OtherDerived>
    explicit StateVectorManaged(
        const StateVectorBase<PrecisionT, OtherDerived> &other)
        : BaseType(other.getNumQubits()), data_{other.getData(),
                                                other.getData() +
                                                    other.getLength()} {}

    explicit StateVectorManaged(
        const std::vector<ComplexPrecisionT> &other_data)
        : BaseType(Util::log2(other_data.size())), data_{other_data} {
        PL_ABORT_IF_NOT(Util::isPerfectPowerOf2(other_data.size()),
                        "The size of provided data must be a power of 2.");
    }

    StateVectorManaged(const ComplexPrecisionT *other_data, size_t other_size)
        : BaseType(Util::log2(other_size)), data_{other_data,
                                                  other_data + other_size} {
        PL_ABORT_IF_NOT(Util::isPerfectPowerOf2(other_size),
                        "The size of provided data must be a power of 2.");
    }

    StateVectorManaged(const StateVectorManaged<PrecisionT> &other) = default;
    StateVectorManaged(StateVectorManaged<PrecisionT> &&other) noexcept =
        default;

    ~StateVectorManaged() = default;

    auto operator=(const StateVectorManaged<PrecisionT> &other)
        -> StateVectorManaged<PrecisionT> & = default;
    auto operator=(StateVectorManaged<PrecisionT> &&other) noexcept
        -> StateVectorManaged<PrecisionT> & = default;

    auto getDataVector() -> std::vector<ComplexPrecisionT> & { return data_; }
    [[nodiscard]] auto getDataVector() const
        -> const std::vector<ComplexPrecisionT> & {
        return data_;
    }

    [[nodiscard]] auto getData() -> ComplexPrecisionT * { return data_.data(); }

    [[nodiscard]] auto getData() const -> const ComplexPrecisionT * {
        return data_.data();
    }

    /**
     * @brief Update data of the class to new_data
     *
     * @param new_data std::vector contains data.
     */
    void updateData(const std::vector<ComplexPrecisionT> &new_data) {
        PL_ABORT_IF_NOT(data_.size() == new_data.size(),
                        "New data must be the same size as old data.")
        std::copy(new_data.begin(), new_data.end(), data_.begin());
    }
};

} // namespace Pennylane
