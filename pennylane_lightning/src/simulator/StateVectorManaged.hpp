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

#include "StateVectorBase.hpp"

namespace Pennylane {

/**
 * @brief Managed memory version of StateVector class. Memory ownership resides
 * within class.
 *
 * This class is only internally used in C++ code.
 *
 * @tparam fp_t
 */
template <class fp_t = double>
class StateVectorManaged
    : public StateVectorBase<fp_t, StateVectorManaged<fp_t>> {
  public:
    using scalar_type_t = fp_t;
    using CFP_t = std::complex<fp_t>;

  private:
    using BaseType = StateVectorBase<fp_t, StateVectorManaged>;

    std::vector<CFP_t> data_;

  public:
    StateVectorManaged() : StateVectorBase<fp_t, StateVectorManaged>() {}

    explicit StateVectorManaged(size_t num_qubits)
        : BaseType(num_qubits),
          data_(static_cast<size_t>(Util::exp2(num_qubits)), CFP_t{0, 0}) {
        data_[0] = {1, 0};
    }

    template <class OtherDerived>
    StateVectorManaged(const StateVectorBase<fp_t, OtherDerived> &other)
        : BaseType(other.getNumQubits()), data_{other.getData(),
                                                other.getData() +
                                                    other.getLength()} {}

    StateVectorManaged(const std::vector<CFP_t> &other_data)
        : BaseType(Util::log2(other_data.size())), data_{other_data} {
        PL_ABORT_IF_NOT(Util::isPerfectPowerOf2(other_data.size()),
                        "The size of provided data must be a power of 2.");
    }

    StateVectorManaged(const CFP_t *other_data, size_t other_size)
        : BaseType(Util::log2(other_size)), data_{other_data,
                                                  other_data + other_size} {
        PL_ABORT_IF_NOT(Util::isPerfectPowerOf2(other_size),
                        "The size of provided data must be a power of 2.");
    }

    StateVectorManaged(const StateVectorManaged<fp_t> &other) = default;
    StateVectorManaged(StateVectorManaged<fp_t> &&other) noexcept = default;

    auto operator=(const StateVectorManaged<fp_t> &other)
        -> StateVectorManaged<fp_t> & = default;
    auto operator=(StateVectorManaged<fp_t> &&other) noexcept
        -> StateVectorManaged<fp_t> & = default;

    auto getDataVector() -> std::vector<CFP_t> & { return data_; }
    [[nodiscard]] auto getDataVector() const -> const std::vector<CFP_t> & {
        return data_;
    }

    [[nodiscard]] auto getData() -> CFP_t * { return data_.data(); }

    [[nodiscard]] auto getData() const -> const CFP_t * { return data_.data(); }

    void updateData(const std::vector<CFP_t> &new_data) {
        PL_ABORT_IF_NOT(data_.size() == new_data.size(),
                        "New data must be the same size as old data.")
        std::copy(new_data.begin(), new_data.end(), data_.begin());
    }
};

} // namespace Pennylane
