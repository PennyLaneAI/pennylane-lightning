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
#include "CPUMemoryModel.hpp"
#include "DispatchKeys.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "Memory.hpp"
#include "StateVectorBase.hpp"
#include "StateVectorCPU.hpp"
#include "Util.hpp"

namespace Pennylane {

/**
 * @brief StateVector class where data resides in CPU memory. Memory ownership
 * resides within class.
 *
 * @tparam PrecisionT
 */
template <class PrecisionT = double>
class StateVectorManagedCPU
    : public StateVectorCPU<PrecisionT, StateVectorManagedCPU<PrecisionT>> {
  public:
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    using BaseType = StateVectorCPU<PrecisionT, StateVectorManagedCPU>;

    // NOLINTNEXTLINE(modernize-avoid-c-arrays,hicpp-avoid-c-arrays)
    std::vector<ComplexPrecisionT, AlignedAllocator<ComplexPrecisionT>> data_;

  public:
    explicit StateVectorManagedCPU(
        size_t num_qubits, Threading threading = bestThreading(),
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType{num_qubits, threading, memory_model},
          data_{Util::exp2(num_qubits), ComplexPrecisionT{0.0, 0.0},
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {
        data_[0] = {1, 0};
    }

    template <class OtherDerived>
    explicit StateVectorManagedCPU(
        const StateVectorCPU<PrecisionT, OtherDerived> &other)
        : BaseType(other.getNumQubits(), other.threading(),
                   other.memoryModel()),
          data_{other.getData(), other.getData() + other.getLength(),
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {}

    StateVectorManagedCPU(const ComplexPrecisionT *other_data,
                          size_t other_size,
                          Threading threading = bestThreading(),
                          CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType(Util::log2PerfectPower(other_size), threading, memory_model),
          data_{other_data, other_data + other_size,
                getAllocator<ComplexPrecisionT>(this->memory_model_)} {
        PL_ABORT_IF_NOT(Util::isPerfectPowerOf2(other_size),
                        "The size of provided data must be a power of 2.");
    }

    // Clang-tidy gives false positive for delegating constructor
    template <class Alloc>
    // NOLINTNEXTLINE(hicpp-member-init)
    explicit StateVectorManagedCPU(
        const std::vector<std::complex<PrecisionT>, Alloc> &rhs,
        Threading threading = bestThreading(),
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : StateVectorManagedCPU(rhs.data(), rhs.size(), threading,
                                memory_model) {}

    StateVectorManagedCPU(const StateVectorManagedCPU &rhs) = default;
    StateVectorManagedCPU(StateVectorManagedCPU &&) noexcept = default;

    StateVectorManagedCPU &operator=(const StateVectorManagedCPU &) = default;
    StateVectorManagedCPU &
    operator=(StateVectorManagedCPU &&) noexcept = default;

    ~StateVectorManagedCPU() = default;

    [[nodiscard]] auto getData() -> ComplexPrecisionT * { return data_.data(); }

    [[nodiscard]] auto getData() const -> const ComplexPrecisionT * {
        return data_.data();
    }

    /**
     * @brief Update data of the class to new_data
     *
     * @param new_data std::vector contains data.
     */
    void updateData(const ComplexPrecisionT *data) {
        std::copy(data, data + BaseType::getLength(), data_.data());
    }
};
} // namespace Pennylane
