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
#include "Gates.hpp"
#include "KernelType.hpp"
#include "Memory.hpp"
#include "StateVectorBase.hpp"
#include "StateVectorCPU.hpp"
#include "Threading.hpp"
#include "Util.hpp"
#include "Error.hpp"

#include <cstring>
#include <span>

namespace Pennylane {

/**
 * @brief StateVector class where data resides in CPU memory. Memory ownership
 * resides within class.
 *
 * @tparam PrecisionT Precision data type
 */
template <class PrecisionT = double>
class StateVectorManagedCPU
    : public StateVectorCPU<PrecisionT, StateVectorManagedCPU<PrecisionT>> {
  public:
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    using BaseType = StateVectorCPU<PrecisionT, StateVectorManagedCPU>;

    ComplexPrecisionT* data_;

  public:
    /**
     * @brief Create a new statevector
     *
     * @param num_qubits Number of qubits
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    explicit StateVectorManagedCPU(
        size_t num_qubits, Threading threading = Threading::SingleThread,
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType{num_qubits, threading, memory_model} {
        data_ = getAllocator<ComplexPrecisionT>(this->memory_model_).allocate(this->getLength());
        std::fill(data_, data_ + this->getLength(), ComplexPrecisionT{0.0, 0.0});
        data_[0] = {1, 0};
    }

    /**
     * @brief Construct a statevector from another statevector
     *
     * @tparam OtherDerived A derived type of StateVectorCPU to use for
     * construction.
     * @param other Another statevector to construct the statevector from
     */
    template <class OtherDerived>
    explicit StateVectorManagedCPU(
        const StateVectorCPU<PrecisionT, OtherDerived> &other)
        : BaseType(other.getNumQubits(), other.threading(),
                   other.memoryModel()) {
        data_ = getAllocator<ComplexPrecisionT>(this->memory_model_).allocate(other.getLength());
        std::copy(other.getData(), other.getData() + other.getLength(), data_);
    }

    /**
     * @brief Construct a statevector from data pointer
     *
     * @param other_data Data pointer to construct the statvector from.
     * @param other_size Size of the data
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    StateVectorManagedCPU(const ComplexPrecisionT *other_data,
                          size_t other_size,
                          Threading threading = Threading::SingleThread,
                          CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType(Util::log2PerfectPower(other_size), threading, memory_model)
    {
        PL_ABORT_IF_NOT(Util::isPerfectPowerOf2(other_size),
                        "The size of provided data must be a power of 2.");
        data_ = getAllocator<ComplexPrecisionT>(this->memory_model_).allocate(other_size);
        std::copy(other_data, other_data + other_size, data_);
    }

    /**
     * @brief Construct a statevector from a data vector
     *
     * @tparam Alloc Allocator type of std::vector to use for constructing
     * statevector.
     * @param other Data to construct the statevector from
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    template <class Alloc>
    explicit StateVectorManagedCPU(
        const std::vector<std::complex<PrecisionT>, Alloc> &other,
        Threading threading = Threading::SingleThread,
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : StateVectorManagedCPU(other.data(), other.size(), threading,
                                memory_model) {}

    StateVectorManagedCPU(const StateVectorManagedCPU &rhs)
        : BaseType(rhs) {
        data_ = getAllocator<ComplexPrecisionT>(this->memory_model_).allocate(this->getLength());
        std::copy(rhs.data_, rhs.data_ + this->getLength(), data_);
    }

    StateVectorManagedCPU(StateVectorManagedCPU && rhs) noexcept
        : BaseType(std::move(rhs)) {
        data_ = rhs.data_;
        rhs.data_ = nullptr;
    }

    /* Use updateData instead */
    StateVectorManagedCPU &operator=(const StateVectorManagedCPU &rhs) = delete;
    StateVectorManagedCPU &operator=(StateVectorManagedCPU &&) noexcept = delete;

    ~StateVectorManagedCPU() {
        getAllocator<ComplexPrecisionT>(this->memory_model_).deallocate(data_, this->getLength());
    }

    [[nodiscard]] auto getData() -> ComplexPrecisionT * { return data_; }

    [[nodiscard]] auto getData() const -> const ComplexPrecisionT * {
        return data_;
    }

    /**
     * @brief Update data of the class to new_data
     *
     * @param new_data new pointer contains data.
     */
    void updateData(std::span<const ComplexPrecisionT> new_data) {
        PL_ASSERT(new_data.size() == this->getLength());
        std::copy(new_data.data(), new_data.data() + new_data.size(), data_);
    }

    /**
     * @brief Update data of the class to new_data
     *
     * @param rhs new statevector 
     */
    void updateData(const StateVectorManagedCPU<PrecisionT>& rhs) {
        updateData(std::span{rhs.data_, rhs.getLength()});
    }

    /**
     * @brief Update data of the class to new_data
     *
     * @tparam Alloc Allocator type of std::vector to use for updating data.
     * @param new_data std::vector contains data.
     */
    template <typename Alloc>
    void updateData(const std::vector<ComplexPrecisionT, Alloc>& new_data) {
        PL_ASSERT(this->getLength() == new_data.size());
        std::copy(new_data.begin(), new_data.end(), data_);
    }

    auto toVector() ->
    std::vector<ComplexPrecisionT, Util::AlignedAllocator<ComplexPrecisionT>> {
        return std::vector<ComplexPrecisionT, Util::AlignedAllocator<ComplexPrecisionT>>(
                data_, data_ + this->getLength(), getAllocator<ComplexPrecisionT>(this->memory_model_)
        );
    }

    Util::AlignedAllocator<ComplexPrecisionT> allocator() const {
        return getAllocator<ComplexPrecisionT>(this->memory_model_);
    }
};
} // namespace Pennylane
