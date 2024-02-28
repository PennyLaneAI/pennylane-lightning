// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
 * @file
 * Statevector simulator where data management resides inside the class.
 */

#pragma once

#include <algorithm> // fill
#include <complex>
#include <iostream>
#include <vector>

#include "BitUtil.hpp"        // log2PerfectPower, isPerfectPowerOf2
#include "CPUMemoryModel.hpp" // bestCPUMemoryModel
#include "Error.hpp"
#include "KernelType.hpp"
#include "Memory.hpp"
#include "StateVectorLQubit.hpp"
#include "Threading.hpp"
#include "Util.hpp" // exp2

/// @cond DEV
namespace {
using Pennylane::Util::AlignedAllocator;
using Pennylane::Util::bestCPUMemoryModel;
using Pennylane::Util::exp2;
using Pennylane::Util::isPerfectPowerOf2;
using Pennylane::Util::log2PerfectPower;
using Pennylane::Util::squaredNorm;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit {
/**
 * @brief StateVector class where data resides in CPU memory. Memory ownership
 * resides within class.
 *
 * @tparam fp_t Precision data type
 */
template <class fp_t = double>
class StateVectorLQubitManaged final
    : public StateVectorLQubit<fp_t, StateVectorLQubitManaged<fp_t>> {
  public:
    using PrecisionT = fp_t;
    using ComplexT = std::complex<PrecisionT>;
    using CFP_t = ComplexT;
    using MemoryStorageT = Pennylane::Util::MemoryStorageLocation::Internal;

  private:
    using BaseType =
        StateVectorLQubit<PrecisionT, StateVectorLQubitManaged<PrecisionT>>;
    std::vector<ComplexT, AlignedAllocator<ComplexT>> data_;

  public:
    /**
     * @brief Create a new statevector in the computational basis state |0...0>
     *
     * @param num_qubits Number of qubits
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    explicit StateVectorLQubitManaged(
        std::size_t num_qubits, Threading threading = Threading::SingleThread,
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType{num_qubits, threading, memory_model},
          data_{exp2(num_qubits), ComplexT{0.0, 0.0},
                getAllocator<ComplexT>(this->memory_model_)} {
        setBasisState(0U);
    }

    /**
     * @brief Construct a statevector from another statevector
     *
     * @tparam OtherDerived A derived type of StateVectorLQubit to use for
     * construction.
     * @param other Another statevector to construct the statevector from
     */
    template <class OtherDerived>
    explicit StateVectorLQubitManaged(
        const StateVectorLQubit<PrecisionT, OtherDerived> &other)
        : BaseType(other.getNumQubits(), other.threading(),
                   other.memoryModel()),
          data_{other.getData(), other.getData() + other.getLength(),
                getAllocator<ComplexT>(this->memory_model_)} {}

    /**
     * @brief Construct a statevector from data pointer
     *
     * @param other_data Data pointer to construct the statevector from.
     * @param other_size Size of the data
     * @param threading Threading option the statevector to use
     * @param memory_model Memory model the statevector will use
     */
    StateVectorLQubitManaged(const ComplexT *other_data, std::size_t other_size,
                             Threading threading = Threading::SingleThread,
                             CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType(log2PerfectPower(other_size), threading, memory_model),
          data_{other_data, other_data + other_size,
                getAllocator<ComplexT>(this->memory_model_)} {
        PL_ABORT_IF_NOT(isPerfectPowerOf2(other_size),
                        "The size of provided data must be a power of 2.");
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
    explicit StateVectorLQubitManaged(
        const std::vector<std::complex<PrecisionT>, Alloc> &other,
        Threading threading = Threading::SingleThread,
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : StateVectorLQubitManaged(other.data(), other.size(), threading,
                                   memory_model) {}

    StateVectorLQubitManaged(const StateVectorLQubitManaged &rhs) = default;
    StateVectorLQubitManaged(StateVectorLQubitManaged &&) noexcept = default;

    StateVectorLQubitManaged &
    operator=(const StateVectorLQubitManaged &) = default;
    StateVectorLQubitManaged &
    operator=(StateVectorLQubitManaged &&) noexcept = default;

    ~StateVectorLQubitManaged() = default;

    /**
     * @brief Prepares a single computational basis state.
     *
     * @param index Index of the target element.
     */
    void setBasisState(const std::size_t index) {
        std::fill(data_.begin(), data_.end(), 0);
        data_[index] = {1, 0};
    }

    /**
     * @brief Set values for a batch of elements of the state-vector.
     *
     * @param values Values to be set for the target elements.
     * @param indices Indices of the target elements.
     */
    void setStateVector(const std::vector<std::size_t> &indices,
                        const std::vector<ComplexT> &values) {
        for (std::size_t n = 0; n < indices.size(); n++) {
            data_[indices[n]] = values[n];
        }
    }

    /**
     * @brief Reset the data back to the \f$\ket{0}\f$ state.
     *
     */
    void resetStateVector() {
        if (this->getLength() > 0) {
            setBasisState(0U);
        }
    }

    [[nodiscard]] auto getData() -> ComplexT * { return data_.data(); }

    [[nodiscard]] auto getData() const -> const ComplexT * {
        return data_.data();
    }

    /**
     * @brief Get underlying data vector
     */
    [[nodiscard]] auto getDataVector()
        -> std::vector<ComplexT, AlignedAllocator<ComplexT>> & {
        return data_;
    }

    [[nodiscard]] auto getDataVector() const
        -> const std::vector<ComplexT, AlignedAllocator<ComplexT>> & {
        return data_;
    }

    /**
     * @brief Update data of the class to new_data
     *
     * @param new_data data pointer to new data.
     * @param new_size size of underlying data storage.
     */
    void updateData(const ComplexT *new_data, std::size_t new_size) {
        PL_ASSERT(data_.size() == new_size);
        std::copy(new_data, new_data + new_size, data_.data());
    }

    /**
     * @brief Update data of the class to new_data
     *
     * @tparam Alloc Allocator type of std::vector to use for updating data.
     * @param new_data std::vector contains data.
     */
    template <class Alloc>
    void updateData(const std::vector<ComplexT, Alloc> &new_data) {
        updateData(new_data.data(), new_data.size());
    }

    AlignedAllocator<ComplexT> allocator() const {
        return data_.get_allocator();
    }

    /**
     * @brief Compute probabilities of measuring 0 or 1 on one specific wire
     *
     * @param wire Wire
     * @return std::vector<PrecisionT> Both probabilities of getting 0 or 1
     */
    auto probs(const std::size_t wire) -> std::vector<PrecisionT> {
        const auto stride = pow(2, this->num_qubits_ - (1 + wire));
        const auto vec_size = pow(2, this->num_qubits_);
        const auto section_size = vec_size / stride;
        const auto half_section_size = section_size / 2;

        PrecisionT prob_0{0.};
        // zero half the entries
        // the "half" entries depend on the stride
        // *_*_*_*_ for stride 1
        // **__**__ for stride 2
        // ****____ for stride 4
        const size_t k = 0;
        for (size_t idx = 0; idx < half_section_size; idx++) {
            for (size_t ids = 0; ids < stride; ids++) {
                auto v = stride * (k + 2 * idx) + ids;
                prob_0 += std::norm(data_[v]);
            }
        }

        prob_0 = sqrt(prob_0);
        const std::vector<PrecisionT> probs{prob_0, PrecisionT(1.) - prob_0};
        return probs;
    }

    /**
     * @brief Collapse state
     *
     * @param wire Wire to collapse.
     * @param branch Branch 0 or 1.
     */
    void collapse(const std::size_t wire, const bool branch) {
        const auto stride = pow(2, this->num_qubits_ - (1 + wire));
        const auto vec_size = pow(2, this->num_qubits_);
        const auto section_size = vec_size / stride;
        const auto half_section_size = section_size / 2;

        // zero half the entries
        // the "half" entries depend on the stride
        // *_*_*_*_ for stride 1
        // **__**__ for stride 2
        // ****____ for stride 4
        const size_t k = branch ? 0 : 1;
        for (size_t idx = 0; idx < half_section_size; idx++) {
            for (size_t ids = 0; ids < stride; ids++) {
                auto v = stride * (k + 2 * idx) + ids;
                data_[v] = {0., 0.};
            }
        }

        PrecisionT norm =
            std::sqrt(squaredNorm(data_.data(), this->getLength()));

        if (norm > std::numeric_limits<PrecisionT>::epsilon() * 1e2) {
            std::complex<PrecisionT> inv_norm = 1. / norm;
            for (size_t k = 0; k < this->getLength(); k++) {
                data_[k] *= inv_norm;
            }
        }
    }
};
} // namespace Pennylane::LightningQubit
