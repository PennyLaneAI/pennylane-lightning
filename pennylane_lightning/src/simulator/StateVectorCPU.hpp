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
#include "DefaultKernelsForStateVector.hpp"
#include "DispatchKeys.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "Memory.hpp"
#include "StateVectorBase.hpp"
#include "Util.hpp"

namespace Pennylane {

/**
 * @brief StateVector class where data resides in CPU memory. Memory ownership
 * resides within class.
 *
 * We currently use std::unique_ptr to C-style array as we want to choose
 * allocator in runtime. This is impossible with std::vector.
 *
 * @tparam PrecisionT
 */
template <class PrecisionT = double>
class StateVectorCPU
    : public StateVectorBase<PrecisionT, StateVectorCPU<PrecisionT>> {
  public:
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    using BaseType = StateVectorBase<PrecisionT, StateVectorCPU>;

    Threading threading_;
    CPUMemoryModel memory_model_;

    std::unordered_map<Gates::GateOperation, Gates::KernelType>
        kernel_for_gates_;
    std::unordered_map<Gates::GeneratorOperation, Gates::KernelType>
        kernel_for_generators_;
    std::unique_ptr<ComplexPrecisionT[]>
        data_; // NOLINT(modernize-avoid-c-arrays)

    void setKernels(size_t num_qubits, Threading threading,
                    CPUMemoryModel memory_model) {
        auto &default_kernels = DefaultKernelsForStateVector::getInstance();
        kernel_for_gates_ = default_kernels.getGateKernelMap(
            num_qubits, threading, memory_model);
        kernel_for_generators_ = default_kernels.getGeneratorKernelMap(
            num_qubits, threading, memory_model);
    }

  public:
    explicit StateVectorCPU(size_t num_qubits,
                            Threading threading = bestThreading(),
                            CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType(num_qubits), threading_{threading}, memory_model_{
                                                           memory_model} {

        setKernels(num_qubits, threading, memory_model);

        size_t length = BaseType::getLength();
        data_ = std::unique_ptr<ComplexPrecisionT[]>{new (std::align_val_t{
            64}) ComplexPrecisionT[length]}; // NOLINT(modernize-avoid-c-arrays)
        std::fill(data_.get(), data_.get() + length,
                  ComplexPrecisionT{0.0, 0.0});
        data_[0] = {1, 0};
    }

    template <class OtherDerived>
    explicit StateVectorCPU(
        const StateVectorBase<PrecisionT, OtherDerived> &other,
        Threading threading = bestThreading(),
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType(other.getNumQubits()), threading_{threading},
          memory_model_{memory_model} {

        size_t length = BaseType::getLength();
        data_ = std::unique_ptr<ComplexPrecisionT[]>{new (std::align_val_t{
            64}) ComplexPrecisionT[length]}; // NOLINT(modernize-avoid-c-arrays)

        std::copy(other.getData(), other.getData() + length, data_.get());

        setKernels(BaseType::getNumQubits(), threading, memory_model);
    }

    StateVectorCPU(const ComplexPrecisionT *other_data, size_t other_size,
                   Threading threading = bestThreading(),
                   CPUMemoryModel memory_model = bestCPUMemoryModel())
        : BaseType(Util::log2PerfectPower(other_size)), threading_{threading},
          memory_model_{memory_model} {
        PL_ABORT_IF_NOT(Util::isPerfectPowerOf2(other_size),
                        "The size of provided data must be a power of 2.");
        data_ = std::unique_ptr<ComplexPrecisionT[]>{
            new (std::align_val_t{64}) ComplexPrecisionT
                [other_size]}; // NOLINT(modernize-avoid-c-arrays)
        setKernels(BaseType::getNumQubits(), threading, memory_model);

        updateData(other_data);
    }

    template <class Alloc>
    explicit StateVectorCPU(
        const std::vector<std::complex<PrecisionT>, Alloc> &rhs,
        Threading threading = bestThreading(),
        CPUMemoryModel memory_model = bestCPUMemoryModel())
        : StateVectorCPU(rhs.data(), rhs.size(), threading,
                         memory_model) // NOLINT(hicpp-member-init)
                                       // this is false positive for delegating
                                       // constructor from clang-tidy
    {}

    StateVectorCPU(const StateVectorCPU &rhs)
        : BaseType(rhs.getNumQubits()), threading_{rhs.threading_},
          memory_model_{rhs.memory_model_} {
        setKernels(BaseType::getNumQubits(), threading_, memory_model_);

        size_t length = BaseType::getLength();
        data_ = std::unique_ptr<ComplexPrecisionT[]>{new (std::align_val_t{
            64}) ComplexPrecisionT[length]}; // NOLINT(modernize-avoid-c-arrays)
        std::copy(rhs.getData(), rhs.getData() + length, data_.get());
    }

    StateVectorCPU(StateVectorCPU &&) noexcept = default;

    StateVectorCPU &operator=(const StateVectorCPU &) = delete;
    StateVectorCPU &operator=(StateVectorCPU &&) noexcept = default;

    ~StateVectorCPU() = default;

    [[nodiscard]] auto getData() -> ComplexPrecisionT * { return data_.get(); }

    [[nodiscard]] auto getData() const -> const ComplexPrecisionT * {
        return data_.get();
    }

    [[nodiscard]] inline auto
    getKernelForGate(Gates::GateOperation gate_op) const -> Gates::KernelType {
        return kernel_for_gates_.at(gate_op);
    }

    [[nodiscard]] inline auto
    getKernelForGenerator(Gates::GeneratorOperation gntr_op) const
        -> Gates::KernelType {
        return kernel_for_generators_.at(gntr_op);
    }

    /**
     * @brief Update data of the class to new_data
     *
     * @param new_data std::vector contains data.
     */
    void updateData(const ComplexPrecisionT *data) {
        std::copy(data, data + BaseType::getLength(), data_.get());
    }
};

} // namespace Pennylane
