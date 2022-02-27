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
 * @brief StateVector class where data resides in CPU memory.
 *
 * @tparam PrecisionT Data floating point type
 * @tparam Derived Derived class for CRTP.
 */
template <class PrecisionT, class Derived>
class StateVectorCPU : public StateVectorBase<PrecisionT, Derived> {
  public:
    using ComplexPrecisionT = std::complex<PrecisionT>;

  private:
    using BaseType = StateVectorBase<PrecisionT, Derived>;

    Threading threading_;
    CPUMemoryModel memory_model_;

    std::unordered_map<Gates::GateOperation, Gates::KernelType>
        kernel_for_gates_;
    std::unordered_map<Gates::GeneratorOperation, Gates::KernelType>
        kernel_for_generators_;

    void setKernels(size_t num_qubits, Threading threading,
                    CPUMemoryModel memory_model) {
        auto &default_kernels = DefaultKernelsForStateVector::getInstance();
        kernel_for_gates_ = default_kernels.getGateKernelMap(
            num_qubits, threading, memory_model);
        kernel_for_generators_ = default_kernels.getGeneratorKernelMap(
            num_qubits, threading, memory_model);
    }

  protected:
    explicit StateVectorCPU(size_t num_qubits, Threading threading,
                            CPUMemoryModel memory_model)
        : BaseType(num_qubits), threading_{threading}, memory_model_{
                                                           memory_model} {
        setKernels(num_qubits, threading, memory_model);
    }

  public:
    [[nodiscard]] inline auto
    getKernelForGate(Gates::GateOperation gate_op) const -> Gates::KernelType {
        return kernel_for_gates_.at(gate_op);
    }

    [[nodiscard]] inline auto
    getKernelForGenerator(Gates::GeneratorOperation gntr_op) const
        -> Gates::KernelType {
        return kernel_for_generators_.at(gntr_op);
    }
};

} // namespace Pennylane
