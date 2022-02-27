// Copyright 2022 Xanadu Quantum Technologies Inc.

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
 */
#include "DispatchKeys.hpp"
#include "GateOperation.hpp"
#include "KernelType.hpp"

#include <functional>
#include <unordered_map>

namespace Pennylane {

inline auto larger_than(size_t size) {
    return [=](size_t num_qubits) { return num_qubits > size; };
}
inline auto larger_than_equal_to(size_t size) {
    return [=](size_t num_qubits) { return num_qubits >= size; };
}
inline auto less_than(size_t size) {
    return [=](size_t num_qubits) { return num_qubits < size; };
}
inline auto less_than_equal_to(size_t size) {
    return [=](size_t num_qubits) { return num_qubits <= size; };
}
inline auto in_between_closed(size_t l1, size_t l2) {
    return [=](size_t num_qubits) {
        return (l1 <= num_qubits) && (num_qubits <= l2);
    };
}

class DefaultKernelsForStateVector {
  private:
    const static inline std::unordered_map<CPUMemoryModel,
                                           std::vector<Gates::KernelType>>
        allowed_kernels{
            {CPUMemoryModel::Unaligned,
             {Gates::KernelType::LM, Gates::KernelType::PI}},
            {CPUMemoryModel::Aligned256,
             {Gates::KernelType::LM, Gates::KernelType::PI}},
            {CPUMemoryModel::Aligned512,
             {Gates::KernelType::LM, Gates::KernelType::PI}},
        };

    std::unordered_map<
        Gates::GateOperation,
        std::vector<std::tuple<uint32_t, std::function<bool(size_t)>,
                               Gates::KernelType>>>
        gate_kernel_map_;

    std::unordered_map<
        Gates::GeneratorOperation,
        std::vector<std::tuple<uint32_t, std::function<bool(size_t)>,
                               Gates::KernelType>>>
        generator_kernel_map_;

    void registerDefaultGates() {
        using Gates::GateOperation;
        auto &instance = *this;
        auto all_qubit_numbers = []([[maybe_unused]] size_t num_qubits) {
            return true;
        };
        /* Single-qubit gates */
        instance.assignKernelForGate(GateOperation::PauliX, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::PauliY, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::PauliZ, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::Hadamard, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::S, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::T, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::PhaseShift, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::RX, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::RY, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::RZ, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::Rot, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        /* Two-qubit gates */
        instance.assignKernelForGate(GateOperation::CNOT, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CY, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CZ, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::ControlledPhaseShift,
                                     all_threading, all_memory_model,
                                     all_qubit_numbers, Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::SWAP, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);

        instance.assignKernelForGate(
            GateOperation::IsingXX, all_threading,
            // NOLINTNEXTLINE(readability-magic-numbers)
            all_memory_model, less_than(12), Gates::KernelType::LM);
        instance.assignKernelForGate(
            GateOperation::IsingXX, all_threading, all_memory_model,
            // NOLINTNEXTLINE(readability-magic-numbers)
            in_between_closed(12, 20), Gates::KernelType::PI);
        instance.assignKernelForGate(
            GateOperation::IsingXX, all_threading,
            // NOLINTNEXTLINE(readability-magic-numbers)
            all_memory_model, larger_than(20), Gates::KernelType::LM);

        instance.assignKernelForGate(GateOperation::IsingYY, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::IsingZZ, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CRX, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CRY, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CRZ, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::CRot, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
        instance.assignKernelForGate(GateOperation::Toffoli, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::PI);
        instance.assignKernelForGate(GateOperation::CSWAP, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::PI);
        instance.assignKernelForGate(GateOperation::MultiRZ, all_threading,
                                     all_memory_model, all_qubit_numbers,
                                     Gates::KernelType::LM);
    }

    void registerDefaultGenerators() {
        using Gates::GeneratorOperation;
        using Gates::KernelType;
        auto &instance = *this;
        auto all_qubit_numbers = []([[maybe_unused]] size_t num_qubits) {
            return true;
        };

        instance.assignKernelForGenerator(GeneratorOperation::PhaseShift,
                                          all_threading, all_memory_model,
                                          all_qubit_numbers, KernelType::LM);
        instance.assignKernelForGenerator(GeneratorOperation::RX, all_threading,
                                          all_memory_model, all_qubit_numbers,
                                          KernelType::LM);
        instance.assignKernelForGenerator(GeneratorOperation::RY, all_threading,
                                          all_memory_model, all_qubit_numbers,
                                          KernelType::LM);
        instance.assignKernelForGenerator(GeneratorOperation::RZ, all_threading,
                                          all_memory_model, all_qubit_numbers,
                                          KernelType::LM);
        instance.assignKernelForGenerator(GeneratorOperation::IsingXX,
                                          all_threading, all_memory_model,
                                          all_qubit_numbers, KernelType::LM);
        instance.assignKernelForGenerator(GeneratorOperation::IsingYY,
                                          all_threading, all_memory_model,
                                          all_qubit_numbers, KernelType::LM);
        instance.assignKernelForGenerator(GeneratorOperation::IsingZZ,
                                          all_threading, all_memory_model,
                                          all_qubit_numbers, KernelType::LM);
        instance.assignKernelForGenerator(GeneratorOperation::CRX,
                                          all_threading, all_memory_model,
                                          all_qubit_numbers, KernelType::LM);
        instance.assignKernelForGenerator(GeneratorOperation::CRY,
                                          all_threading, all_memory_model,
                                          all_qubit_numbers, KernelType::LM);
        instance.assignKernelForGenerator(GeneratorOperation::CRZ,
                                          all_threading, all_memory_model,
                                          all_qubit_numbers, KernelType::LM);
        instance.assignKernelForGenerator(
            GeneratorOperation::ControlledPhaseShift, all_threading,
            all_memory_model, all_qubit_numbers, KernelType::LM);
        instance.assignKernelForGenerator(GeneratorOperation::MultiRZ,
                                          all_threading, all_memory_model,
                                          all_qubit_numbers, KernelType::LM);
    }

    DefaultKernelsForStateVector() {
        registerDefaultGates();
        registerDefaultGenerators();
    }

  public:
    struct AllThreading {};

    struct AllMemoryModel {};

    constexpr static AllThreading all_threading{};
    constexpr static AllMemoryModel all_memory_model{};

    static auto getInstance() -> DefaultKernelsForStateVector & {
        static DefaultKernelsForStateVector instance;

        return instance;
    }

    void
    assignKernelForGate(Gates::GateOperation gate_op, Threading threading,
                        CPUMemoryModel memory_model,
                        const std::function<bool(size_t)> &num_qubits_criterion,
                        Gates::KernelType kernel) {
        if (std::find(allowed_kernels.at(memory_model).cbegin(),
                      allowed_kernels.at(memory_model).cend(),
                      kernel) == allowed_kernels.at(memory_model).cend()) {
            throw std::invalid_argument("The given kernel is now allowed for "
                                        "the given memory model.");
        }
        gate_kernel_map_[gate_op].emplace_back(
            toDispatchKey(threading, memory_model), num_qubits_criterion,
            kernel);
    }

    void
    assignKernelForGate(Gates::GateOperation gate_op,
                        [[maybe_unused]] AllThreading dummy,
                        CPUMemoryModel memory_model,
                        const std::function<bool(size_t)> &num_qubits_criterion,
                        Gates::KernelType kernel) {
        Util::for_each_enum<Threading>([=](Threading threading) {
            assignKernelForGate(gate_op, threading, memory_model,
                                num_qubits_criterion, kernel);
        });
    }

    void
    assignKernelForGate(Gates::GateOperation gate_op, Threading threading,
                        [[maybe_unused]] AllMemoryModel dummy,
                        const std::function<bool(size_t)> &num_qubits_criterion,
                        Gates::KernelType kernel) {
        Util::for_each_enum<CPUMemoryModel>([=](CPUMemoryModel memory_model) {
            assignKernelForGate(gate_op, threading, memory_model,
                                num_qubits_criterion, kernel);
        });
    }

    void
    assignKernelForGate(Gates::GateOperation gate_op,
                        [[maybe_unused]] AllThreading dummy1,
                        [[maybe_unused]] AllMemoryModel dummy2,
                        const std::function<bool(size_t)> &num_qubits_criterion,
                        Gates::KernelType kernel) {
        Util::for_each_enum<Threading, CPUMemoryModel>(
            [=](Threading threading, CPUMemoryModel memory_model) {
                assignKernelForGate(gate_op, threading, memory_model,
                                    num_qubits_criterion, kernel);
            });
    }

    void assignKernelForGenerator(
        Gates::GeneratorOperation gntr_op, Threading threading,
        CPUMemoryModel memory_model,
        const std::function<bool(size_t)> &num_qubits_criterion,
        Gates::KernelType kernel) {
        if (std::find(allowed_kernels.at(memory_model).cbegin(),
                      allowed_kernels.at(memory_model).cend(),
                      kernel) == allowed_kernels.at(memory_model).cend()) {
            throw std::invalid_argument("The given kernel is now allowed for "
                                        "the given memory model.");
        }
        generator_kernel_map_[gntr_op].emplace_back(
            toDispatchKey(threading, memory_model), num_qubits_criterion,
            kernel);
    }

    void assignKernelForGenerator(
        Gates::GeneratorOperation gntr_op, [[maybe_unused]] AllThreading dummy,
        CPUMemoryModel memory_model,
        const std::function<bool(size_t)> &num_qubits_criterion,
        Gates::KernelType kernel) {
        Util::for_each_enum<Threading>([=](Threading threading) {
            assignKernelForGenerator(gntr_op, threading, memory_model,
                                     num_qubits_criterion, kernel);
        });
    }

    void assignKernelForGenerator(
        Gates::GeneratorOperation gntr_op, Threading threading,
        [[maybe_unused]] AllMemoryModel dummy,
        const std::function<bool(size_t)> &num_qubits_criterion,
        Gates::KernelType kernel) {
        Util::for_each_enum<CPUMemoryModel>([=](CPUMemoryModel memory_model) {
            assignKernelForGenerator(gntr_op, threading, memory_model,
                                     num_qubits_criterion, kernel);
        });
    }

    void assignKernelForGenerator(
        Gates::GeneratorOperation gntr_op, [[maybe_unused]] AllThreading dummy1,
        [[maybe_unused]] AllMemoryModel dummy2,
        const std::function<bool(size_t)> &num_qubits_criterion,
        Gates::KernelType kernel) {
        Util::for_each_enum<Threading, CPUMemoryModel>(
            [=](Threading threading, CPUMemoryModel memory_model) {
                assignKernelForGenerator(gntr_op, threading, memory_model,
                                         num_qubits_criterion, kernel);
            });
    }

    /**
     * @brief Create default kernels for all generators
     * @param num_qubits Number of qubits
     * @param threading Threading context
     * @param memory_model Memory model of the underlying data
     */
    auto getGeneratorKernelMap(size_t num_qubits, Threading threading,
                               CPUMemoryModel memory_model) const
        -> std::unordered_map<Gates::GeneratorOperation, Gates::KernelType> {
        uint32_t dispatch_key = toDispatchKey(threading, memory_model);

        std::unordered_map<Gates::GeneratorOperation, Gates::KernelType>
            kernel_for_generators;

        for (auto generator = Gates::GeneratorOperation::BEGIN;
             generator != Gates::GeneratorOperation::END;
             generator = static_cast<Gates::GeneratorOperation>(
                 static_cast<uint32_t>(generator) + 1)) {

            const auto iter =
                std::find_if(generator_kernel_map_.at(generator).cbegin(),
                             generator_kernel_map_.at(generator).cend(),
                             [dispatch_key = dispatch_key,
                              num_qubits = num_qubits](const auto &t) {
                                 return (std::get<0>(t) == dispatch_key &&
                                         std::get<1>(t)(num_qubits));
                             });
            if (iter == generator_kernel_map_.at(generator).cend()) {
                throw std::range_error("Cannot find registered kernel for a "
                                       "dispatch key and number of qubits.");
            }
            kernel_for_generators.emplace(generator, std::get<2>(*iter));
        }
        return kernel_for_generators;
    }

    auto getGateKernelMap(size_t num_qubits, Threading threading,
                          CPUMemoryModel memory_model) const
        -> std::unordered_map<Gates::GateOperation, Gates::KernelType> {
        uint32_t dispatch_key = toDispatchKey(threading, memory_model);

        std::unordered_map<Gates::GateOperation, Gates::KernelType>
            kernel_for_gates;

        for (auto gate = Gates::GateOperation::BEGIN;
             gate != Gates::GateOperation::END;
             gate = static_cast<Gates::GateOperation>(
                 static_cast<uint32_t>(gate) + 1)) {

            if (gate == Gates::GateOperation::Matrix) {
                continue;
            }

            const auto iter = std::find_if(
                gate_kernel_map_.at(gate).cbegin(),
                gate_kernel_map_.at(gate).cend(), [=](const auto &t) {
                    return (std::get<0>(t) == dispatch_key &&
                            std::get<1>(t)(num_qubits));
                });
            if (iter == gate_kernel_map_.at(gate).cend()) {
                throw std::range_error("Cannot find registered kernel for a "
                                       "dispatch key and number of qubits.");
            }
            kernel_for_gates.emplace(gate, std::get<2>(*iter));
        }
        return kernel_for_gates;
    }
};
} // namespace Pennylane
