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
 * Set/get Default kernels for statevector
 */
#include "DispatchKeys.hpp"
#include "GateOperation.hpp"
#include "IntegerInterval.hpp"
#include "KernelType.hpp"
#include "Util.hpp"

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Pennylane {

///@cond DEV
struct DispatchElement {
    uint32_t priority;
    Util::IntegerInterval<size_t> interval;
    Gates::KernelType kernel;
};

inline bool lower_priority(const DispatchElement &lhs,
                           const DispatchElement &rhs) {
    return lhs.priority < rhs.priority;
}

inline bool higher_priority(const DispatchElement &lhs,
                            const DispatchElement &rhs) {
    return lhs.priority > rhs.priority;
}

/**
 * @brief Maintain dispatch element using a vector decreasingly-ordered by
 * priority.
 */
class PriorityDispatchSet {
  private:
    std::vector<DispatchElement> ordered_vec_;

  public:
    [[nodiscard]] bool
    conflict(uint32_t test_priority,
             const Util::IntegerInterval<size_t> &test_interval) const {
        const auto test_elt = DispatchElement{test_priority, test_interval,
                                              Gates::KernelType::None};
        const auto [b, e] =
            std::equal_range(ordered_vec_.begin(), ordered_vec_.end(), test_elt,
                             higher_priority);
        for (auto iter = b; iter != e; ++iter) {
            if (!is_disjoint(iter->interval, test_interval)) {
                return true;
            }
        }
        return false;
    }

    void insert(const DispatchElement &elt) {
        const auto iter_to_insert = std::upper_bound(
            ordered_vec_.begin(), ordered_vec_.end(), elt, &higher_priority);
        ordered_vec_.insert(iter_to_insert, elt);
    }

    template <typename... Ts> void emplace(Ts &&...args) {
        const auto elt = DispatchElement{std::forward<Ts>(args)...};
        const auto iter_to_insert = std::upper_bound(
            ordered_vec_.begin(), ordered_vec_.end(), elt, &higher_priority);
        ordered_vec_.insert(iter_to_insert, elt);
    }

    [[nodiscard]] Gates::KernelType getKernel(size_t num_qubits) const {
        for (const auto &elt : ordered_vec_) {
            if (elt.interval(num_qubits)) {
                return elt.kernel;
            }
        }
        throw std::range_error(
            "Cannot find a kernel for the given number of qubits.");
    }

    void clearPriority(uint32_t remove_priority) {
        const auto begin = std::lower_bound(
            ordered_vec_.begin(), ordered_vec_.end(), remove_priority,
            [](const auto &elt, uint32_t p) { return elt.priority > p; });
        const auto end = std::upper_bound(
            ordered_vec_.begin(), ordered_vec_.end(), remove_priority,
            [](uint32_t p, const auto &elt) { return p > elt.priority; });
        ordered_vec_.erase(begin, end);
    }
};

///@endcond

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
        std::pair<Gates::GateOperation, uint32_t /* dispatch_key */>,
        PriorityDispatchSet, Util::PairHash>
        gate_kernel_map_;

    std::unordered_map<
        std::pair<Gates::GeneratorOperation, uint32_t /* dispatch_key */>,
        PriorityDispatchSet, Util::PairHash>
        generator_kernel_map_;

    std::unordered_map<
        std::pair<Gates::MatrixOperation, uint32_t /* dispatch_key */>,
        PriorityDispatchSet, Util::PairHash>
        matrix_kernel_map_;

    void registerDefaultGates() {
        using Gates::GateOperation;
        using Util::full_domain;
        using Util::in_between_closed;
        using Util::larger_than;
        using Util::larger_than_equal_to;
        using Util::less_than;
        using Util::less_than_equal_to;

        auto &instance = *this;
        auto all_qubit_numbers = full_domain<size_t>();
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
            all_memory_model, less_than<size_t>(12), Gates::KernelType::LM);
        instance.assignKernelForGate(
            GateOperation::IsingXX, all_threading, all_memory_model,
            // NOLINTNEXTLINE(readability-magic-numbers)
            in_between_closed<size_t>(12, 20), Gates::KernelType::PI);
        instance.assignKernelForGate(
            GateOperation::IsingXX, all_threading,
            // NOLINTNEXTLINE(readability-magic-numbers)
            all_memory_model, larger_than<size_t>(20), Gates::KernelType::LM);

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
        using Gates::GateOperation;
        using Gates::GeneratorOperation;
        using Gates::KernelType;
        using Util::full_domain;
        using Util::in_between_closed;
        using Util::larger_than;
        using Util::larger_than_equal_to;
        using Util::less_than;
        using Util::less_than_equal_to;

        auto &instance = *this;
        auto all_qubit_numbers = full_domain<size_t>();

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

    void registerDefaultMatrices() {
        using Gates::GateOperation;
        using Gates::KernelType;
        using Gates::MatrixOperation;
        using Util::full_domain;
        using Util::in_between_closed;
        using Util::larger_than;
        using Util::larger_than_equal_to;
        using Util::less_than;
        using Util::less_than_equal_to;

        auto &instance = *this;
        auto all_qubit_numbers = full_domain<size_t>();

        instance.assignKernelForMatrix(MatrixOperation::SingleQubitOp,
                                       all_threading, all_memory_model,
                                       all_qubit_numbers, KernelType::LM);
        instance.assignKernelForMatrix(MatrixOperation::TwoQubitOp,
                                       all_threading, all_memory_model,
                                       all_qubit_numbers, KernelType::LM);
        instance.assignKernelForMatrix(MatrixOperation::MultiQubitOp,
                                       all_threading, all_memory_model,
                                       all_qubit_numbers, KernelType::PI);
    }

    DefaultKernelsForStateVector() {
        registerDefaultGates();
        registerDefaultGenerators();
        registerDefaultMatrices();
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

    void assignKernelForGate(Gates::GateOperation gate_op, Threading threading,
                             CPUMemoryModel memory_model, uint32_t priority,
                             const Util::IntegerInterval<size_t> &interval,
                             Gates::KernelType kernel) {
        if (std::find(allowed_kernels.at(memory_model).cbegin(),
                      allowed_kernels.at(memory_model).cend(),
                      kernel) == allowed_kernels.at(memory_model).cend()) {
            throw std::invalid_argument("The given kernel is now allowed for "
                                        "the given memory model.");
        }
        const auto dispatch_key = toDispatchKey(threading, memory_model);
        auto &set = gate_kernel_map_[std::make_pair(gate_op, dispatch_key)];

        if (set.conflict(priority, interval)) {
            throw std::invalid_argument("The given interval conflicts with "
                                        "existing intervals.");
        }
        set.emplace(priority, interval, kernel);
    }

    void assignKernelForGate(Gates::GateOperation gate_op,
                             [[maybe_unused]] AllThreading dummy,
                             CPUMemoryModel memory_model,
                             const Util::IntegerInterval<size_t> &interval,
                             Gates::KernelType kernel) {
        /* Priority for all threading is 1 */
        Util::for_each_enum<Threading>([=](Threading threading) {
            assignKernelForGate(gate_op, threading, memory_model, 1, interval,
                                kernel);
        });
    }

    void assignKernelForGate(Gates::GateOperation gate_op, Threading threading,
                             [[maybe_unused]] AllMemoryModel dummy,
                             const Util::IntegerInterval<size_t> &interval,
                             Gates::KernelType kernel) {
        /* Priority for all memory model is 2 */
        Util::for_each_enum<CPUMemoryModel>([=](CPUMemoryModel memory_model) {
            assignKernelForGate(gate_op, threading, memory_model, 2, interval,
                                kernel);
        });
    }

    void assignKernelForGate(Gates::GateOperation gate_op,
                             [[maybe_unused]] AllThreading dummy1,
                             [[maybe_unused]] AllMemoryModel dummy2,
                             const Util::IntegerInterval<size_t> &interval,
                             Gates::KernelType kernel) {
        /* Priority is 0 */
        Util::for_each_enum<Threading, CPUMemoryModel>(
            [=](Threading threading, CPUMemoryModel memory_model) {
                assignKernelForGate(gate_op, threading, memory_model, 0,
                                    interval, kernel);
            });
    }

    void assignKernelForGenerator(Gates::GeneratorOperation gntr_op,
                                  Threading threading,
                                  CPUMemoryModel memory_model,
                                  uint32_t priority,
                                  const Util::IntegerInterval<size_t> &interval,
                                  Gates::KernelType kernel) {
        if (std::find(allowed_kernels.at(memory_model).cbegin(),
                      allowed_kernels.at(memory_model).cend(),
                      kernel) == allowed_kernels.at(memory_model).cend()) {
            throw std::invalid_argument("The given kernel is now allowed for "
                                        "the given memory model.");
        }
        const auto dispatch_key = toDispatchKey(threading, memory_model);
        auto &set =
            generator_kernel_map_[std::make_pair(gntr_op, dispatch_key)];

        if (set.conflict(priority, interval)) {
            throw std::invalid_argument("The given interval conflicts with "
                                        "existing intervals.");
        }
        set.emplace(priority, interval, kernel);
    }

    void assignKernelForGenerator(Gates::GeneratorOperation gntr_op,
                                  [[maybe_unused]] AllThreading dummy,
                                  CPUMemoryModel memory_model,
                                  const Util::IntegerInterval<size_t> &interval,
                                  Gates::KernelType kernel) {
        Util::for_each_enum<Threading>([=](Threading threading) {
            assignKernelForGenerator(gntr_op, threading, memory_model, 1,
                                     interval, kernel);
        });
    }

    void assignKernelForGenerator(Gates::GeneratorOperation gntr_op,
                                  Threading threading,
                                  [[maybe_unused]] AllMemoryModel dummy,
                                  const Util::IntegerInterval<size_t> &interval,
                                  Gates::KernelType kernel) {
        Util::for_each_enum<CPUMemoryModel>([=](CPUMemoryModel memory_model) {
            assignKernelForGenerator(gntr_op, threading, memory_model, 2,
                                     interval, kernel);
        });
    }

    void assignKernelForGenerator(Gates::GeneratorOperation gntr_op,
                                  [[maybe_unused]] AllThreading dummy1,
                                  [[maybe_unused]] AllMemoryModel dummy2,
                                  const Util::IntegerInterval<size_t> &interval,
                                  Gates::KernelType kernel) {
        Util::for_each_enum<Threading, CPUMemoryModel>(
            [=](Threading threading, CPUMemoryModel memory_model) {
                assignKernelForGenerator(gntr_op, threading, memory_model, 0,
                                         interval, kernel);
            });
    }

    void assignKernelForMatrix(Gates::MatrixOperation mat_op,
                               Threading threading, CPUMemoryModel memory_model,
                               uint32_t priority,
                               const Util::IntegerInterval<size_t> &interval,
                               Gates::KernelType kernel) {
        if (std::find(allowed_kernels.at(memory_model).cbegin(),
                      allowed_kernels.at(memory_model).cend(),
                      kernel) == allowed_kernels.at(memory_model).cend()) {
            throw std::invalid_argument("The given kernel is now allowed for "
                                        "the given memory model.");
        }
        const auto dispatch_key = toDispatchKey(threading, memory_model);
        auto &set = matrix_kernel_map_[std::make_pair(mat_op, dispatch_key)];

        if (set.conflict(priority, interval)) {
            throw std::invalid_argument("The given interval conflicts with "
                                        "existing intervals.");
        }
        set.emplace(priority, interval, kernel);
    }

    void assignKernelForMatrix(Gates::MatrixOperation mat_op,
                               [[maybe_unused]] AllThreading dummy,
                               CPUMemoryModel memory_model,
                               const Util::IntegerInterval<size_t> &interval,
                               Gates::KernelType kernel) {
        Util::for_each_enum<Threading>([=](Threading threading) {
            assignKernelForMatrix(mat_op, threading, memory_model, 1, interval,
                                  kernel);
        });
    }

    void assignKernelForMatrix(Gates::MatrixOperation mat_op,
                               Threading threading,
                               [[maybe_unused]] AllMemoryModel dummy,
                               const Util::IntegerInterval<size_t> &interval,
                               Gates::KernelType kernel) {
        Util::for_each_enum<CPUMemoryModel>([=](CPUMemoryModel memory_model) {
            assignKernelForMatrix(mat_op, threading, memory_model, 2, interval,
                                  kernel);
        });
    }

    void assignKernelForMatrix(Gates::MatrixOperation mat_op,
                               [[maybe_unused]] AllThreading dummy1,
                               [[maybe_unused]] AllMemoryModel dummy2,
                               const Util::IntegerInterval<size_t> &interval,
                               Gates::KernelType kernel) {
        Util::for_each_enum<Threading, CPUMemoryModel>(
            [=](Threading threading, CPUMemoryModel memory_model) {
                assignKernelForMatrix(mat_op, threading, memory_model, 0,
                                      interval, kernel);
            });
    }

    /**
     * @brief Create default kernels for all gates
     * @param num_qubits Number of qubits
     * @param threading Threading context
     * @param memory_model Memory model of the underlying data
     */
    auto getGateKernelMap(size_t num_qubits, Threading threading,
                          CPUMemoryModel memory_model) const
        -> std::unordered_map<Gates::GateOperation, Gates::KernelType> {
        uint32_t dispatch_key = toDispatchKey(threading, memory_model);

        std::unordered_map<Gates::GateOperation, Gates::KernelType>
            kernel_for_gates;

        Util::for_each_enum<Gates::GateOperation>(
            [&](Gates::GateOperation gate_op) {
                const auto key = std::make_pair(gate_op, dispatch_key);
                const auto &set = gate_kernel_map_.at(key);
                kernel_for_gates.emplace(gate_op, set.getKernel(num_qubits));
            });
        return kernel_for_gates;
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

        Util::for_each_enum<Gates::GeneratorOperation>(
            [&](Gates::GeneratorOperation gntr_op) {
                const auto key = std::make_pair(gntr_op, dispatch_key);
                const auto &set = generator_kernel_map_.at(key);
                kernel_for_generators.emplace(gntr_op,
                                              set.getKernel(num_qubits));
            });
        return kernel_for_generators;
    }

    /**
     * @brief Create default kernels for all matrix operations
     * @param num_qubits Number of qubits
     * @param threading Threading context
     * @param memory_model Memory model of the underlying data
     */
    auto getMatrixKernelMap(size_t num_qubits, Threading threading,
                            CPUMemoryModel memory_model) const
        -> std::unordered_map<Gates::MatrixOperation, Gates::KernelType> {
        uint32_t dispatch_key = toDispatchKey(threading, memory_model);

        std::unordered_map<Gates::MatrixOperation, Gates::KernelType>
            kernel_for_matrices;

        Util::for_each_enum<Gates::MatrixOperation>(
            [&](Gates::MatrixOperation mat_op) {
                const auto key = std::make_pair(mat_op, dispatch_key);
                const auto &set = matrix_kernel_map_.at(key);
                kernel_for_matrices.emplace(mat_op, set.getKernel(num_qubits));
            });
        return kernel_for_matrices;
    }

    void removeKernelForGenerator(Gates::GateOperation gate_op,
                                  Threading threading,
                                  CPUMemoryModel memory_model,
                                  uint32_t priority) {
        uint32_t dispatch_key = toDispatchKey(threading, memory_model);
        gate_kernel_map_[std::make_pair(gate_op, dispatch_key)].clearPriority(
            priority);
    }

    void removeKernelForMatrix(Gates::MatrixOperation mat_op,
                               Threading threading, CPUMemoryModel memory_model,
                               uint32_t priority) {
        uint32_t dispatch_key = toDispatchKey(threading, memory_model);
        matrix_kernel_map_[std::make_pair(mat_op, dispatch_key)].clearPriority(
            priority);
    }
};
} // namespace Pennylane
