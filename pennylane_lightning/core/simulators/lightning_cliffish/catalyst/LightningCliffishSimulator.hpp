// Copyright 2026 Xanadu Quantum Technologies Inc.

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
 * @file LightningCliffishSimulator.hpp
 */

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "Core.hpp"
#include "Observables.hpp"
#include "QuantumDevice.hpp"

// Catalyst releases from device ABI 1 carry the extension tables.
#if __has_include("DeviceExtensions.h")
#include "DeviceExtensions.h"
#define LIGHTNING_CLIFFISH_DEVICE_EXTENSIONS 1
#endif

namespace Catalyst::Runtime::Simulator {
/**
 * @brief Catalyst device for lightning.cliffish.
 *
 * The device adapter: the only code that knows Catalyst's device interface.
 * It turns each call into a call on the Lightning-Cliffish core, which
 * buffers ops and simulates them when a value or an output is demanded.
 * More info:
 * https://github.com/PennyLaneAI/catalyst/blob/main/runtime/include/QuantumDevice.hpp
 */
class LightningCliffishSimulator final
    : public Catalyst::Runtime::QuantumDevice {
  private:
    using CoreObservable = Pennylane::LightningCliffish::Planner::Observable;

    // static constants for RESULT values
    static constexpr bool GLOBAL_RESULT_TRUE_CONST = true;
    static constexpr bool GLOBAL_RESULT_FALSE_CONST = false;

    Pennylane::LightningCliffish::Core core{};

    /// Live qubits in allocation order. Outputs without wires act on these.
    std::vector<QubitIdType> qubits{};

    std::size_t device_shots{0};

    /// Observables by key, as sums of Pauli words.
    std::vector<CoreObservable> observables{};

    [[nodiscard]] auto observable(ObsIdType key) const
        -> const CoreObservable &;

    using ResultId = Pennylane::LightningCliffish::Program::ResultId;

    /// Delegated feed-forward: the predicate of each open scope, each the
    /// conjunction of all scopes up to it.
    std::vector<ResultId> predicates{};
    /// One stable address per result handed out as a handle.
    std::deque<bool> handle_storage{};
    std::unordered_map<const bool *, ResultId> handle_results{};

    /// The handle for `result`. Its bit is unspecified.
    auto handle(ResultId result) -> Result;
    [[nodiscard]] auto resultOf(Result handle) const -> ResultId;
    /// The predicate new ops carry.
    [[nodiscard]] auto predicate() const -> ResultId;
    /// A measurement: a handle on a delegated program, else a value now.
    auto measurement(Pennylane::LightningCliffish::Program::Op op) -> Result;

#ifdef LIGHTNING_CLIFFISH_DEVICE_EXTENSIONS
    CatalystDelegatedTable delegated_table{};
    CatalystGadgetTable gadget_table{};
#endif

    void sampleInto(DataView<double, 2> &samples,
                    const std::vector<QubitIdType> &wires);
    void countsInto(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                    const std::vector<QubitIdType> &wires);
    void probsInto(DataView<double, 1> &probs,
                   const std::vector<QubitIdType> &wires);

  public:
    /**
     * @param kwargs The device's keyword arguments: max_dense_qubits,
     * recompression ("auto", "none", "t1" or "t2"), dense_checks and
     * gadget_collapse ("measured", "source" or "off").
     */
    explicit LightningCliffishSimulator(const std::string &kwargs = "{}");
    ~LightningCliffishSimulator() noexcept = default;

    LightningCliffishSimulator(const LightningCliffishSimulator &) = delete;
    LightningCliffishSimulator &
    operator=(const LightningCliffishSimulator &) = delete;
    LightningCliffishSimulator(LightningCliffishSimulator &&) = delete;
    LightningCliffishSimulator &
    operator=(LightningCliffishSimulator &&) = delete;

    auto AllocateQubit() -> QubitIdType override;
    auto AllocateQubits(std::size_t num_qubits)
        -> std::vector<QubitIdType> override;
    void ReleaseQubit(QubitIdType q) override;
    void ReleaseQubits(const std::vector<QubitIdType> &ids) override;
    [[nodiscard]] auto GetNumQubits() const -> std::size_t override;
    void SetDeviceShots(std::size_t shots) override;
    [[nodiscard]] auto GetDeviceShots() const -> std::size_t override;
    void SetDevicePRNG(std::mt19937 *gen) override;

    void NamedOperation(
        const std::string &name, const std::vector<double> &params,
        const std::vector<QubitIdType> &wires, bool inverse = false,
        const std::vector<QubitIdType> &controlled_wires = {},
        const std::vector<bool> &controlled_values = {},
        const std::vector<std::string> &optional_params = {}) override;

    auto Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                    const std::vector<QubitIdType> &wires)
        -> ObsIdType override;
    auto TensorObservable(const std::vector<ObsIdType> &obs)
        -> ObsIdType override;
    auto HamiltonianObservable(const std::vector<double> &coeffs,
                               const std::vector<ObsIdType> &obs)
        -> ObsIdType override;
    auto Expval(ObsIdType obsKey) -> double override;
    auto Var(ObsIdType obsKey) -> double override;
    void State(DataView<std::complex<double>, 1> &state) override;
    void Probs(DataView<double, 1> &probs) override;
    void PartialProbs(DataView<double, 1> &probs,
                      const std::vector<QubitIdType> &wires) override;
    void Sample(DataView<double, 2> &samples) override;
    void PartialSample(DataView<double, 2> &samples,
                       const std::vector<QubitIdType> &wires) override;
    void Counts(DataView<double, 1> &eigvals,
                DataView<int64_t, 1> &counts) override;
    void PartialCounts(DataView<double, 1> &eigvals,
                       DataView<int64_t, 1> &counts,
                       const std::vector<QubitIdType> &wires) override;
    auto Measure(QubitIdType wire,
                 std::optional<int32_t> postselect = std::nullopt)
        -> Result override;
    auto PauliMeasure(const std::string &pauli_word,
                      const std::vector<QubitIdType> &wires) -> Result override;

    // Delegated feed-forward and gadget brackets, behind the "Delegated" and
    // "Gadget" extension tables.
    void BeginDelegatedProgram(uint32_t postselect_mode);
    void PushPredicate(Result condition);
    void PopPredicate();
    auto ResultConstant(bool value) -> Result;
    auto ResultXor(Result a, Result b) -> Result;
    auto ResultNot(Result a) -> Result;
    auto ResultSelect(Result condition, Result if_true, Result if_false)
        -> Result;
    void ReleaseResult(Result result);
    auto SignedPauliMeasure(const std::string &pauli_word,
                            const std::vector<QubitIdType> &wires, bool negated,
                            std::optional<int32_t> postselect) -> Result;
    void BeginGadget(const std::string &pauli_word,
                     const std::vector<QubitIdType> &wires, double theta,
                     uint32_t verdict);
    void EndGadget();

    /// What gadget collapse did on this device so far.
    [[nodiscard]] auto gadgetReport() const
        -> const Pennylane::LightningCliffish::Planner::GadgetReport & {
        return core.gadgetReport();
    }

#ifdef LIGHTNING_CLIFFISH_DEVICE_EXTENSIONS
    auto QueryInterface(const char *name, uint32_t min_version) -> const
        void * override;
#endif
};

} // namespace Catalyst::Runtime::Simulator
