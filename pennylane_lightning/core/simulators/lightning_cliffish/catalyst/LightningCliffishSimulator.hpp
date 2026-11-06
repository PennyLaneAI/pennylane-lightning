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
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "Core.hpp"
#include "Observables.hpp"
#include "QuantumDevice.hpp"

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

    void sampleInto(DataView<double, 2> &samples,
                    const std::vector<QubitIdType> &wires);
    void countsInto(DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
                    const std::vector<QubitIdType> &wires);
    void probsInto(DataView<double, 1> &probs,
                   const std::vector<QubitIdType> &wires);

  public:
    /**
     * @param kwargs The device's keyword arguments: max_dense_qubits,
     * recompression ("auto", "none", "t1" or "t2") and dense_checks.
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
};

} // namespace Catalyst::Runtime::Simulator
