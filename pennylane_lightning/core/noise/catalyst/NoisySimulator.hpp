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
 * @file NoisySimulator.hpp
 * Generic Catalyst runtime device for the Lightning backends that can
 * represent mixed states (lightning.tensornet).
 *
 * The class implements the whole Catalyst::Runtime::QuantumDevice interface
 * on top of a duck-typed state class `StateT` and applies a native noise
 * model (see NoiseModel.hpp) to every gate and terminal measurement. The
 * noise model is handed over by the Python layer through the device kwargs
 * (key `noise_model`), so it also works for programs compiled with qjit.
 *
 * Required StateT API (device wire indices, wire 0 = most significant bit):
 *   StateT(const StateT &);                         // deep copy
 *   std::size_t getNumQubits() const;
 *   void applyOperation(name, ctrls, ctrl_values, wires, inverse, params);
 *   void applyPauliRot(wires, inverse, params, word);
 *   void applyMatrix(const complex<double> *, size, wires, inverse);
 *   void applyControlledMatrix(ptr, size, ctrls, ctrl_values, wires, inverse);
 *   void applyChannel(const KrausChannel<double> &, wires);
 *   std::vector<double> probs(wires) const;
 *   double expval(const ObsTerms<double> &) const;
 *   double var(const ObsTerms<double> &) const;
 *   std::vector<std::size_t> sampleWires(wires, shots, rng) const;
 *   void collapse(std::size_t wire, bool branch);
 *   void setBasisState(std::vector<std::size_t>, wires);
 *   void setPureState(const complex<double> *, length, wires);
 *   void appendQubit();                       // new last wire in |0>
 *   void traceOut(wires);                     // partial trace
 *   void resetQubit(std::size_t wire);        // reset channel to |0>
 *   static constexpr bool hasStateVector();
 *   void getStateVector(complex<double> *, length) const; // if above true
 */
#pragma once

#include <algorithm>
#include <bitset>
#include <climits>
#include <complex>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Exception.hpp"
#include "LightningQubitManager.hpp"
#include "QuantumDevice.hpp"
#include "Utils.hpp"

#include "GateMatrices.hpp"
#include "HermitianEigen.hpp"
#include "NoiseModel.hpp"
#include "ObservableTerms.hpp"

namespace Catalyst::Runtime::Simulator {

template <class StateT>
class NoisySimulator : public Catalyst::Runtime::QuantumDevice {
  public:
    using CVecD = Pennylane::LightningNoisy::CVec<double>;
    using ObsTermsD = Pennylane::LightningNoisy::ObsTerms<double>;
    using NoiseModelD = Pennylane::LightningNoisy::NoiseModel<double>;
    using ResolvedNoise = Pennylane::LightningNoisy::ResolvedNoise;

    explicit NoisySimulator(const std::string &kwargs = "{}") {
        options_ = Catalyst::Runtime::parse_kwargs(kwargs);
        if (auto it = options_.find("noise_model"); it != options_.end()) {
            try {
                noise_ = NoiseModelD(it->second);
            } catch (const std::exception &e) {
                RT_FAIL(
                    ("Invalid noise model: " + std::string(e.what())).c_str());
            }
        }
    }
    ~NoisySimulator() override = default;

    NoisySimulator(const NoisySimulator &) = delete;
    NoisySimulator &operator=(const NoisySimulator &) = delete;
    NoisySimulator(NoisySimulator &&) = delete;
    NoisySimulator &operator=(NoisySimulator &&) = delete;

    // ------------------------------------------------------------------
    // Accessors used by the tests and the concrete devices
    // ------------------------------------------------------------------
    [[nodiscard]] const NoiseModelD &getNoiseModel() const { return noise_; }
    [[nodiscard]] const std::unordered_map<std::string, std::string> &
    getOptions() const {
        return options_;
    }
    /// Device-side state after removing released qubits
    [[nodiscard]] StateT &getState() {
        reduce();
        return *state_;
    }

    // ------------------------------------------------------------------
    // Qubit management
    // ------------------------------------------------------------------

    auto AllocateQubit() -> QubitIdType override {
        if (state_ == nullptr || state_->getNumQubits() == 0U) {
            state_ = makeState(1);
            qubit_manager_.ReleaseAll();
            needs_reduction_ = false;
            auto ids = qubit_manager_.AllocateRange(0, 1);
            register_base_ = ids[0];
            return ids[0];
        }
        if (auto cand = qubit_manager_.popFreeQubit(); cand.has_value()) {
            // A released qubit is reused: bring it back to |0>. For mixed
            // states this is exact even if the qubit was entangled.
            state_->resetQubit(cand.value());
            return qubit_manager_.Allocate(cand.value());
        }
        const std::size_t dev = state_->getNumQubits();
        state_->appendQubit();
        return qubit_manager_.Allocate(dev);
    }

    auto AllocateQubits(std::size_t num_qubits)
        -> std::vector<QubitIdType> override {
        if (num_qubits == 0U) {
            return {};
        }
        if (GetNumQubits() == 0U) {
            state_ = makeState(num_qubits);
            qubit_manager_.ReleaseAll();
            needs_reduction_ = false;
            auto ids = qubit_manager_.AllocateRange(0, num_qubits);
            register_base_ = ids[0];
            return ids;
        }
        std::vector<QubitIdType> ids(num_qubits);
        std::generate_n(ids.begin(), num_qubits,
                        [this]() { return AllocateQubit(); });
        return ids;
    }

    void ReleaseQubit(QubitIdType q) override {
        RT_FAIL_IF(!qubit_manager_.isValidQubitId(q),
                   "Invalid qubit to release");
        qubit_manager_.Release(q);
        needs_reduction_ = true;
    }

    void ReleaseQubits(const std::vector<QubitIdType> &ids) override {
        if (!ids.empty() && GetNumQubits() == ids.size() &&
            qubit_manager_.isValidQubitId(ids)) {
            qubit_manager_.ReleaseAll();
            state_ = makeState(0);
            needs_reduction_ = false;
            return;
        }
        for (auto id : ids) {
            ReleaseQubit(id);
        }
    }

    [[nodiscard]] auto GetNumQubits() const -> std::size_t override {
        return qubit_manager_.getNumQubits();
    }

    void SetDeviceShots(std::size_t shots) override { device_shots_ = shots; }
    [[nodiscard]] auto GetDeviceShots() const -> std::size_t override {
        return device_shots_;
    }
    void SetDevicePRNG(std::mt19937 *gen) override { gen_ = gen; }

    // ------------------------------------------------------------------
    // State preparation
    // ------------------------------------------------------------------

    void SetState(DataView<std::complex<double>, 1> &data,
                  std::vector<QubitIdType> &wires) override {
        reduce();
        RT_FAIL_IF(!validQubits(wires), "Invalid wires for SetState");
        RT_FAIL_IF(data.size() != (std::size_t{1} << wires.size()),
                   "State size does not match the number of wires");
        std::vector<std::complex<double>> psi(data.begin(), data.end());
        state_->setPureState(psi.data(), psi.size(), deviceWires(wires));
    }

    void SetBasisState(DataView<int8_t, 1> &data,
                       std::vector<QubitIdType> &wires) override {
        reduce();
        RT_FAIL_IF(!validQubits(wires), "Invalid wires for SetBasisState");
        std::vector<std::size_t> bits(data.begin(), data.end());
        state_->setBasisState(bits, deviceWires(wires));
    }

    // ------------------------------------------------------------------
    // Gates
    // ------------------------------------------------------------------

    void
    NamedOperation(const std::string &name, const std::vector<double> &params,
                   const std::vector<QubitIdType> &wires, bool inverse,
                   const std::vector<QubitIdType> &controlled_wires,
                   const std::vector<bool> &controlled_values,
                   const std::vector<std::string> &optional_params) override {
        RT_FAIL_IF(controlled_wires.size() != controlled_values.size(),
                   "Controlled wires/values size mismatch");
        RT_FAIL_IF(!validQubits(wires), "Given wires do not refer to qubits");
        RT_FAIL_IF(!validQubits(controlled_wires),
                   "Given controlled wires do not refer to qubits");
        reduce();

        std::string key = inverse ? "Adjoint(" + name + ")" : name;
        if (!controlled_wires.empty()) {
            key = "C(" + key + ")";
        }
        const auto noisy = lookupGateNoise(key, controlled_wires, wires);
        applyNoise(before_);

        const auto dev_wires = deviceWires(wires);
        const auto dev_ctrls = deviceWires(controlled_wires);
        if (name == "PauliRot") {
            RT_FAIL_IF(optional_params.size() != 1,
                       "PauliRot operation requires one string parameter "
                       "for the Pauli word");
            RT_FAIL_IF(!controlled_wires.empty(),
                       "Controlled PauliRot is not supported");
            applyPauliRotRaw(optional_params[0], dev_wires, params, inverse);
        } else {
            const std::size_t nch = channelArity(name);
            if (nch != 0U) {
                RT_FAIL_IF(!controlled_wires.empty(),
                           "Controlled noise channels are not supported");
                applyNamedChannel(name, params, dev_wires);
            } else {
                state_->applyOperation(name, dev_ctrls, controlled_values,
                                       dev_wires, inverse, params);
            }
        }
        if (noisy) {
            applyNoise(after_);
        }
    }

    void MatrixOperation(const std::vector<std::complex<double>> &matrix,
                         const std::vector<QubitIdType> &wires, bool inverse,
                         const std::vector<QubitIdType> &controlled_wires,
                         const std::vector<bool> &controlled_values) override {
        RT_FAIL_IF(controlled_wires.size() != controlled_values.size(),
                   "Controlled wires/values size mismatch");
        RT_FAIL_IF(!validQubits(wires), "Given wires do not refer to qubits");
        RT_FAIL_IF(!validQubits(controlled_wires),
                   "Given controlled wires do not refer to qubits");
        reduce();
        std::string key = inverse ? "Adjoint(QubitUnitary)" : "QubitUnitary";
        if (!controlled_wires.empty()) {
            key = "C(" + key + ")";
        }
        const auto noisy = lookupGateNoise(key, controlled_wires, wires);
        applyNoise(before_);
        state_->applyControlledMatrix(
            matrix.data(), matrix.size(), deviceWires(controlled_wires),
            controlled_values, deviceWires(wires), inverse);
        if (noisy) {
            applyNoise(after_);
        }
    }

    // ------------------------------------------------------------------
    // Observables
    // ------------------------------------------------------------------

    auto Observable(ObsId id, const std::vector<std::complex<double>> &matrix,
                    const std::vector<QubitIdType> &wires)
        -> ObsIdType override {
        RT_FAIL_IF(!validQubits(wires), "Invalid given wires");
        RT_FAIL_IF(wires.size() > GetNumQubits(), "Invalid number of wires");
        ObsNode node;
        node.kind = ObsNode::Kind::Leaf;
        node.id = id;
        node.wires = wires;
        if (id == ObsId::Hermitian) {
            const std::size_t d = std::size_t{1} << wires.size();
            RT_FAIL_IF(matrix.size() != d * d,
                       "Hermitian matrix size does not match the wires");
            node.matrix = CVecD(matrix.begin(), matrix.end());
        } else {
            RT_FAIL_IF(wires.size() != 1U,
                       "Named observables act on a single wire");
        }
        observables_.push_back(std::move(node));
        return static_cast<ObsIdType>(observables_.size() - 1);
    }

    auto TensorObservable(const std::vector<ObsIdType> &obs)
        -> ObsIdType override {
        RT_FAIL_IF(obs.empty(), "Empty tensor product observable");
        checkObsKeys(obs);
        for (auto k : obs) {
            RT_FAIL_IF(observables_[static_cast<std::size_t>(k)].kind ==
                           ObsNode::Kind::Hamiltonian,
                       "A tensor product cannot contain a Hamiltonian");
        }
        ObsNode node;
        node.kind = ObsNode::Kind::Tensor;
        node.children = obs;
        observables_.push_back(std::move(node));
        return static_cast<ObsIdType>(observables_.size() - 1);
    }

    auto HamiltonianObservable(const std::vector<double> &coeffs,
                               const std::vector<ObsIdType> &obs)
        -> ObsIdType override {
        RT_FAIL_IF(coeffs.size() != obs.size(),
                   "Hamiltonian coefficients and observables size mismatch");
        checkObsKeys(obs);
        ObsNode node;
        node.kind = ObsNode::Kind::Hamiltonian;
        node.children = obs;
        node.coeffs = coeffs;
        observables_.push_back(std::move(node));
        return static_cast<ObsIdType>(observables_.size() - 1);
    }

    // ------------------------------------------------------------------
    // Measurements
    // ------------------------------------------------------------------

    auto Expval(ObsIdType key) -> double override {
        checkObsKeys({key});
        reduce();
        const auto &st = measuredState("expval", observableWires(key));
        return device_shots_ ? shotExpval(st, key).first
                             : st.expval(toTerms(key));
    }

    auto Var(ObsIdType key) -> double override {
        checkObsKeys({key});
        reduce();
        const auto &st = measuredState("var", observableWires(key));
        return device_shots_ ? shotExpval(st, key).second
                             : st.var(toTerms(key));
    }

    void State(DataView<std::complex<double>, 1> &state) override {
        if constexpr (StateT::hasStateVector()) {
            reduce();
            const std::size_t n = state_->getNumQubits();
            RT_FAIL_IF(state.size() != (std::size_t{1} << n),
                       "Invalid size for the pre-allocated state vector");
            std::vector<std::complex<double>> buf(state.size());
            state_->getStateVector(buf.data(), buf.size());
            std::move(buf.begin(), buf.end(), state.begin());
        } else {
            (void)state;
            RT_FAIL("The state of a mixed-state device is a density matrix; "
                    "qml.state() is not supported under qjit. Use "
                    "qml.density_matrix without qjit, or probs/expval.");
        }
    }

    void Probs(DataView<double, 1> &probs) override {
        PartialProbs(probs, allProgramWires());
    }

    void PartialProbs(DataView<double, 1> &probs,
                      const std::vector<QubitIdType> &wires) override {
        RT_FAIL_IF(wires.size() > GetNumQubits(), "Invalid number of wires");
        RT_FAIL_IF(!validQubits(wires), "Invalid given wires to measure");
        reduce();
        const auto &st = measuredState("probs", wires);
        const auto dev = deviceWires(wires);
        std::vector<double> p;
        if (device_shots_ == 0U) {
            p = st.probs(dev);
        } else {
            p.assign(std::size_t{1} << dev.size(), 0.0);
            const auto bits = st.sampleWires(dev, device_shots_, rng());
            const std::size_t k = dev.size();
            for (std::size_t s = 0; s < device_shots_; s++) {
                p[bitsToIndex(bits, s, k)] += 1.0;
            }
            for (auto &x : p) {
                x /= static_cast<double>(device_shots_);
            }
        }
        RT_FAIL_IF(probs.size() != p.size(),
                   "Invalid size for the pre-allocated probabilities");
        std::move(p.begin(), p.end(), probs.begin());
    }

    void Sample(DataView<double, 2> &samples) override {
        PartialSample(samples, allProgramWires());
    }

    void PartialSample(DataView<double, 2> &samples,
                       const std::vector<QubitIdType> &wires) override {
        RT_FAIL_IF(wires.size() > GetNumQubits(), "Invalid number of wires");
        RT_FAIL_IF(!validQubits(wires), "Invalid given wires to measure");
        RT_FAIL_IF(samples.size() != device_shots_ * wires.size(),
                   "Invalid size for the pre-allocated samples");
        reduce();
        const auto &st = measuredState("sample", wires);
        const auto bits =
            st.sampleWires(deviceWires(wires), device_shots_, rng());
        auto it = samples.begin();
        for (auto b : bits) {
            *(it++) = static_cast<double>(b);
        }
    }

    void Counts(DataView<double, 1> &eigvals,
                DataView<int64_t, 1> &counts) override {
        PartialCounts(eigvals, counts, allProgramWires());
    }

    void PartialCounts(DataView<double, 1> &eigvals,
                       DataView<int64_t, 1> &counts,
                       const std::vector<QubitIdType> &wires) override {
        const std::size_t k = wires.size();
        const std::size_t nel = std::size_t{1} << k;
        RT_FAIL_IF(k > GetNumQubits(), "Invalid number of wires");
        RT_FAIL_IF(!validQubits(wires), "Invalid given wires to measure");
        RT_FAIL_IF(eigvals.size() != nel || counts.size() != nel,
                   "Invalid size for the pre-allocated counts");
        reduce();
        const auto &st = measuredState("counts", wires);
        const auto bits =
            st.sampleWires(deviceWires(wires), device_shots_, rng());
        std::iota(eigvals.begin(), eigvals.end(), 0);
        std::fill(counts.begin(), counts.end(), 0);
        for (std::size_t s = 0; s < device_shots_; s++) {
            counts(bitsToIndex(bits, s, k)) += 1;
        }
    }

    auto Measure(QubitIdType wire, std::optional<int32_t> postselect)
        -> Result override {
        RT_FAIL_IF(!qubit_manager_.isValidQubitId(wire),
                   "Invalid wire to measure");
        reduce();
        const std::size_t dev = qubit_manager_.getDeviceId(wire);
        const bool mres = measureDevice(dev, postselect);
        return mres ? const_cast<Result>(&RESULT_TRUE)
                    : const_cast<Result>(&RESULT_FALSE);
    }

    /**
     * Projective measurement of a Pauli word. The word is rotated to Z..Z,
     * the parity is accumulated on the first non-identity wire with CNOTs,
     * only that wire is measured, and the rotations are undone. Result
     * `true` means eigenvalue -1. The internal gates are noiseless.
     */
    auto PauliMeasure(const std::string &pauli_word,
                      const std::vector<QubitIdType> &wires)
        -> Result override {
        RT_FAIL_IF(pauli_word.size() != wires.size(),
                   "Pauli word length must match number of wires");
        RT_FAIL_IF(!validQubits(wires), "Invalid wires for PauliMeasure");
        reduce();
        std::vector<std::size_t> act;
        std::string word;
        const auto dev = deviceWires(wires);
        for (std::size_t i = 0; i < pauli_word.size(); i++) {
            const char p = pauli_word[i];
            RT_FAIL_IF(p != 'I' && p != 'X' && p != 'Y' && p != 'Z',
                       "Invalid character in Pauli word");
            if (p != 'I') {
                act.push_back(dev[i]);
                word.push_back(p);
            }
        }
        if (act.empty()) {
            return const_cast<Result>(&RESULT_FALSE);
        }
        rotateToZ(word, act, false);
        for (std::size_t i = 1; i < act.size(); i++) {
            state_->applyOperation("CNOT", {}, {}, {act[i], act[0]}, false, {});
        }
        const bool mres = measureDevice(act[0], std::nullopt);
        for (std::size_t i = act.size() - 1; i >= 1; i--) {
            state_->applyOperation("CNOT", {}, {}, {act[i], act[0]}, false, {});
        }
        rotateToZ(word, act, true);
        return mres ? const_cast<Result>(&RESULT_TRUE)
                    : const_cast<Result>(&RESULT_FALSE);
    }

  protected:
    /// Create a state of `n` qubits in |0..0>. Backends may override it to
    /// forward device options (bond dimension, ...).
    [[nodiscard]] virtual std::unique_ptr<StateT>
    makeState(std::size_t n) const {
        return std::make_unique<StateT>(n);
    }

  private:
    struct ObsNode {
        enum class Kind { Leaf, Tensor, Hamiltonian };
        Kind kind{Kind::Leaf};
        ObsId id{ObsId::Identity};
        CVecD matrix{};
        std::vector<QubitIdType> wires{};
        std::vector<ObsIdType> children{};
        std::vector<double> coeffs{};
    };

    static constexpr bool RESULT_TRUE = true;
    static constexpr bool RESULT_FALSE = false;

    std::unordered_map<std::string, std::string> options_{};
    NoiseModelD noise_{};
    QubitManager<QubitIdType, std::size_t> qubit_manager_{};
    QubitIdType register_base_{0};
    std::unique_ptr<StateT> state_{};
    bool needs_reduction_{false};
    std::size_t device_shots_{0};
    std::mt19937 *gen_{nullptr};
    std::mt19937 own_gen_{std::random_device{}()};
    std::vector<ObsNode> observables_{};
    std::vector<ResolvedNoise> before_{};
    std::vector<ResolvedNoise> after_{};
    std::vector<ResolvedNoise> readout_{};
    std::unique_ptr<StateT> scratch_{};

    std::mt19937 &rng() { return gen_ != nullptr ? *gen_ : own_gen_; }

    // ---------------- wires ----------------

    [[nodiscard]] bool validQubits(const std::vector<QubitIdType> &w) const {
        return std::all_of(w.begin(), w.end(), [this](QubitIdType q) {
            return qubit_manager_.isValidQubitId(q);
        });
    }

    [[nodiscard]] std::vector<std::size_t>
    deviceWires(const std::vector<QubitIdType> &w) const {
        std::vector<std::size_t> out;
        out.reserve(w.size());
        for (auto q : w) {
            out.push_back(qubit_manager_.getDeviceId(q));
        }
        return out;
    }

    /// Program qubit -> wire label used by the noise model (index in the
    /// register created by the first allocation of an execution).
    [[nodiscard]] std::size_t label(QubitIdType q) const {
        RT_FAIL_IF(q < register_base_, "Invalid qubit id");
        return static_cast<std::size_t>(q - register_base_);
    }

    [[nodiscard]] std::size_t labelToDevice(std::size_t lbl) const {
        const auto q = register_base_ + static_cast<QubitIdType>(lbl);
        RT_FAIL_IF(!qubit_manager_.isValidQubitId(q),
                   "Noise model refers to a wire that is not allocated");
        return qubit_manager_.getDeviceId(q);
    }

    [[nodiscard]] std::vector<QubitIdType> allProgramWires() {
        auto ids = qubit_manager_.getAllQubitIds();
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    static std::size_t bitsToIndex(const std::vector<std::size_t> &bits,
                                   std::size_t shot, std::size_t k) {
        std::size_t idx = 0;
        for (std::size_t j = 0; j < k; j++) {
            idx = (idx << 1U) | bits[shot * k + j];
        }
        return idx;
    }

    /// Remove released qubits from the state (partial trace) and compact
    /// the device indices.
    void reduce() {
        RT_FAIL_IF(state_ == nullptr, "No qubits have been allocated");
        if (!needs_reduction_) {
            return;
        }
        const auto &freed = qubit_manager_.getFreeDeviceQubits();
        std::vector<std::size_t> drop(freed.begin(), freed.end());
        std::sort(drop.begin(), drop.end());
        if (!drop.empty()) {
            state_->traceOut(drop);
            std::unordered_map<std::size_t, std::size_t> remap;
            for (auto q : qubit_manager_.getAllQubitIds()) {
                const std::size_t d = qubit_manager_.getDeviceId(q);
                const auto below = static_cast<std::size_t>(std::distance(
                    drop.begin(),
                    std::lower_bound(drop.begin(), drop.end(), d)));
                if (below != 0U) {
                    remap[d] = d - below;
                }
            }
            if (!remap.empty()) {
                qubit_manager_.RemapDeviceIds(remap);
            }
            qubit_manager_.ClearFreeQubits();
        }
        needs_reduction_ = false;
    }

    // ---------------- noise ----------------

    /// Fill before_/after_ for a gate; returns true if any noise applies.
    bool lookupGateNoise(const std::string &key,
                         const std::vector<QubitIdType> &ctrls,
                         const std::vector<QubitIdType> &wires) {
        before_.clear();
        after_.clear();
        if (noise_.empty()) {
            return false;
        }
        std::vector<std::size_t> labels;
        labels.reserve(ctrls.size() + wires.size());
        for (auto q : ctrls) {
            labels.push_back(label(q));
        }
        for (auto q : wires) {
            labels.push_back(label(q));
        }
        noise_.lookupGate(key, labels, before_, after_);
        return !after_.empty();
    }

    void applyNoise(const std::vector<ResolvedNoise> &noise) {
        applyNoiseTo(*state_, noise);
    }

    void applyNoiseTo(StateT &st, const std::vector<ResolvedNoise> &noise) {
        for (const auto &n : noise) {
            std::vector<std::size_t> dev;
            dev.reserve(n.wires.size());
            for (auto l : n.wires) {
                dev.push_back(labelToDevice(l));
            }
            st.applyChannel(noise_.channel(n.channel), dev);
        }
    }

    /// State to measure: the device state, or a noisy copy if readout noise
    /// applies to this measurement.
    const StateT &measuredState(const std::string &kind,
                                const std::vector<QubitIdType> &wires) {
        if (!noise_.hasReadoutNoise()) {
            return *state_;
        }
        std::vector<std::size_t> labels;
        labels.reserve(wires.size());
        for (auto q : wires) {
            labels.push_back(label(q));
        }
        noise_.lookupReadout(kind, labels, readout_);
        if (readout_.empty()) {
            return *state_;
        }
        scratch_ = std::make_unique<StateT>(*state_);
        applyNoiseTo(*scratch_, readout_);
        return *scratch_;
    }

    // ---------------- channels given as operations ----------------

    /// Number of wires of a channel given as a named operation (0: not a
    /// channel)
    static std::size_t channelArity(const std::string &name) {
        static const std::unordered_set<std::string> channels{
            "AmplitudeDamping", "PhaseDamping", "DepolarizingChannel",
            "BitFlip",          "PhaseFlip",    "GeneralizedAmplitudeDamping",
            "ResetError"};
        return channels.contains(name) ? 1U : 0U;
    }

    void applyNamedChannel(const std::string &name,
                           const std::vector<double> &params,
                           const std::vector<std::size_t> &dev_wires) {
        using Pennylane::LightningNoisy::KrausChannel;
        RT_FAIL_IF(dev_wires.size() != 1U,
                   "Only single-qubit named channels are supported");
        KrausChannel<double> ch;
        ch.num_qubits = 1;
        ch.kraus = namedChannelKraus(name, params);
        ch.finalize();
        state_->applyChannel(ch, dev_wires);
    }

    static std::vector<CVecD> namedChannelKraus(const std::string &name,
                                                const std::vector<double> &p) {
        using C = std::complex<double>;
        auto need = [&](std::size_t n) {
            RT_FAIL_IF(p.size() != n, "Wrong number of channel parameters");
        };
        auto scaled = [](const CVecD &m, double s) {
            CVecD out(m);
            for (auto &x : out) {
                x *= s;
            }
            return out;
        };
        const CVecD I{C{1, 0}, C{0, 0}, C{0, 0}, C{1, 0}};
        const CVecD X{C{0, 0}, C{1, 0}, C{1, 0}, C{0, 0}};
        const CVecD Y{C{0, 0}, C{0, -1}, C{0, 1}, C{0, 0}};
        const CVecD Z{C{1, 0}, C{0, 0}, C{0, 0}, C{-1, 0}};
        if (name == "AmplitudeDamping") {
            need(1);
            const double g = p[0];
            return {{C{1, 0}, C{0, 0}, C{0, 0}, C{std::sqrt(1 - g), 0}},
                    {C{0, 0}, C{std::sqrt(g), 0}, C{0, 0}, C{0, 0}}};
        }
        if (name == "PhaseDamping") {
            need(1);
            const double g = p[0];
            return {{C{1, 0}, C{0, 0}, C{0, 0}, C{std::sqrt(1 - g), 0}},
                    {C{0, 0}, C{0, 0}, C{0, 0}, C{std::sqrt(g), 0}}};
        }
        if (name == "DepolarizingChannel") {
            need(1);
            const double q = p[0];
            return {scaled(I, std::sqrt(1 - q)), scaled(X, std::sqrt(q / 3)),
                    scaled(Y, std::sqrt(q / 3)), scaled(Z, std::sqrt(q / 3))};
        }
        if (name == "BitFlip") {
            need(1);
            return {scaled(I, std::sqrt(1 - p[0])), scaled(X, std::sqrt(p[0]))};
        }
        if (name == "PhaseFlip") {
            need(1);
            return {scaled(I, std::sqrt(1 - p[0])), scaled(Z, std::sqrt(p[0]))};
        }
        if (name == "GeneralizedAmplitudeDamping") {
            need(2);
            const double g = p[0];
            const double q = p[1];
            const double a = std::sqrt(q);
            const double b = std::sqrt(1 - q);
            return {{C{a, 0}, C{0, 0}, C{0, 0}, C{a * std::sqrt(1 - g), 0}},
                    {C{0, 0}, C{a * std::sqrt(g), 0}, C{0, 0}, C{0, 0}},
                    {C{b * std::sqrt(1 - g), 0}, C{0, 0}, C{0, 0}, C{b, 0}},
                    {C{0, 0}, C{0, 0}, C{b * std::sqrt(g), 0}, C{0, 0}}};
        }
        if (name == "ResetError") {
            need(2);
            const double p0 = p[0];
            const double p1 = p[1];
            return {scaled(I, std::sqrt(1 - p0 - p1)),
                    {C{std::sqrt(p0), 0}, C{0, 0}, C{0, 0}, C{0, 0}},
                    {C{0, 0}, C{std::sqrt(p0), 0}, C{0, 0}, C{0, 0}},
                    {C{0, 0}, C{0, 0}, C{std::sqrt(p1), 0}, C{0, 0}},
                    {C{0, 0}, C{0, 0}, C{0, 0}, C{std::sqrt(p1), 0}}};
        }
        RT_FAIL(("Channel " + name +
                 " is not supported by the runtime; "
                 "use a noise model instead")
                    .c_str());
    }

    // ---------------- gates used internally (noiseless) ----------------

    void applyPauliRotRaw(const std::string &word,
                          const std::vector<std::size_t> &dev_wires,
                          const std::vector<double> &params, bool inverse) {
        RT_FAIL_IF(word.size() != dev_wires.size(),
                   "PauliRot word and wires must have the same length");
        std::string w;
        std::vector<std::size_t> wires;
        for (std::size_t i = 0; i < word.size(); i++) {
            if (word[i] != 'I') {
                w.push_back(word[i]);
                wires.push_back(dev_wires[i]);
            }
        }
        if (w.empty()) {
            return; // global phase
        }
        state_->applyPauliRot(wires, inverse, params, w);
    }

    /// Rotate `word` on `wires` to Z...Z (or undo it)
    void rotateToZ(const std::string &word,
                   const std::vector<std::size_t> &wires, bool undo) {
        for (std::size_t i = 0; i < word.size(); i++) {
            if (word[i] == 'X') {
                state_->applyOperation("Hadamard", {}, {}, {wires[i]}, false,
                                       {});
            } else if (word[i] == 'Y') {
                if (undo) {
                    state_->applyOperation("Hadamard", {}, {}, {wires[i]},
                                           false, {});
                    state_->applyOperation("S", {}, {}, {wires[i]}, false, {});
                } else {
                    state_->applyOperation("S", {}, {}, {wires[i]}, true, {});
                    state_->applyOperation("Hadamard", {}, {}, {wires[i]},
                                           false, {});
                }
            }
        }
    }

    bool measureDevice(std::size_t dev, std::optional<int32_t> postselect) {
        auto p = state_->probs({dev});
        const bool mres = Lightning::simulateDraw(p, postselect, &rng());
        state_->collapse(dev, mres);
        return mres;
    }

    // ---------------- observables ----------------

    void checkObsKeys(const std::vector<ObsIdType> &keys) const {
        for (auto k : keys) {
            RT_FAIL_IF(k < 0 ||
                           static_cast<std::size_t>(k) >= observables_.size(),
                       "Invalid key for cached observables");
        }
    }

    static std::string obsName(ObsId id) {
        switch (id) {
        case ObsId::Identity:
            return "Identity";
        case ObsId::PauliX:
            return "PauliX";
        case ObsId::PauliY:
            return "PauliY";
        case ObsId::PauliZ:
            return "PauliZ";
        case ObsId::Hadamard:
            return "Hadamard";
        case ObsId::Hermitian:
            return "Hermitian";
        }
        RT_FAIL("Unknown observable");
    }

    [[nodiscard]] ObsTermsD toTerms(ObsIdType key) const {
        const auto &n = observables_[static_cast<std::size_t>(key)];
        switch (n.kind) {
        case ObsNode::Kind::Leaf:
            return Pennylane::LightningNoisy::namedObservable<double>(
                obsName(n.id), deviceWires(n.wires), n.matrix);
        case ObsNode::Kind::Tensor: {
            std::vector<ObsTermsD> f;
            for (auto c : n.children) {
                f.push_back(toTerms(c));
            }
            return Pennylane::LightningNoisy::productObservable<double>(f);
        }
        case ObsNode::Kind::Hamiltonian: {
            std::vector<ObsTermsD> f;
            for (auto c : n.children) {
                f.push_back(toTerms(c));
            }
            return Pennylane::LightningNoisy::linearCombination<double>(
                n.coeffs, f);
        }
        }
        RT_FAIL("Unknown observable kind");
    }

    /// Program wires of an observable (sorted, unique)
    [[nodiscard]] std::vector<QubitIdType>
    observableWires(ObsIdType key) const {
        std::vector<QubitIdType> out;
        collectWires(key, out);
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    void collectWires(ObsIdType key, std::vector<QubitIdType> &out) const {
        const auto &n = observables_[static_cast<std::size_t>(key)];
        if (n.kind == ObsNode::Kind::Leaf) {
            out.insert(out.end(), n.wires.begin(), n.wires.end());
            return;
        }
        for (auto c : n.children) {
            collectWires(c, out);
        }
    }

    void collectLeaves(ObsIdType key, std::vector<const ObsNode *> &out) const {
        const auto &n = observables_[static_cast<std::size_t>(key)];
        if (n.kind == ObsNode::Kind::Leaf) {
            out.push_back(&n);
            return;
        }
        for (auto c : n.children) {
            collectLeaves(c, out);
        }
    }

    /**
     * @brief Finite-shot (mean, variance) of an observable.
     *
     * Hamiltonians are estimated term by term (expval = sum c_i <O_i>,
     * var = sum c_i^2 var(O_i)), like lightning.kokkos. Each tensor product
     * is measured by rotating every factor to its eigenbasis and sampling
     * the union of their wires.
     */
    std::pair<double, double> shotExpval(const StateT &st, ObsIdType key) {
        const auto &n = observables_[static_cast<std::size_t>(key)];
        if (n.kind == ObsNode::Kind::Hamiltonian) {
            double e = 0;
            double v = 0;
            for (std::size_t i = 0; i < n.children.size(); i++) {
                const auto [ei, vi] = shotExpval(st, n.children[i]);
                e += n.coeffs[i] * ei;
                v += n.coeffs[i] * n.coeffs[i] * vi;
            }
            return {e, v};
        }
        std::vector<const ObsNode *> leaves;
        collectLeaves(key, leaves);

        struct Factor {
            std::vector<std::size_t> wires; // device wires
            std::vector<double> eig;        // eigenvalue per basis index
            CVecD rot;                      // V^dagger (empty for Z/I)
            std::string named;              // "X", "Y" for fast rotations
        };
        std::vector<Factor> factors;
        std::unordered_set<std::size_t> seen;
        bool overlap = false;
        for (const auto *l : leaves) {
            if (l->id == ObsId::Identity) {
                continue;
            }
            for (auto w : deviceWires(l->wires)) {
                overlap = overlap || !seen.insert(w).second;
            }
        }
        if (overlap) {
            // overlapping factors: diagonalise the product as a whole
            auto terms = toTerms(key);
            RT_FAIL_IF(terms.size() != 1U,
                       "Unsupported observable for finite shots");
            const auto &t = terms[0];
            Factor f;
            f.wires = t.wires;
            auto m = Pennylane::LightningNoisy::termMatrix<double>(t);
            for (auto &x : m) {
                x *= t.coeff;
            }
            const std::size_t d = std::size_t{1} << t.wires.size();
            auto [w, v] =
                Pennylane::LightningNoisy::hermitianEigen<double>(m, d);
            f.eig = w;
            f.rot = Pennylane::LightningNoisy::adjoint<double>(v, d);
            factors.push_back(std::move(f));
        } else {
            for (const auto *l : leaves) {
                Factor f;
                f.wires = deviceWires(l->wires);
                switch (l->id) {
                case ObsId::Identity:
                    continue;
                case ObsId::PauliX:
                    f.named = "X";
                    f.eig = {1, -1};
                    break;
                case ObsId::PauliY:
                    f.named = "Y";
                    f.eig = {1, -1};
                    break;
                case ObsId::PauliZ:
                    f.eig = {1, -1};
                    break;
                case ObsId::Hadamard:
                case ObsId::Hermitian: {
                    CVecD m = l->matrix;
                    if (l->id == ObsId::Hadamard) {
                        const double s = 1.0 / std::sqrt(2.0);
                        m = {{s, 0}, {s, 0}, {s, 0}, {-s, 0}};
                    }
                    const std::size_t d = std::size_t{1} << f.wires.size();
                    auto [w, v] =
                        Pennylane::LightningNoisy::hermitianEigen<double>(m, d);
                    f.eig = w;
                    f.rot = Pennylane::LightningNoisy::adjoint<double>(v, d);
                    break;
                }
                }
                factors.push_back(std::move(f));
            }
        }
        if (factors.empty()) {
            return {1.0, 0.0}; // identity
        }
        StateT rotated(st);
        std::vector<std::size_t> all;
        for (const auto &f : factors) {
            if (f.named == "X") {
                rotated.applyOperation("Hadamard", {}, {}, f.wires, false, {});
            } else if (f.named == "Y") {
                rotated.applyOperation("S", {}, {}, f.wires, true, {});
                rotated.applyOperation("Hadamard", {}, {}, f.wires, false, {});
            } else if (!f.rot.empty()) {
                rotated.applyMatrix(f.rot.data(), f.rot.size(), f.wires, false);
            }
            all.insert(all.end(), f.wires.begin(), f.wires.end());
        }
        const auto bits = rotated.sampleWires(all, device_shots_, rng());
        const std::size_t k = all.size();
        double s1 = 0;
        double s2 = 0;
        for (std::size_t s = 0; s < device_shots_; s++) {
            double val = 1;
            std::size_t off = 0;
            for (const auto &f : factors) {
                std::size_t idx = 0;
                for (std::size_t j = 0; j < f.wires.size(); j++) {
                    idx = (idx << 1U) | bits[s * k + off + j];
                }
                off += f.wires.size();
                val *= f.eig[idx];
            }
            s1 += val;
            s2 += val * val;
        }
        const double ns = static_cast<double>(device_shots_);
        const double mean = s1 / ns;
        return {mean, s2 / ns - mean * mean};
    }
};

} // namespace Catalyst::Runtime::Simulator
