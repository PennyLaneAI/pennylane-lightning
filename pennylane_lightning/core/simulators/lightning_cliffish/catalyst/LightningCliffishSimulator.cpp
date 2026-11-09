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

#include "LightningCliffishSimulator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string_view>
#include <unordered_set>

#include "Exception.hpp"
#include "Utils.hpp"

namespace {
using Pennylane::LightningCliffish::Planner::HybridOptions;
using Pennylane::LightningCliffish::Planner::Observable;
using Pennylane::LightningCliffish::Planner::PauliTerm;
using Pennylane::LightningCliffish::Program::Op;
using Pennylane::LightningCliffish::Program::OpKind;
using Pennylane::LightningCliffish::Program::QubitId;

auto toCore(const std::vector<QubitIdType> &wires) -> std::vector<QubitId> {
    std::vector<QubitId> result(wires.size());
    std::transform(wires.begin(), wires.end(), result.begin(),
                   [](QubitIdType wire) { return static_cast<QubitId>(wire); });
    return result;
}

auto parseCollapse(const std::string &kwargs)
    -> Pennylane::LightningCliffish::Planner::GadgetCollapse {
    auto args = Catalyst::Runtime::parse_kwargs(kwargs);
    if (args.contains("gadget_collapse")) {
        return Pennylane::LightningCliffish::Planner::parseGadgetCollapse(
            args["gadget_collapse"]);
    }
    return Pennylane::LightningCliffish::Planner::GadgetCollapse::Measured;
}

auto parseOptions(const std::string &kwargs) -> HybridOptions {
    auto args = Catalyst::Runtime::parse_kwargs(kwargs);
    HybridOptions options;
    if (args.contains("max_dense_qubits")) {
        options.max_dense_qubits =
            static_cast<std::size_t>(std::stoull(args["max_dense_qubits"]));
    }
    if (args.contains("recompression")) {
        options.recompression =
            Pennylane::LightningCliffish::Planner::parseRecompression(
                args["recompression"]);
    }
    if (args.contains("dense_checks")) {
        options.dense_checks = args["dense_checks"] == "True";
    }
    return options;
}
} // namespace

namespace Catalyst::Runtime::Simulator {

LightningCliffishSimulator::LightningCliffishSimulator(
    const std::string &kwargs)
    : core{parseOptions(kwargs), parseCollapse(kwargs)} {
#ifdef LIGHTNING_CLIFFISH_DEVICE_EXTENSIONS
    using Self = LightningCliffishSimulator;
    delegated_table = CatalystDelegatedTable{
        .header = {.abi_version = 1, .reserved = 0, .self = this},
        .begin_program =
            [](void *device, uint32_t mode) {
                static_cast<Self *>(device)->BeginDelegatedProgram(mode);
            },
        .push_predicate =
            [](void *device, RESULT *condition) {
                static_cast<Self *>(device)->PushPredicate(condition);
            },
        .pop_predicate =
            [](void *device) { static_cast<Self *>(device)->PopPredicate(); },
        .result_constant =
            [](void *device, bool value) {
                return static_cast<Self *>(device)->ResultConstant(value);
            },
        .result_xor =
            [](void *device, RESULT *a, RESULT *b) {
                return static_cast<Self *>(device)->ResultXor(a, b);
            },
        .result_not =
            [](void *device, RESULT *a) {
                return static_cast<Self *>(device)->ResultNot(a);
            },
        .result_select =
            [](void *device, RESULT *condition, RESULT *if_true,
               RESULT *if_false) {
                return static_cast<Self *>(device)->ResultSelect(
                    condition, if_true, if_false);
            },
        .release_result =
            [](void *device, RESULT *result) {
                static_cast<Self *>(device)->ReleaseResult(result);
            },
        .pauli_measure =
            [](void *device, const char *word, const QubitIdType *wires,
               size_t num_wires, bool negated, int32_t postselect) {
                return static_cast<Self *>(device)->SignedPauliMeasure(
                    word, std::vector<QubitIdType>(wires, wires + num_wires),
                    negated,
                    postselect == CATALYST_NO_POSTSELECT
                        ? std::nullopt
                        : std::optional<int32_t>{postselect});
            },
        // Outcomes reach the host only through measurement processes for now.
        .result_samples = nullptr,
    };
    gadget_table = CatalystGadgetTable{
        .header = {.abi_version = 1, .reserved = 0, .self = this},
        .begin_gadget =
            [](void *device, const char *word, const QubitIdType *wires,
               size_t num_wires, double theta, uint32_t verdict) {
                static_cast<Self *>(device)->BeginGadget(
                    word, std::vector<QubitIdType>(wires, wires + num_wires),
                    theta, verdict);
            },
        .end_gadget =
            [](void *device) { static_cast<Self *>(device)->EndGadget(); },
    };
#endif
}

#ifdef LIGHTNING_CLIFFISH_DEVICE_EXTENSIONS
auto LightningCliffishSimulator::QueryInterface(const char *name,
                                                uint32_t min_version) -> const
    void * {
    if (min_version > 1) {
        return nullptr;
    }
    const std::string_view table{name};
    if (table == "Delegated") {
        return &delegated_table;
    }
    if (table == "Gadget") {
        return &gadget_table;
    }
    return nullptr;
}
#endif

auto LightningCliffishSimulator::handle(ResultId result) -> Result {
    bool &slot = handle_storage.emplace_back(false);
    handle_results.emplace(&slot, result);
    return &slot;
}

auto LightningCliffishSimulator::resultOf(Result handle) const -> ResultId {
    const auto it = handle_results.find(handle);
    RT_FAIL_IF(it == handle_results.end(),
               "lightning.cliffish: unknown result handle on a delegated "
               "program");
    return it->second;
}

auto LightningCliffishSimulator::predicate() const -> ResultId {
    return predicates.empty()
               ? Pennylane::LightningCliffish::Program::result_true
               : predicates.back();
}

auto LightningCliffishSimulator::measurement(Op op) -> Result {
    if (core.isDelegated()) {
        op.predicate = predicate();
        return handle(core.measure(std::move(op)));
    }
    // Without delegated feed-forward, a measurement value is demanded now.
    const bool outcome = core.value(core.measure(std::move(op)));
    return outcome ? const_cast<Result>(&GLOBAL_RESULT_TRUE_CONST)
                   : const_cast<Result>(&GLOBAL_RESULT_FALSE_CONST);
}

void LightningCliffishSimulator::BeginDelegatedProgram(
    uint32_t postselect_mode) {
    RT_FAIL_IF(postselect_mode != 0,
               "lightning.cliffish supports only postselect_mode='fill-shots' "
               "on a delegated program");
    core.beginDelegated();
    predicates.clear();
}

void LightningCliffishSimulator::PushPredicate(Result condition) {
    ResultId conjunction = resultOf(condition);
    if (!predicates.empty()) {
        // a AND b is SELECT(a, b, false).
        const ResultId never =
            core.combine(Op{.kind = OpKind::ResultConstant, .params = {0.0}});
        conjunction = core.combine(
            Op{.kind = OpKind::ResultSelect,
               .operands = {predicates.back(), conjunction, never}});
    }
    predicates.push_back(conjunction);
}

void LightningCliffishSimulator::PopPredicate() {
    RT_FAIL_IF(predicates.empty(), "Predicate scopes are unbalanced");
    predicates.pop_back();
}

auto LightningCliffishSimulator::ResultConstant(bool value) -> Result {
    return handle(core.combine(
        Op{.kind = OpKind::ResultConstant, .params = {value ? 1.0 : 0.0}}));
}

auto LightningCliffishSimulator::ResultXor(Result a, Result b) -> Result {
    return handle(core.combine(
        Op{.kind = OpKind::ResultXor, .operands = {resultOf(a), resultOf(b)}}));
}

auto LightningCliffishSimulator::ResultNot(Result a) -> Result {
    return handle(
        core.combine(Op{.kind = OpKind::ResultNot, .operands = {resultOf(a)}}));
}

auto LightningCliffishSimulator::ResultSelect(Result condition, Result if_true,
                                              Result if_false) -> Result {
    return handle(
        core.combine(Op{.kind = OpKind::ResultSelect,
                        .operands = {resultOf(condition), resultOf(if_true),
                                     resultOf(if_false)}}));
}

void LightningCliffishSimulator::ReleaseResult(Result result) {
    core.mark(
        Op{.kind = OpKind::ResultRelease, .operands = {resultOf(result)}});
    handle_results.erase(result);
}

auto LightningCliffishSimulator::SignedPauliMeasure(
    const std::string &pauli_word, const std::vector<QubitIdType> &wires,
    bool negated, std::optional<int32_t> postselect) -> Result {
    Op op{.kind = OpKind::PauliMeasure,
          .wires = toCore(wires),
          .pauli_word = pauli_word,
          .negated = negated};
    if (postselect.has_value()) {
        RT_FAIL_IF(*postselect != 0 && *postselect != 1,
                   "Postselection must be 0 or 1");
        op.postselect = *postselect == 1;
    }
    return measurement(std::move(op));
}

void LightningCliffishSimulator::BeginGadget(
    const std::string &pauli_word, const std::vector<QubitIdType> &wires,
    double theta, uint32_t verdict) {
    // Only a delegated program sees a gadget whole; elsewhere the span runs as
    // given.
    if (core.isDelegated()) {
        core.mark(Op{.kind = OpKind::GadgetBegin,
                     .params = {theta, static_cast<double>(verdict)},
                     .wires = toCore(wires),
                     .pauli_word = pauli_word,
                     .predicate = predicate()});
    }
}

void LightningCliffishSimulator::EndGadget() {
    if (core.isDelegated()) {
        core.mark(Op{.kind = OpKind::GadgetEnd});
    }
}

auto LightningCliffishSimulator::observable(ObsIdType key) const
    -> const CoreObservable & {
    RT_FAIL_IF(key < 0 || static_cast<std::size_t>(key) >= observables.size(),
               "Invalid observable key");
    return observables[static_cast<std::size_t>(key)];
}

auto LightningCliffishSimulator::AllocateQubit() -> QubitIdType {
    return AllocateQubits(1).front();
}

auto LightningCliffishSimulator::AllocateQubits(std::size_t num_qubits)
    -> std::vector<QubitIdType> {
    std::vector<QubitIdType> ids;
    ids.reserve(num_qubits);
    for (auto id : core.allocate(num_qubits)) {
        ids.push_back(static_cast<QubitIdType>(id));
    }
    qubits.insert(qubits.end(), ids.begin(), ids.end());
    return ids;
}

void LightningCliffishSimulator::ReleaseQubit(QubitIdType q) {
    core.release(static_cast<QubitId>(q));
    qubits.erase(std::find(qubits.begin(), qubits.end(), q));
}

void LightningCliffishSimulator::ReleaseQubits(
    const std::vector<QubitIdType> &ids) {
    for (auto id : ids) {
        ReleaseQubit(id);
    }
}

auto LightningCliffishSimulator::GetNumQubits() const -> std::size_t {
    return qubits.size();
}

void LightningCliffishSimulator::SetDeviceShots(std::size_t shots) {
    device_shots = shots;
    // Catalyst sets the shots when it initializes the device, so a new
    // execution starts here: it is not delegated until it says so.
    core.endDelegated();
    predicates.clear();
    handle_storage.clear();
    handle_results.clear();
}

auto LightningCliffishSimulator::GetDeviceShots() const -> std::size_t {
    return device_shots;
}

void LightningCliffishSimulator::SetDevicePRNG(std::mt19937 *gen) {
    core.setPRNG(gen);
}

void LightningCliffishSimulator::NamedOperation(
    const std::string &name, const std::vector<double> &params,
    const std::vector<QubitIdType> &wires, bool inverse,
    const std::vector<QubitIdType> &controlled_wires,
    const std::vector<bool> &controlled_values,
    const std::vector<std::string> &optional_params) {
    Op op{.kind = OpKind::Gate,
          .name = name,
          .params = params,
          .wires = toCore(wires),
          .inverse = inverse,
          .controlled_wires = toCore(controlled_wires),
          .controlled_values = controlled_values};
    if (name == "PauliRot") {
        RT_FAIL_IF(optional_params.size() != 1,
                   "PauliRot needs its Pauli word as its only optional "
                   "parameter");
        op.kind = OpKind::PauliRot;
        op.pauli_word = optional_params.front();
    }
    op.predicate = predicate();
    core.append(std::move(op));
}

auto LightningCliffishSimulator::Observable(
    ObsId id, [[maybe_unused]] const std::vector<std::complex<double>> &matrix,
    const std::vector<QubitIdType> &wires) -> ObsIdType {
    const auto core_wires = toCore(wires);
    CoreObservable result;
    switch (id) {
    case ObsId::Identity:
        result = {PauliTerm{1.0, "", {}}};
        break;
    case ObsId::PauliX:
        result = {PauliTerm{1.0, "X", core_wires}};
        break;
    case ObsId::PauliY:
        result = {PauliTerm{1.0, "Y", core_wires}};
        break;
    case ObsId::PauliZ:
        result = {PauliTerm{1.0, "Z", core_wires}};
        break;
    case ObsId::Hadamard: {
        // H = (X + Z) / sqrt(2)
        const double coefficient = 1.0 / std::sqrt(2.0);
        result = {PauliTerm{coefficient, "X", core_wires},
                  PauliTerm{coefficient, "Z", core_wires}};
        break;
    }
    default:
        RT_FAIL("lightning.cliffish supports only Pauli observables, their "
                "products and sums");
    }
    RT_FAIL_IF(id != ObsId::Identity && core_wires.size() != 1,
               "A named observable acts on exactly one wire");
    observables.push_back(std::move(result));
    return static_cast<ObsIdType>(observables.size() - 1);
}

auto LightningCliffishSimulator::TensorObservable(
    const std::vector<ObsIdType> &obs) -> ObsIdType {
    // The product of sums, expanded term by term.
    CoreObservable product{PauliTerm{1.0, "", {}}};
    for (auto key : obs) {
        CoreObservable next;
        for (const auto &left : product) {
            for (const auto &right : observable(key)) {
                std::unordered_set<QubitId> seen(left.wires.begin(),
                                                 left.wires.end());
                for (auto wire : right.wires) {
                    RT_FAIL_IF(!seen.insert(wire).second,
                               "A tensor product of observables must act on "
                               "distinct wires");
                }
                PauliTerm term{left.coefficient * right.coefficient,
                               left.pauli + right.pauli, left.wires};
                term.wires.insert(term.wires.end(), right.wires.begin(),
                                  right.wires.end());
                next.push_back(std::move(term));
            }
        }
        product = std::move(next);
    }
    observables.push_back(std::move(product));
    return static_cast<ObsIdType>(observables.size() - 1);
}

auto LightningCliffishSimulator::HamiltonianObservable(
    const std::vector<double> &coeffs, const std::vector<ObsIdType> &obs)
    -> ObsIdType {
    RT_FAIL_IF(coeffs.size() != obs.size(),
               "A Hamiltonian needs one coefficient per observable");
    CoreObservable sum;
    for (std::size_t i = 0; i < obs.size(); i++) {
        for (auto term : observable(obs[i])) {
            term.coefficient *= coeffs[i];
            sum.push_back(std::move(term));
        }
    }
    observables.push_back(std::move(sum));
    return static_cast<ObsIdType>(observables.size() - 1);
}

auto LightningCliffishSimulator::Expval(ObsIdType obsKey) -> double {
    const auto &obs = observable(obsKey);
    return device_shots > 0 ? core.expval(obs, device_shots) : core.expval(obs);
}

auto LightningCliffishSimulator::Var(ObsIdType obsKey) -> double {
    const auto &obs = observable(obsKey);
    return device_shots > 0 ? core.var(obs, device_shots) : core.var(obs);
}

void LightningCliffishSimulator::State(
    [[maybe_unused]] DataView<std::complex<double>, 1> &state) {
    RT_FAIL("lightning.cliffish cannot return the state vector");
}

void LightningCliffishSimulator::probsInto(
    DataView<double, 1> &probs, const std::vector<QubitIdType> &wires) {
    const auto result = core.probabilities(toCore(wires));
    RT_FAIL_IF(probs.size() != result.size(),
               "Invalid size for the pre-allocated probabilities");
    std::copy(result.begin(), result.end(), probs.begin());
}

void LightningCliffishSimulator::Probs(DataView<double, 1> &probs) {
    probsInto(probs, qubits);
}

void LightningCliffishSimulator::PartialProbs(
    DataView<double, 1> &probs, const std::vector<QubitIdType> &wires) {
    probsInto(probs, wires);
}

void LightningCliffishSimulator::sampleInto(
    DataView<double, 2> &samples, const std::vector<QubitIdType> &wires) {
    const auto result = core.sample(toCore(wires), device_shots);
    RT_FAIL_IF(samples.size() != device_shots * wires.size(),
               "Invalid size for the pre-allocated samples");
    for (std::size_t shot = 0; shot < device_shots; shot++) {
        for (std::size_t k = 0; k < wires.size(); k++) {
            samples(shot, k) = result.bit(shot, k) ? 1.0 : 0.0;
        }
    }
}

void LightningCliffishSimulator::Sample(DataView<double, 2> &samples) {
    sampleInto(samples, qubits);
}

void LightningCliffishSimulator::PartialSample(
    DataView<double, 2> &samples, const std::vector<QubitIdType> &wires) {
    sampleInto(samples, wires);
}

void LightningCliffishSimulator::countsInto(
    DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
    const std::vector<QubitIdType> &wires) {
    RT_FAIL_IF(wires.size() >= 64, "Counts hold at most 63 wires");
    const std::size_t num_elements = std::size_t{1} << wires.size();
    RT_FAIL_IF(eigvals.size() != num_elements || counts.size() != num_elements,
               "Invalid size for the pre-allocated counts");

    // The eigenvalue of each entry is the integer of its basis state, with
    // wires[0] the most significant bit.
    std::iota(eigvals.begin(), eigvals.end(), 0);
    std::fill(counts.begin(), counts.end(), 0);
    for (const auto &[index, count] :
         core.counts(toCore(wires), device_shots)) {
        counts(static_cast<std::size_t>(index)) = static_cast<int64_t>(count);
    }
}

void LightningCliffishSimulator::Counts(DataView<double, 1> &eigvals,
                                        DataView<int64_t, 1> &counts) {
    countsInto(eigvals, counts, qubits);
}

void LightningCliffishSimulator::PartialCounts(
    DataView<double, 1> &eigvals, DataView<int64_t, 1> &counts,
    const std::vector<QubitIdType> &wires) {
    countsInto(eigvals, counts, wires);
}

auto LightningCliffishSimulator::Measure(QubitIdType wire,
                                         std::optional<int32_t> postselect)
    -> Result {
    Op op{.kind = OpKind::Measure, .wires = {static_cast<QubitId>(wire)}};
    if (postselect.has_value()) {
        RT_FAIL_IF(*postselect != 0 && *postselect != 1,
                   "Postselection must be 0 or 1");
        op.postselect = *postselect == 1;
    }
    return measurement(std::move(op));
}

auto LightningCliffishSimulator::PauliMeasure(
    const std::string &pauli_word, const std::vector<QubitIdType> &wires)
    -> Result {
    return SignedPauliMeasure(pauli_word, wires, false, std::nullopt);
}

} // namespace Catalyst::Runtime::Simulator

/// LCOV_EXCL_START
GENERATE_DEVICE_FACTORY(
    LightningCliffishSimulator,
    Catalyst::Runtime::Simulator::LightningCliffishSimulator);
/// LCOV_EXCL_STOP
