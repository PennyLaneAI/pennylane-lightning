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
 * @file NoiseModel.hpp
 * Native (compiled) noise model of the noisy Kokkos backends
 * (lightning.tensornet).
 *
 * A PennyLane `qml.NoiseModel` is lowered in Python
 * (`pennylane_lightning.lightning_base._noise`) into a flat rule table that
 * the C++ backends (and the Catalyst runtime devices) apply natively after /
 * before each gate and before terminal measurements.
 *
 * Wire format (a single string, safe for Catalyst's `parse_kwargs`, i.e. no
 * ',', ':', '{', '}', ' ' or quotes):
 *
 *   LNM1;<entry>;<entry>;...
 *
 *   K=<nq>/<nk>/<re>/<im>/...         channel (Kraus ops, row-major, all nk
 *                                     Kraus operators concatenated); channels
 *                                     are numbered in order of appearance
 *   G=<gate>/<wires>/<actions>        gate rule
 *   M=<kind>/<wires>/<actions>        readout rule (applied before a terminal
 *                                     measurement of type <kind>; <wires> is
 *                                     a sorted tuple or '*<k>')
 *
 *   <wires>   '*<k>' (wildcard for arity k) or w0_w1_... (exact wires)
 *   <actions> '-' (no noise) or act+act+... with
 *             act = ('b'|'a')<channel>@ref_ref_...
 *             ref = 'r<k>' (k-th wire of the operation) or an absolute wire
 */
#pragma once

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "GateMatrices.hpp"

namespace Pennylane::LightningNoisy {

/**
 * @brief A quantum channel described by its Kraus operators.
 */
template <class T> struct KrausChannel {
    std::size_t num_qubits{1};
    std::vector<CVec<T>> kraus{};
    CVec<T> superop{}; // cached sum_k K (x) K*

    void finalize() { superop = krausToSuperop<T>(kraus, pow2(num_qubits)); }
};

struct NoiseAction {
    bool before{false};
    std::size_t channel{0};
    // each wire reference: (is_relative, value)
    std::vector<std::pair<bool, std::size_t>> refs{};
};

struct ResolvedNoise {
    std::size_t channel;
    std::vector<std::size_t> wires;
};

template <class T> class NoiseModel {
  public:
    NoiseModel() = default;

    explicit NoiseModel(const std::string &spec) { parse(spec); }

    [[nodiscard]] bool empty() const {
        return gate_rules_.empty() && meas_rules_.empty();
    }
    [[nodiscard]] bool hasReadoutNoise() const { return !meas_rules_.empty(); }
    [[nodiscard]] std::size_t numChannels() const { return channels_.size(); }
    [[nodiscard]] const KrausChannel<T> &channel(std::size_t i) const {
        return channels_.at(i);
    }
    [[nodiscard]] const std::vector<KrausChannel<T>> &channels() const {
        return channels_;
    }

    /**
     * @brief Find the noise to apply around a gate.
     *
     * @param name Gate name as seen by the backend ("C(<name>)" for
     * controlled gates).
     * @param wires All wires of the gate (controls first, then targets).
     */
    void lookupGate(const std::string &name,
                    const std::vector<std::size_t> &wires,
                    std::vector<ResolvedNoise> &before,
                    std::vector<ResolvedNoise> &after) const {
        before.clear();
        after.clear();
        const auto *acts = find(gate_rules_, name, wires);
        if (acts == nullptr) {
            return;
        }
        for (const auto &a : *acts) {
            ResolvedNoise r{a.channel, resolve(a, wires)};
            (a.before ? before : after).push_back(std::move(r));
        }
    }

    /**
     * @brief Readout noise to apply before measuring `wires` with a
     * measurement of type `kind` (expval, var, probs, sample, counts).
     *
     * Lookup order: exact (sorted) wire tuple, then arity wildcard, then the
     * union of the single-wire rules of every measured wire.
     */
    void lookupReadout(const std::string &kind,
                       const std::vector<std::size_t> &wires,
                       std::vector<ResolvedNoise> &out) const {
        out.clear();
        if (meas_rules_.empty() || wires.empty()) {
            return;
        }
        std::vector<std::size_t> sorted = wires;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.size() > 1) {
            const auto *acts = find(meas_rules_, kind, sorted);
            if (acts != nullptr) {
                for (const auto &a : *acts) {
                    out.push_back({a.channel, resolve(a, sorted)});
                }
                return;
            }
        }
        for (auto w : wires) {
            const std::vector<std::size_t> one{w};
            const auto *acts = find(meas_rules_, kind, one);
            if (acts == nullptr) {
                continue;
            }
            for (const auto &a : *acts) {
                out.push_back({a.channel, resolve(a, one)});
            }
        }
    }

    /**
     * @brief Parse a serialized noise model (see file documentation).
     */
    void parse(const std::string &spec) {
        channels_.clear();
        gate_rules_.clear();
        meas_rules_.clear();
        if (spec.empty()) {
            return;
        }
        auto entries = split(spec, ';');
        PL_ABORT_IF(entries.empty() || entries[0] != "LNM1",
                    "Invalid noise model specification (missing LNM1 "
                    "header).");
        for (std::size_t e = 1; e < entries.size(); e++) {
            const auto &ent = entries[e];
            if (ent.empty()) {
                continue;
            }
            PL_ABORT_IF(ent.size() < 2 || ent[1] != '=',
                        "Invalid noise model entry: " + ent);
            const char kind = ent[0];
            const auto body = ent.substr(2);
            if (kind == 'K') {
                parseChannel(body);
            } else if (kind == 'G') {
                parseRule(body, gate_rules_);
            } else if (kind == 'M') {
                parseRule(body, meas_rules_);
            } else {
                PL_ABORT("Invalid noise model entry: " + ent);
            }
        }
    }

  private:
    using RuleMap = std::unordered_map<std::string, std::vector<NoiseAction>>;

    std::vector<KrausChannel<T>> channels_;
    RuleMap gate_rules_;
    RuleMap meas_rules_;

    static std::vector<std::string> split(const std::string &s, char sep) {
        std::vector<std::string> out;
        std::string cur;
        for (char ch : s) {
            if (ch == sep) {
                out.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(ch);
            }
        }
        out.push_back(cur);
        return out;
    }

    static std::string exactKey(const std::string &name,
                                const std::vector<std::size_t> &wires) {
        std::string key = name + "|";
        for (std::size_t i = 0; i < wires.size(); i++) {
            if (i) {
                key += "_";
            }
            key += std::to_string(wires[i]);
        }
        return key;
    }

    static std::string wildKey(const std::string &name, std::size_t arity) {
        return name + "#" + std::to_string(arity);
    }

    static const std::vector<NoiseAction> *
    find(const RuleMap &rules, const std::string &name,
         const std::vector<std::size_t> &wires) {
        if (rules.empty()) {
            return nullptr;
        }
        auto it = rules.find(exactKey(name, wires));
        if (it != rules.end()) {
            return &it->second;
        }
        it = rules.find(wildKey(name, wires.size()));
        if (it != rules.end()) {
            return &it->second;
        }
        return nullptr;
    }

    static std::vector<std::size_t>
    resolve(const NoiseAction &a, const std::vector<std::size_t> &op_wires) {
        std::vector<std::size_t> w;
        w.reserve(a.refs.size());
        for (auto [rel, v] : a.refs) {
            if (rel) {
                PL_ABORT_IF(v >= op_wires.size(),
                            "Noise rule references a wire outside of the "
                            "operation.");
                w.push_back(op_wires[v]);
            } else {
                w.push_back(v);
            }
        }
        return w;
    }

    static std::size_t toSize(const std::string &s) {
        PL_ABORT_IF(s.empty(), "Invalid integer in noise model");
        char *end = nullptr;
        const auto v = std::strtoull(s.c_str(), &end, 10);
        PL_ABORT_IF(*end != '\0', "Invalid integer in noise model: " + s);
        return static_cast<std::size_t>(v);
    }

    static T toReal(const std::string &s) {
        char *end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        PL_ABORT_IF(end == s.c_str() || *end != '\0',
                    "Invalid number in noise model: " + s);
        return static_cast<T>(v);
    }

    void parseChannel(const std::string &body) {
        auto tok = split(body, '/');
        PL_ABORT_IF(tok.size() < 2, "Invalid channel entry");
        KrausChannel<T> ch;
        ch.num_qubits = toSize(tok[0]);
        const std::size_t nk = toSize(tok[1]);
        const std::size_t dim = pow2(ch.num_qubits);
        PL_ABORT_IF(tok.size() != 2 + 2 * nk * dim * dim,
                    "Channel entry has an inconsistent number of values");
        std::size_t p = 2;
        for (std::size_t k = 0; k < nk; k++) {
            CVec<T> m(dim * dim);
            for (std::size_t i = 0; i < dim * dim; i++) {
                m[i] = {toReal(tok[p]), toReal(tok[p + 1])};
                p += 2;
            }
            ch.kraus.push_back(std::move(m));
        }
        ch.finalize();
        channels_.push_back(std::move(ch));
    }

    void parseRule(const std::string &body, RuleMap &rules) {
        auto tok = split(body, '/');
        PL_ABORT_IF(tok.size() != 3, "Invalid noise rule: " + body);
        const auto &name = tok[0];
        std::string key;
        if (!tok[1].empty() && tok[1][0] == '*') {
            key = wildKey(name, toSize(tok[1].substr(1)));
        } else {
            std::vector<std::size_t> wires;
            for (const auto &w : split(tok[1], '_')) {
                wires.push_back(toSize(w));
            }
            key = exactKey(name, wires);
        }
        std::vector<NoiseAction> acts;
        if (tok[2] != "-") {
            for (const auto &a : split(tok[2], '+')) {
                PL_ABORT_IF(a.size() < 4, "Invalid noise action: " + a);
                NoiseAction na;
                PL_ABORT_IF(a[0] != 'a' && a[0] != 'b',
                            "Invalid noise action: " + a);
                na.before = (a[0] == 'b');
                const auto at = a.find('@');
                PL_ABORT_IF(at == std::string::npos,
                            "Invalid noise action: " + a);
                na.channel = toSize(a.substr(1, at - 1));
                PL_ABORT_IF(na.channel >= channels_.size(),
                            "Noise action references an unknown channel");
                for (const auto &r : split(a.substr(at + 1), '_')) {
                    if (!r.empty() && r[0] == 'r') {
                        na.refs.emplace_back(true, toSize(r.substr(1)));
                    } else {
                        na.refs.emplace_back(false, toSize(r));
                    }
                }
                PL_ABORT_IF(na.refs.size() != channels_[na.channel].num_qubits,
                            "Noise action wires do not match the channel "
                            "size");
                acts.push_back(std::move(na));
            }
        }
        rules[key] = std::move(acts);
    }
};

} // namespace Pennylane::LightningNoisy
