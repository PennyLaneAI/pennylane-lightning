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
 * @file GateExpansions.hpp
 * The static per-name table of exact gate expansions.
 *
 * Every supported gate is, up to a global phase, a product of rotations
 * exp(-i theta/2 P) for Pauli words P. A Clifford rotation is held as
 * exp(-i k pi/4 P) with k in {1, 2, 3}; any other angle is a non-Clifford
 * rotation for the dense register.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "Error.hpp"
#include "Program.hpp"

namespace Pennylane::LightningCliffish::Planner {

/**
 * @brief exp(-i quarter_turns pi/4 P), or exp(-i angle/2 P) for a
 * non-Clifford angle, with character k of `pauli` acting on wire k of the op.
 */
struct LocalRotation {
    std::string pauli;
    std::uint8_t quarter_turns;
    /// Non-Clifford rotations only; quarter_turns is then 0.
    std::optional<double> angle{};

    /// The rotation is exp(-i halfAngle() P).
    [[nodiscard]] auto halfAngle() const -> double {
        return angle.has_value() ? *angle / 2 : quarter_turns * M_PI / 4;
    }

    friend auto operator==(const LocalRotation &, const LocalRotation &)
        -> bool = default;
};

/// A rotation snaps to a Clifford angle only this close to a non-identity one.
inline constexpr double clifford_snap = 1e-12;

/**
 * @brief The quarter turns k with exp(-i angle/2 P) = exp(-i k pi/4 P), when
 * `angle` is a Clifford angle.
 *
 * An angle within `clifford_snap` of a non-identity Clifford angle snaps to
 * it. A rotation near the identity is never snapped: only an angle of exactly
 * zero is the identity.
 *
 * @return k in [0, 4), or nothing for a non-Clifford angle.
 */
inline auto cliffordQuarterTurns(double angle) -> std::optional<std::uint8_t> {
    const double quarter = M_PI / 2;
    const long long turns = std::llround(angle / quarter);
    const double remainder = angle - static_cast<double>(turns) * quarter;
    const auto quarter_turns = static_cast<std::uint8_t>(((turns % 4) + 4) % 4);
    if (quarter_turns == 0) {
        return remainder == 0.0 ? std::optional<std::uint8_t>{0} : std::nullopt;
    }
    if (std::abs(remainder) <= clifford_snap) {
        return quarter_turns;
    }
    return std::nullopt;
}

/**
 * @brief Expansions of the fixed Clifford gates, in application order.
 */
inline auto fixedGateExpansions() -> const
    std::unordered_map<std::string_view, std::vector<LocalRotation>> & {
    // CNOT = exp(i pi/4 (1 - Z_c)(1 - X_t)), so it is
    // exp(-i pi/4 Z_c) exp(-i pi/4 X_t) exp(i pi/4 Z_c X_t) up to a phase;
    // CY and CZ follow the same pattern. H = exp(-i pi/4 Y) Z. The device has
    // no state output, so a global phase is not observable.
    static const std::unordered_map<std::string_view,
                                    std::vector<LocalRotation>>
        expansions{
            {"Identity", {}},
            {"GlobalPhase", {}},
            {"PauliX", {{"X", 2}}},
            {"PauliY", {{"Y", 2}}},
            {"PauliZ", {{"Z", 2}}},
            {"Hadamard", {{"Z", 2}, {"Y", 1}}},
            {"S", {{"Z", 1}}},
            {"SX", {{"X", 1}}},
            {"CNOT", {{"ZI", 1}, {"IX", 1}, {"ZX", 3}}},
            {"CY", {{"ZI", 1}, {"IY", 1}, {"ZY", 3}}},
            {"CZ", {{"ZI", 1}, {"IZ", 1}, {"ZZ", 3}}},
            {"SWAP", {{"XX", 3}, {"YY", 3}, {"ZZ", 3}}},
            {"ISWAP", {{"XX", 3}, {"YY", 3}}},
        };
    return expansions;
}

/**
 * @brief The Pauli word a parametric rotation turns about, as
 * exp(-i params[0]/2 P), for a gate on `num_wires` wires.
 */
inline auto rotationWord(std::string_view name, std::size_t num_wires)
    -> std::optional<std::string> {
    static const std::unordered_map<std::string_view, std::string> words{
        {"RX", "X"},         {"RY", "Y"},       {"RZ", "Z"},
        {"PhaseShift", "Z"}, {"IsingXX", "XX"}, {"IsingYY", "YY"},
        {"IsingZZ", "ZZ"},
    };
    if (name == "MultiRZ") {
        return std::string(num_wires, 'Z');
    }
    if (const auto it = words.find(name); it != words.end()) {
        return it->second;
    }
    return std::nullopt;
}

/// T = exp(i pi/8) exp(-i pi/8 Z), so it is RZ(pi/4) up to a phase.
inline constexpr double t_gate_angle = M_PI / 4;

/**
 * @brief Expand a gate or PauliRot op into rotations, in application order.
 */
inline auto expandGate(const Program::Op &op) -> std::vector<LocalRotation> {
    using Program::OpKind;
    PL_ABORT_IF_NOT(op.kind == OpKind::Gate || op.kind == OpKind::PauliRot,
                    "Only gates and PauliRot expand to rotations");
    const std::string name =
        op.kind == OpKind::PauliRot ? std::string{"PauliRot"} : op.name;
    PL_ABORT_IF_NOT(op.controlled_wires.empty(),
                    "lightning.cliffish does not support controlled "
                    "operations ('" +
                        name + "' with control wires)");

    std::vector<LocalRotation> rotations;
    std::optional<std::string> word =
        op.kind == OpKind::PauliRot ? std::optional<std::string>{op.pauli_word}
                                    : rotationWord(op.name, op.wires.size());
    std::optional<double> angle{};
    if (word.has_value()) {
        PL_ABORT_IF_NOT(op.params.size() == 1,
                        "'" + name + "' takes exactly one parameter");
        angle = op.params[0];
    } else if (op.name == "T") {
        word = "Z";
        angle = t_gate_angle;
    }

    if (word.has_value()) {
        const auto quarter_turns = cliffordQuarterTurns(*angle);
        if (!quarter_turns.has_value()) {
            rotations.push_back({*word, 0, *angle});
        } else if (*quarter_turns != 0) {
            rotations.push_back({*word, *quarter_turns});
        }
    } else if (const auto it = fixedGateExpansions().find(op.name);
               it != fixedGateExpansions().end()) {
        rotations = it->second;
    } else {
        PL_ABORT("lightning.cliffish does not support the operation '" + name +
                 "'");
    }

    for (const auto &rotation : rotations) {
        PL_ABORT_IF_NOT(
            rotation.pauli.size() == op.wires.size(),
            "'" + name + "' acts on " + std::to_string(rotation.pauli.size()) +
                " wires, but the op names " + std::to_string(op.wires.size()));
    }

    if (op.inverse) {
        std::reverse(rotations.begin(), rotations.end());
        for (auto &rotation : rotations) {
            rotation.quarter_turns =
                static_cast<std::uint8_t>((4 - rotation.quarter_turns) % 4);
            if (rotation.angle.has_value()) {
                rotation.angle = -*rotation.angle;
            }
        }
    }
    return rotations;
}

} // namespace Pennylane::LightningCliffish::Planner
