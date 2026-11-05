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
 * @file Observables.hpp
 * Observables as sums of Pauli words.
 */

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "Program.hpp"

namespace Pennylane::LightningCliffish::Planner {

/**
 * @brief coefficient times the Pauli word with pauli[k] on wires[k]. An empty
 * word is the identity.
 */
struct PauliTerm {
    double coefficient;
    std::string pauli;
    std::vector<Program::QubitId> wires;
};

/**
 * @brief A real linear combination of Pauli words.
 *
 * Named Pauli observables, their tensor products and Hamiltonians made of
 * them all lower to this form.
 */
using Observable = std::vector<PauliTerm>;

/// The observable coefficient * (pauli[k] on wires[k]).
inline auto pauliObservable(std::string pauli,
                            std::vector<Program::QubitId> wires,
                            double coefficient = 1.0) -> Observable {
    return {PauliTerm{coefficient, std::move(pauli), std::move(wires)}};
}

} // namespace Pennylane::LightningCliffish::Planner
