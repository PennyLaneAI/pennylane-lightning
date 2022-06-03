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
/**
 * @file
 * Defines qchem kernel functions with generators
 */
#pragma once

#include "cpu_kernels/GateImplementationsLM.hpp"
#include "cpu_kernels/GateImplementationsPI.hpp"

namespace Pennylane::Gates {
template <class PrecisionT, class ParamT>
void GateImplementationsLM::applySingleExcitation(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, ParamT angle) {
    PL_ASSERT(wires.size() == 2);
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);

    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;

        const std::complex<PrecisionT> v01 = arr[i01];
        const std::complex<PrecisionT> v10 = arr[i10];

        arr[i01] = c * v01 - s * v10;
        arr[i10] = s * v01 + c * v10;
    }
}

template <class PrecisionT, class ParamT>
void GateImplementationsLM::applySingleExcitationMinus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, ParamT angle) {
    PL_ASSERT(wires.size() == 2);
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const std::complex<PrecisionT> e =
        inverse ? std::exp(std::complex<PrecisionT>(0, angle / 2))
                : std::exp(-std::complex<PrecisionT>(0, angle / 2));
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);

    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        const std::complex<PrecisionT> v01 = arr[i01];
        const std::complex<PrecisionT> v10 = arr[i10];

        arr[i00] *= e;
        arr[i01] = c * v01 - s * v10;
        arr[i10] = s * v01 + c * v10;
        arr[i11] *= e;
    }
}

template <class PrecisionT, class ParamT>
void GateImplementationsLM::applySingleExcitationPlus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, ParamT angle) {
    PL_ASSERT(wires.size() == 2);
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const std::complex<PrecisionT> e =
        inverse ? std::exp(-std::complex<PrecisionT>(0, angle / 2))
                : std::exp(std::complex<PrecisionT>(0, angle / 2));
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);

    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;
        const std::complex<PrecisionT> v01 = arr[i01];
        const std::complex<PrecisionT> v10 = arr[i10];
        arr[i00] *= e;
        arr[i01] = c * v01 - s * v10;
        arr[i10] = s * v01 + c * v10;
        arr[i11] *= e;
    }
}

template <class PrecisionT>
auto GateImplementationsLM::applyGeneratorSingleExcitation(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 2);
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);

    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] = std::complex<PrecisionT>{};
        arr[i01] *= Util::IMAG<PrecisionT>();
        arr[i10] *= -Util::IMAG<PrecisionT>();
        arr[i11] = std::complex<PrecisionT>{};

        std::swap(arr[i10], arr[i01]);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT>
auto GateImplementationsLM::applyGeneratorSingleExcitationMinus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 2);
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);

    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i10 = i00 | rev_wire1_shift;

        arr[i01] *= Util::IMAG<PrecisionT>();
        arr[i10] *= -Util::IMAG<PrecisionT>();

        std::swap(arr[i10], arr[i01]);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT>
auto GateImplementationsLM::applyGeneratorSingleExcitationPlus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 2);
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);

    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] *= -1;
        arr[i01] *= Util::IMAG<PrecisionT>();
        arr[i10] *= -Util::IMAG<PrecisionT>();
        arr[i11] *= -1;

        std::swap(arr[i10], arr[i01]);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT, class ParamT>
void GateImplementationsPI::applyDoubleExcitation(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool inverse,
    ParamT angle) {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);

    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;
        const std::complex<PrecisionT> v3 = shiftedState[indices[i0]];
        const std::complex<PrecisionT> v12 = shiftedState[indices[i1]];

        shiftedState[indices[i0]] = c * v3 - s * v12;
        shiftedState[indices[i1]] = s * v3 + c * v12;
    }
}

template <class PrecisionT, class ParamT>
void GateImplementationsPI::applyDoubleExcitationMinus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool inverse,
    ParamT angle) {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const std::complex<PrecisionT> e =
        inverse ? std::exp(std::complex<PrecisionT>(0, angle / 2))
                : std::exp(-std::complex<PrecisionT>(0, angle / 2));

    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;
        // NOLINTNEXTLINE(readability-magic-numbers)
        const std::complex<PrecisionT> v3 = shiftedState[indices[i0]];
        // NOLINTNEXTLINE(readability-magic-numbers)
        const std::complex<PrecisionT> v12 = shiftedState[indices[i1]];
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[0]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[1]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[2]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[i0]] = c * v3 - s * v12;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[4]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[5]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[6]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[7]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[8]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[9]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[10]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[11]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[i1]] = s * v3 + c * v12;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[13]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[14]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[15]] *= e;
    }
}

template <class PrecisionT, class ParamT>
void GateImplementationsPI::applyDoubleExcitationPlus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool inverse,
    ParamT angle) {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const std::complex<PrecisionT> e =
        inverse ? std::exp(-std::complex<PrecisionT>(0, angle / 2))
                : std::exp(std::complex<PrecisionT>(0, angle / 2));
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;
        // NOLINTNEXTLINE(readability-magic-numbers)
        const std::complex<PrecisionT> v3 = shiftedState[indices[i0]];
        // NOLINTNEXTLINE(readability-magic-numbers)
        const std::complex<PrecisionT> v12 = shiftedState[indices[i1]];
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[0]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[1]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[2]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[i0]] = c * v3 - s * v12;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[4]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[5]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[6]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[7]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[8]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[9]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[10]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[11]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[i1]] = s * v3 + c * v12;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[13]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[14]] *= e;
        // NOLINTNEXTLINE(readability-magic-numbers)
        shiftedState[indices[15]] *= e;
    }
}

template <class PrecisionT>
auto GateImplementationsPI::applyGeneratorDoubleExcitation(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;
        const std::complex<PrecisionT> v3 = shiftedState[indices[i0]];
        const std::complex<PrecisionT> v12 = shiftedState[indices[i1]];
        for (const size_t &i : indices) {
            shiftedState[i] = std::complex<PrecisionT>{};
        }

        shiftedState[indices[i0]] = -v12 * Util::IMAG<PrecisionT>();
        shiftedState[indices[i1]] = v3 * Util::IMAG<PrecisionT>();
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT>
auto GateImplementationsPI::applyGeneratorDoubleExcitationMinus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;

        shiftedState[indices[i0]] *= Util::IMAG<PrecisionT>();
        shiftedState[indices[i1]] *= -Util::IMAG<PrecisionT>();

        std::swap(shiftedState[indices[i0]], shiftedState[indices[i1]]);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT>
auto GateImplementationsPI::applyGeneratorDoubleExcitationPlus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 4);
    const auto [indices, externalIndices] = GateIndices(wires, num_qubits);

    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i0 = 3;
    // NOLINTNEXTLINE(readability-magic-numbers)
    const size_t i1 = 12;

    for (const size_t &externalIndex : externalIndices) {
        std::complex<PrecisionT> *shiftedState = arr + externalIndex;
        for (const size_t &i : indices) {
            shiftedState[i] *= -1;
        }

        shiftedState[indices[i0]] *= -Util::IMAG<PrecisionT>();
        shiftedState[indices[i1]] *= Util::IMAG<PrecisionT>();

        std::swap(shiftedState[indices[i0]], shiftedState[indices[i1]]);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}
} // namespace Pennylane::Gates
