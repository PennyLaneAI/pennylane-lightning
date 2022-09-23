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

#include "GateImplementationsLMN.hpp"
namespace Pennylane::Gates {

template <class PrecisionT, class ParamT>
void GateImplementationsLMN::applySingleExcitation(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, ParamT angle) {
    PL_ASSERT(wires.size() == 2);
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    /*
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);
    */
    const size_t wires_size = 2;
        std::array<size_t, wires_size> rev_wires_shift;
        std::array<size_t, wires_size + 1> parity;
        revWireParity<wires_size>(num_qubits, wires, rev_wires_shift,
                                       parity);
    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity[2]) |
                           ((k << 1U) & parity[1]) | (k & parity[0]);
        const size_t i10 = i00 | rev_wires_shift[1];
        const size_t i01 = i00 | rev_wires_shift[0];

        const std::complex<PrecisionT> v01 = arr[i01];
        const std::complex<PrecisionT> v10 = arr[i10];

        arr[i01] = c * v01 - s * v10;
        arr[i10] = s * v01 + c * v10;
    }
}

template <class PrecisionT, class ParamT>
void GateImplementationsLMN::applySingleExcitationMinus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, ParamT angle) {
    PL_ASSERT(wires.size() == 2);
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const std::complex<PrecisionT> e =
        inverse ? std::exp(std::complex<PrecisionT>(0, angle / 2))
                : std::exp(-std::complex<PrecisionT>(0, angle / 2));
    /*
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);
    */
    const size_t wires_size = 2;
        std::array<size_t, wires_size> rev_wires_shift;
        std::array<size_t, wires_size + 1> parity;
        revWireParity<wires_size>(num_qubits, wires, rev_wires_shift,
                                       parity);

    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity[2]) |
                           ((k << 1U) & parity[1]) | (k & parity[0]);
        const size_t i10 = i00 | rev_wires_shift[1];
        const size_t i01 = i00 | rev_wires_shift[0];
        const size_t i11 = i00 | rev_wires_shift[0] | rev_wires_shift[1];

        const std::complex<PrecisionT> v01 = arr[i01];
        const std::complex<PrecisionT> v10 = arr[i10];

        arr[i00] *= e;
        arr[i01] = c * v01 - s * v10;
        arr[i10] = s * v01 + c * v10;
        arr[i11] *= e;
    }
}

template <class PrecisionT, class ParamT>
void GateImplementationsLMN::applySingleExcitationPlus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, ParamT angle) {
    PL_ASSERT(wires.size() == 2);
    const PrecisionT c = std::cos(angle / 2);
    const PrecisionT s = inverse ? -std::sin(angle / 2) : std::sin(angle / 2);
    const std::complex<PrecisionT> e =
        inverse ? std::exp(-std::complex<PrecisionT>(0, angle / 2))
                : std::exp(std::complex<PrecisionT>(0, angle / 2));
    /*
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);
    */
    const size_t wires_size = 2;
        std::array<size_t, wires_size> rev_wires_shift;
        std::array<size_t, wires_size + 1> parity;
        revWireParity<wires_size>(num_qubits, wires, rev_wires_shift,
                                       parity);
    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity[2]) |
                           ((k << 1U) & parity[1]) | (k & parity[0]);
        const size_t i10 = i00 | rev_wires_shift[1];
        const size_t i01 = i00 | rev_wires_shift[0];
        const size_t i11 = i00 | rev_wires_shift[0] | rev_wires_shift[1];
        const std::complex<PrecisionT> v01 = arr[i01];
        const std::complex<PrecisionT> v10 = arr[i10];
        arr[i00] *= e;
        arr[i01] = c * v01 - s * v10;
        arr[i10] = s * v01 + c * v10;
        arr[i11] *= e;
    }
}

template <class PrecisionT>
auto GateImplementationsLMN::applyGeneratorSingleExcitation(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 2);
    /*
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);
    */
    const size_t wires_size = 2;
        std::array<size_t, wires_size> rev_wires_shift;
        std::array<size_t, wires_size + 1> parity;
        revWireParity<wires_size>(num_qubits, wires, rev_wires_shift,
                                       parity);
    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity[2]) |
                           ((k << 1U) & parity[1]) | (k & parity[0]);
        const size_t i01 = i00 | rev_wires_shift[0];
        const size_t i10 = i00 | rev_wires_shift[1];
        const size_t i11 = i00 | rev_wires_shift[0] | rev_wires_shift[1];

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
auto GateImplementationsLMN::applyGeneratorSingleExcitationMinus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 2);
    /*
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);
    */
    const size_t wires_size = 2;
        std::array<size_t, wires_size> rev_wires_shift;
        std::array<size_t, wires_size + 1> parity;
        revWireParity<wires_size>(num_qubits, wires, rev_wires_shift,
                                       parity);
    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity[2]) |
                           ((k << 1U) & parity[1]) | (k & parity[0]);
        const size_t i01 = i00 | rev_wires_shift[0];
        const size_t i10 = i00 | rev_wires_shift[1];

        arr[i01] *= Util::IMAG<PrecisionT>();
        arr[i10] *= -Util::IMAG<PrecisionT>();

        std::swap(arr[i10], arr[i01]);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT>
auto GateImplementationsLMN::applyGeneratorSingleExcitationPlus(
    std::complex<PrecisionT> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> PrecisionT {
    PL_ASSERT(wires.size() == 2);
    /*
    const size_t rev_wire0 = num_qubits - wires[1] - 1;
    const size_t rev_wire1 = num_qubits - wires[0] - 1;
    const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
    const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;
    const auto [parity_high, parity_middle, parity_low] =
        revWireParity(rev_wire0, rev_wire1);
    */
    const size_t wires_size = 2;
    std::array<size_t, wires_size> rev_wires_shift;
    std::array<size_t, wires_size + 1> parity;
    revWireParity<wires_size>(num_qubits, wires, rev_wires_shift,
                                       parity);

    for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity[2]) |
                           ((k << 1U) & parity[1]) | (k & parity[0]);
        const size_t i01 = i00 | rev_wires_shift[0];
        const size_t i10 = i00 | rev_wires_shift[1];
        const size_t i11 = i00 | rev_wires_shift[0] | rev_wires_shift[1];

        arr[i00] *= -1;
        arr[i01] *= Util::IMAG<PrecisionT>();
        arr[i10] *= -Util::IMAG<PrecisionT>();
        arr[i11] *= -1;

        std::swap(arr[i10], arr[i01]);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}

// Explicit instantiation starts
/* Matrix operations */
template void GateImplementationsLMN::applySingleQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);

template void GateImplementationsLMN::applySingleQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

template void GateImplementationsLMN::applyTwoQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);

template void GateImplementationsLMN::applyTwoQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsLMN::applyMultiQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);

template void GateImplementationsLMN::applyMultiQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

/* Single-qubit gates */
template void
GateImplementationsLMN::applyIdentity<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
template void
GateImplementationsLMN::applyIdentity<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

template void
GateImplementationsLMN::applyPauliX<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsLMN::applyPauliX<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsLMN::applyPauliY<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsLMN::applyPauliY<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsLMN::applyPauliZ<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsLMN::applyPauliZ<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsLMN::applyHadamard<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
template void
GateImplementationsLMN::applyHadamard<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

template void GateImplementationsLMN::applyS<float>(std::complex<float> *,
                                                   size_t,
                                                   const std::vector<size_t> &,
                                                   bool);
template void GateImplementationsLMN::applyS<double>(std::complex<double> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);

template void GateImplementationsLMN::applyT<float>(std::complex<float> *,
                                                   size_t,
                                                   const std::vector<size_t> &,
                                                   bool);
template void GateImplementationsLMN::applyT<double>(std::complex<double> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);

template void GateImplementationsLMN::applyPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLMN::applyRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLMN::applyRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLMN::applyRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void
GateImplementationsLMN::applyRot<float, float>(std::complex<float> *, size_t,
                                              const std::vector<size_t> &, bool,
                                              float, float, float);
template void
GateImplementationsLMN::applyRot<double, double>(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool, double, double, double);

/* Two-qubit gates */
template void
GateImplementationsLMN::applyCNOT<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
template void
GateImplementationsLMN::applyCNOT<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

template void GateImplementationsLMN::applyCY<float>(std::complex<float> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);
template void
GateImplementationsLMN::applyCY<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

template void GateImplementationsLMN::applyCZ<float>(std::complex<float> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);
template void
GateImplementationsLMN::applyCZ<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

template void
GateImplementationsLMN::applySWAP<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
template void
GateImplementationsLMN::applySWAP<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

template void
GateImplementationsLMN::applyCSWAP<float>(std::complex<float> *, size_t,
                                         const std::vector<size_t> &, bool);
template void
GateImplementationsLMN::applyCSWAP<double>(std::complex<double> *, size_t,
                                          const std::vector<size_t> &, bool);

template void
GateImplementationsLMN::applyToffoli<float>(std::complex<float> *, size_t,
                                           const std::vector<size_t> &, bool);
template void
GateImplementationsLMN::applyToffoli<double>(std::complex<double> *, size_t,
                                            const std::vector<size_t> &, bool);

template void GateImplementationsLMN::applyIsingXX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyIsingXX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLMN::applyIsingXY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyIsingXY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLMN::applyIsingYY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyIsingYY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLMN::applyIsingZZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyIsingZZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLMN::applyControlledPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyControlledPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLMN::applyCRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyCRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLMN::applyCRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyCRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLMN::applyCRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyCRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void
GateImplementationsLMN::applyCRot<float, float>(std::complex<float> *, size_t,
                                               const std::vector<size_t> &,
                                               bool, float, float, float);
template void
GateImplementationsLMN::applyCRot<double, double>(std::complex<double> *, size_t,
                                                 const std::vector<size_t> &,
                                                 bool, double, double, double);

template void GateImplementationsLMN::applyMultiRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLMN::applyMultiRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

/* QChem functions */
template void GateImplementationsLMN::applySingleExcitation<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLMN::applySingleExcitation<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

template void GateImplementationsLMN::applySingleExcitationMinus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLMN::applySingleExcitationMinus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

template void GateImplementationsLMN::applySingleExcitationPlus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLMN::applySingleExcitationPlus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

template void GateImplementationsLMN::applyDoubleExcitation<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLMN::applyDoubleExcitation<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

template void GateImplementationsLMN::applyDoubleExcitationMinus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLMN::applyDoubleExcitationMinus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

template void GateImplementationsLMN::applyDoubleExcitationPlus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLMN::applyDoubleExcitationPlus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

// Generators
template auto PauliGenerator<GateImplementationsLMN>::applyGeneratorRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsLMN>::applyGeneratorRX(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

template auto PauliGenerator<GateImplementationsLMN>::applyGeneratorRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsLMN>::applyGeneratorRY(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
template auto PauliGenerator<GateImplementationsLMN>::applyGeneratorRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsLMN>::applyGeneratorRZ(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLMN::applyGeneratorPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLMN::applyGeneratorPhaseShift(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool) -> double;

template auto GateImplementationsLMN::applyGeneratorCRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLMN::applyGeneratorCRX(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

template auto GateImplementationsLMN::applyGeneratorCRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLMN::applyGeneratorCRY(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

template auto GateImplementationsLMN::applyGeneratorCRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLMN::applyGeneratorCRZ(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

template auto
GateImplementationsLMN::applyGeneratorIsingXX(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLMN::applyGeneratorIsingXX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

template auto
GateImplementationsLMN::applyGeneratorIsingXY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLMN::applyGeneratorIsingXY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

template auto GateImplementationsLMN::applyGeneratorIsingYY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLMN::applyGeneratorIsingYY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;

template auto
GateImplementationsLMN::applyGeneratorIsingZZ(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLMN::applyGeneratorIsingZZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

template auto GateImplementationsLMN::applyGeneratorControlledPhaseShift(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLMN::applyGeneratorControlledPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

template auto
GateImplementationsLMN::applyGeneratorMultiRZ(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLMN::applyGeneratorMultiRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

/* QChem */
template auto GateImplementationsLMN::applyGeneratorSingleExcitation<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;

template auto GateImplementationsLMN::applyGeneratorSingleExcitation<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;

template auto GateImplementationsLMN::applyGeneratorSingleExcitationMinus<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;

template auto
GateImplementationsLMN::applyGeneratorSingleExcitationMinus<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;

template auto GateImplementationsLMN::applyGeneratorSingleExcitationPlus<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;

template auto GateImplementationsLMN::applyGeneratorSingleExcitationPlus<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;

// Explicit instantiations ends

} // namespace Pennylane::Gates
