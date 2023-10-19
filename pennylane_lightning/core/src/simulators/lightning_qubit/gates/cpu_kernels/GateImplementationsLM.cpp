// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "GateImplementationsLM.hpp"
#include "Util.hpp" // exp2

/// @cond DEV
namespace {
using Pennylane::Util::exp2;
} // namespace
/// @endcond

namespace Pennylane::LightningQubit::Gates {
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

    for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
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

    for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
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

    for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
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

    for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] = std::complex<PrecisionT>{};
        arr[i01] *= Pennylane::Util::IMAG<PrecisionT>();
        arr[i10] *= -Pennylane::Util::IMAG<PrecisionT>();
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

    for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i10 = i00 | rev_wire1_shift;

        arr[i01] *= Pennylane::Util::IMAG<PrecisionT>();
        arr[i10] *= -Pennylane::Util::IMAG<PrecisionT>();

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

    for (size_t k = 0; k < exp2(num_qubits - 2); k++) {
        const size_t i00 = ((k << 2U) & parity_high) |
                           ((k << 1U) & parity_middle) | (k & parity_low);
        const size_t i01 = i00 | rev_wire0_shift;
        const size_t i10 = i00 | rev_wire1_shift;
        const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;

        arr[i00] *= -1;
        arr[i01] *= Pennylane::Util::IMAG<PrecisionT>();
        arr[i10] *= -Pennylane::Util::IMAG<PrecisionT>();
        arr[i11] *= -1;

        std::swap(arr[i10], arr[i01]);
    }
    // NOLINTNEXTLINE(readability-magic-numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT>
auto GateImplementationsLM::applyGeneratorDoubleExcitation(
    std::complex<PrecisionT> *arr, std::size_t num_qubits,
    const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
    -> PrecisionT {
    using ComplexT = std::complex<PrecisionT>;
    PL_ASSERT(wires.size() == 4);
    constexpr ComplexT zero{};
    constexpr ComplexT imag{0, 1};
    constexpr std::size_t one{1};

    const std::array<std::size_t, 4> rev_wires{
        num_qubits - wires[3] - 1, num_qubits - wires[2] - 1,
        num_qubits - wires[1] - 1, num_qubits - wires[0] - 1};

    const std::array<std::size_t, 4> rev_wire_shifts{
        one << rev_wires[0], one << rev_wires[1], one << rev_wires[2],
        one << rev_wires[3]};

    const auto parity = Pennylane::Util::revWireParity(rev_wires);

    for (std::size_t k = 0; k < exp2(num_qubits - 4); k++) {
        const auto indices =
            GateImplementationsLM::parity2indices(k, parity, rev_wire_shifts);
        const ComplexT v3 = arr[indices[0B0011]];
        const ComplexT v12 = arr[indices[0B1100]];
        for (const auto &i : indices) {
            arr[i] = zero;
        }
        arr[indices[0B0011]] = -v12 * imag;
        arr[indices[0B1100]] = v3 * imag;
    }
    // NOLINTNEXTLINE(readability - magic - numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT>
auto GateImplementationsLM::applyGeneratorDoubleExcitationMinus(
    std::complex<PrecisionT> *arr, std::size_t num_qubits,
    const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
    -> PrecisionT {
    using ComplexT = std::complex<PrecisionT>;
    PL_ASSERT(wires.size() == 4);
    constexpr ComplexT imag{0, 1};
    constexpr std::size_t one{1};

    const std::array<std::size_t, 4> rev_wires{
        num_qubits - wires[3] - 1, num_qubits - wires[2] - 1,
        num_qubits - wires[1] - 1, num_qubits - wires[0] - 1};

    const std::array<std::size_t, 4> rev_wire_shifts{
        one << rev_wires[0], one << rev_wires[1], one << rev_wires[2],
        one << rev_wires[3]};

    const auto parity = Pennylane::Util::revWireParity(rev_wires);

    for (std::size_t k = 0; k < exp2(num_qubits - 4); k++) {
        const std::size_t i0000 =
            ((k << 4U) & parity[4]) | ((k << 3U) & parity[3]) |
            ((k << 2U) & parity[2]) | ((k << 1U) & parity[1]) | (k & parity[0]);
        const std::size_t i0011 =
            i0000 | rev_wire_shifts[1] | rev_wire_shifts[0];
        const std::size_t i1100 =
            i0000 | rev_wire_shifts[3] | rev_wire_shifts[2];

        arr[i0011] *= imag;
        arr[i1100] *= -imag;
        swap(arr[i1100], arr[i0011]);
    }
    // NOLINTNEXTLINE(readability - magic - numbers)
    return -static_cast<PrecisionT>(0.5);
}

template <class PrecisionT>
auto GateImplementationsLM::applyGeneratorDoubleExcitationPlus(
    std::complex<PrecisionT> *arr, std::size_t num_qubits,
    const std::vector<std::size_t> &wires, [[maybe_unused]] bool adj)
    -> PrecisionT {
    using ComplexT = std::complex<PrecisionT>;
    PL_ASSERT(wires.size() == 4);
    constexpr ComplexT imag{0, 1};
    constexpr std::size_t one{1};

    const std::array<std::size_t, 4> rev_wires{
        num_qubits - wires[3] - 1, num_qubits - wires[2] - 1,
        num_qubits - wires[1] - 1, num_qubits - wires[0] - 1};

    const std::array<std::size_t, 4> rev_wire_shifts{
        one << rev_wires[0], one << rev_wires[1], one << rev_wires[2],
        one << rev_wires[3]};

    const auto parity = Pennylane::Util::revWireParity(rev_wires);

    for (std::size_t k = 0; k < exp2(num_qubits - 4); k++) {
        const std::size_t i0000 =
            ((k << 4U) & parity[4]) | ((k << 3U) & parity[3]) |
            ((k << 2U) & parity[2]) | ((k << 1U) & parity[1]) | (k & parity[0]);
        const std::size_t i0011 =
            i0000 | rev_wire_shifts[1] | rev_wire_shifts[0];
        const std::size_t i1100 =
            i0000 | rev_wire_shifts[3] | rev_wire_shifts[2];

        arr[i0011] *= -imag;
        arr[i1100] *= imag;
        swap(arr[i1100], arr[i0011]);
    }
    // NOLINTNEXTLINE(readability - magic - numbers)
    return static_cast<PrecisionT>(0.5);
}

// Explicit instantiation starts
/* Matrix operations */
template void GateImplementationsLM::applySingleQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);

template void GateImplementationsLM::applySingleQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

template void GateImplementationsLM::applyTwoQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);

template void GateImplementationsLM::applyTwoQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);
template void GateImplementationsLM::applyMultiQubitOp<float>(
    std::complex<float> *, size_t, const std::complex<float> *,
    const std::vector<size_t> &, bool);

template void GateImplementationsLM::applyMultiQubitOp<double>(
    std::complex<double> *, size_t, const std::complex<double> *,
    const std::vector<size_t> &, bool);

/* Single-qubit gates */
template void
GateImplementationsLM::applyIdentity<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyIdentity<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyPauliX<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyPauliX<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyPauliY<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyPauliY<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyPauliZ<float>(std::complex<float> *, size_t,
                                          const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyPauliZ<double>(std::complex<double> *, size_t,
                                           const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyHadamard<float>(std::complex<float> *, size_t,
                                            const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyHadamard<double>(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool);

template void GateImplementationsLM::applyS<float>(std::complex<float> *,
                                                   size_t,
                                                   const std::vector<size_t> &,
                                                   bool);
template void GateImplementationsLM::applyS<double>(std::complex<double> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);

template void GateImplementationsLM::applyT<float>(std::complex<float> *,
                                                   size_t,
                                                   const std::vector<size_t> &,
                                                   bool);
template void GateImplementationsLM::applyT<double>(std::complex<double> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);

template void GateImplementationsLM::applyPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void
GateImplementationsLM::applyRot<float, float>(std::complex<float> *, size_t,
                                              const std::vector<size_t> &, bool,
                                              float, float, float);
template void
GateImplementationsLM::applyRot<double, double>(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool, double, double, double);

/* Two-qubit gates */
template void
GateImplementationsLM::applyCNOT<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyCNOT<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

template void GateImplementationsLM::applyCY<float>(std::complex<float> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);
template void
GateImplementationsLM::applyCY<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

template void GateImplementationsLM::applyCZ<float>(std::complex<float> *,
                                                    size_t,
                                                    const std::vector<size_t> &,
                                                    bool);
template void
GateImplementationsLM::applyCZ<double>(std::complex<double> *, size_t,
                                       const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applySWAP<float>(std::complex<float> *, size_t,
                                        const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applySWAP<double>(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyCSWAP<float>(std::complex<float> *, size_t,
                                         const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyCSWAP<double>(std::complex<double> *, size_t,
                                          const std::vector<size_t> &, bool);

template void
GateImplementationsLM::applyToffoli<float>(std::complex<float> *, size_t,
                                           const std::vector<size_t> &, bool);
template void
GateImplementationsLM::applyToffoli<double>(std::complex<double> *, size_t,
                                            const std::vector<size_t> &, bool);

template void GateImplementationsLM::applyIsingXX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyIsingXX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyIsingXY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyIsingXY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyIsingYY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyIsingYY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyIsingZZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyIsingZZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyControlledPhaseShift<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyControlledPhaseShift<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyCRX<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyCRX<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyCRY<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyCRY<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void GateImplementationsLM::applyCRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyCRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

template void
GateImplementationsLM::applyCRot<float, float>(std::complex<float> *, size_t,
                                               const std::vector<size_t> &,
                                               bool, float, float, float);
template void
GateImplementationsLM::applyCRot<double, double>(std::complex<double> *, size_t,
                                                 const std::vector<size_t> &,
                                                 bool, double, double, double);

template void GateImplementationsLM::applyMultiRZ<float, float>(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool, float);
template void GateImplementationsLM::applyMultiRZ<double, double>(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool, double);

/* QChem functions */
template void GateImplementationsLM::applySingleExcitation<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLM::applySingleExcitation<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

template void GateImplementationsLM::applySingleExcitationMinus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLM::applySingleExcitationMinus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

template void GateImplementationsLM::applySingleExcitationPlus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLM::applySingleExcitationPlus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

template void GateImplementationsLM::applyDoubleExcitation<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLM::applyDoubleExcitation<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

template void GateImplementationsLM::applyDoubleExcitationMinus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLM::applyDoubleExcitationMinus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

template void GateImplementationsLM::applyDoubleExcitationPlus<float, float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, float angle);

template void GateImplementationsLM::applyDoubleExcitationPlus<double, double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, bool inverse, double angle);

// Generators
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRX(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;

template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRY(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto PauliGenerator<GateImplementationsLM>::applyGeneratorRZ(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorPhaseShift(std::complex<double> *, size_t,
                                                const std::vector<size_t> &,
                                                bool) -> double;

template auto GateImplementationsLM::applyGeneratorCRX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorCRX(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

template auto GateImplementationsLM::applyGeneratorCRY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorCRY(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

template auto GateImplementationsLM::applyGeneratorCRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorCRZ(std::complex<double> *, size_t,
                                         const std::vector<size_t> &, bool)
    -> double;

template auto
GateImplementationsLM::applyGeneratorIsingXX(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorIsingXX(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

template auto
GateImplementationsLM::applyGeneratorIsingXY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorIsingXY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

template auto GateImplementationsLM::applyGeneratorIsingYY(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;
template auto
GateImplementationsLM::applyGeneratorIsingYY(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;

template auto
GateImplementationsLM::applyGeneratorIsingZZ(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorIsingZZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

template auto GateImplementationsLM::applyGeneratorControlledPhaseShift(
    std::complex<double> *, size_t, const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorControlledPhaseShift(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

template auto
GateImplementationsLM::applyGeneratorMultiRZ(std::complex<double> *, size_t,
                                             const std::vector<size_t> &, bool)
    -> double;
template auto GateImplementationsLM::applyGeneratorMultiRZ(
    std::complex<float> *, size_t, const std::vector<size_t> &, bool) -> float;

/* QChem */
template auto GateImplementationsLM::applyGeneratorSingleExcitation<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;

template auto GateImplementationsLM::applyGeneratorSingleExcitation<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;

template auto GateImplementationsLM::applyGeneratorSingleExcitationMinus<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;

template auto
GateImplementationsLM::applyGeneratorSingleExcitationMinus<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;

template auto GateImplementationsLM::applyGeneratorSingleExcitationPlus<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;

template auto GateImplementationsLM::applyGeneratorSingleExcitationPlus<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;

template auto GateImplementationsLM::applyGeneratorDoubleExcitation<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;

template auto GateImplementationsLM::applyGeneratorDoubleExcitation<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;

template auto GateImplementationsLM::applyGeneratorDoubleExcitationMinus<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;

template auto
GateImplementationsLM::applyGeneratorDoubleExcitationMinus<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;

template auto GateImplementationsLM::applyGeneratorDoubleExcitationPlus<float>(
    std::complex<float> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> float;

template auto GateImplementationsLM::applyGeneratorDoubleExcitationPlus<double>(
    std::complex<double> *arr, size_t num_qubits,
    const std::vector<size_t> &wires, [[maybe_unused]] bool adj) -> double;

// Explicit instantiations ends

} // namespace Pennylane::LightningQubit::Gates
