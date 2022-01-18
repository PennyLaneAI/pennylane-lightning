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
 * @file GateOperationsLM.hpp
 * Defines kernel functions with less memory (and fast)
 */
#pragma once

#include "Error.hpp"
#include "GateOperations.hpp"
#include "Gates.hpp"
#include "KernelType.hpp"
#include "Util.hpp"

#include <climits>
#include <complex>
#include <vector>

namespace Pennylane {
/**
 * @brief Fill ones from LSB to rev_wire
 */
auto constexpr fillTrailingOnes(size_t pos) -> size_t {
    return (pos == 0) ? 0 : (~size_t(0) >> (CHAR_BIT * sizeof(size_t) - pos));
}
/**
 * @brief Fill ones from MSB to pos
 */
auto constexpr fillLeadingOnes(size_t pos) -> size_t {
    return (~size_t(0)) << pos;
}

/**
 * @brief A gate operation implementation with less memory.
 *
 * We use a bitwise operation to calculate the indices where the gate
 * applies to on the fly.
 *
 * @tparam fp_t Floating point precision of underlying statevector data
 */
template <class fp_t> class GateOperationsLM {
  public:
    using scalar_type_t = fp_t;
    using CFP_t = std::complex<fp_t>;

    constexpr static KernelType kernel_id = KernelType::LM;

    constexpr static std::array implemented_gates = {
        GateOperations::PauliX,
        GateOperations::PauliY,
        GateOperations::PauliZ,
        GateOperations::Hadamard,
        GateOperations::S,
        GateOperations::T,
        GateOperations::RX,
        GateOperations::RY,
        GateOperations::RZ,
        GateOperations::PhaseShift,
        GateOperations::Rot,
        GateOperations::CZ,
        GateOperations::CNOT,
        GateOperations::SWAP,
        GateOperations::GeneratorPhaseShift};

  private:
    static inline void applySingleQubitOp(CFP_t *arr, size_t num_qubits,
                                          const CFP_t *op_matrix, size_t wire) {
        const size_t rev_wire = num_qubits - wire - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        for (size_t n = 0; n < Util::exp2(num_qubits - 1); n++) {
            const size_t k = n;
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            const CFP_t v0 = arr[i0];
            const CFP_t v1 = arr[i1];
            arr[i0] = op_matrix[0B00] * v0 +
                      op_matrix[0B01] * v1; // NOLINT(readability-magic-numbers)
            arr[i1] = op_matrix[0B10] * v0 +
                      op_matrix[0B11] * v1; // NOLINT(readability-magic-numbers)
        }
    }

  public:
    static void applyMatrix(CFP_t *arr, size_t num_qubits, const CFP_t *matrix,
                            const std::vector<size_t> &wires, bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(matrix);
        static_cast<void>(wires);
        static_cast<void>(inverse);
        PL_ABORT("Called unimplemented gate operation applyMatrix.");
    }

    static void applyPauliX(CFP_t *arr, const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            std::swap(arr[i0], arr[i1]);
        }
    }

    static void applyPauliY(CFP_t *arr, const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            const auto v0 = arr[i0];
            const auto v1 = arr[i1];
            arr[i0] = {std::imag(v1), -std::real(v1)};
            arr[i1] = {-std::imag(v0), std::real(v0)};
        }
    }

    static void applyPauliZ(CFP_t *arr, const size_t num_qubits,
                            const std::vector<size_t> &wires,
                            [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            arr[i1] *= -1;
        }
    }

    static void applyHadamard(CFP_t *arr, const size_t num_qubits,
                              const std::vector<size_t> &wires,
                              [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        constexpr fp_t isqrt2 = Util::INVSQRT2<fp_t>();
        constexpr static std::array<CFP_t, 4> hadamardMat = {isqrt2, isqrt2,
                                                             isqrt2, -isqrt2};
        applySingleQubitOp(arr, num_qubits, hadamardMat.data(), wires[0]);
    }

    static void applyS(CFP_t *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const CFP_t shift =
            (inverse) ? -Util::IMAG<fp_t>() : Util::IMAG<fp_t>();

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            arr[i1] *= shift;
        }
    }

    static void applyT(CFP_t *arr, const size_t num_qubits,
                       const std::vector<size_t> &wires,
                       [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const CFP_t shift =
            (inverse)
                ? std::conj(std::exp(CFP_t(0, static_cast<fp_t>(M_PI / 4))))
                : std::exp(CFP_t(0, static_cast<fp_t>(M_PI / 4)));

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            arr[i1] *= shift;
        }
    }

    template <class Param_t = fp_t>
    static void applyRX(CFP_t *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        Param_t angle) {
        assert(wires.size() == 1);

        const fp_t c = std::cos(angle / 2);
        const fp_t js =
            (inverse) ? -std::sin(-angle / 2) : std::sin(-angle / 2);

        const std::array<CFP_t, 4> RXMat = {c, Util::IMAG<fp_t>() * js,
                                            Util::IMAG<fp_t>() * js, c};
        applySingleQubitOp(arr, num_qubits, RXMat.data(), wires[0]);
    }

    template <class Param_t = fp_t>
    static void applyRY(CFP_t *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        Param_t angle) {
        assert(wires.size() == 1);

        const fp_t c = std::cos(angle / 2);
        const fp_t s = (inverse) ? -std::sin(angle / 2) : std::sin(angle / 2);

        const std::array<CFP_t, 4> RYMat = {c, -s, s, c};
        applySingleQubitOp(arr, num_qubits, RYMat.data(), wires[0]);
    }

    template <class Param_t = fp_t>
    static void applyRZ(CFP_t *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires, bool inverse,
                        Param_t angle) {
        assert(wires.size() == 1);

        const CFP_t first = CFP_t{std::cos(angle / 2), -std::sin(angle / 2)};
        const CFP_t second = CFP_t{std::cos(angle / 2), std::sin(angle / 2)};

        const std::array<CFP_t, 2> shifts = {
            (inverse) ? std::conj(first) : first,
            (inverse) ? std::conj(second) : second};
        const size_t rev_wire = num_qubits - wires[0] - 1;

        for (size_t k = 0; k < Util::exp2(num_qubits); k++) {
            arr[k] *= shifts[(k >> rev_wire) & 1U];
        }
    }

    template <class Param_t = fp_t>
    static void applyPhaseShift(CFP_t *arr, const size_t num_qubits,
                                const std::vector<size_t> &wires, bool inverse,
                                Param_t angle) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t rev_wire_shift = (static_cast<size_t>(1U) << rev_wire);
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        const CFP_t s =
            inverse ? std::exp(-CFP_t(0, angle)) : std::exp(CFP_t(0, angle));

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            const size_t i1 = i0 | rev_wire_shift;
            arr[i1] *= s;
        }
    }

    template <class Param_t = fp_t>
    static void applyRot(CFP_t *arr, const size_t num_qubits,
                         const std::vector<size_t> &wires, bool inverse,
                         Param_t phi, Param_t theta, Param_t omega) {
        assert(wires.size() == 1);

        const auto rotMat = (inverse)
                                ? Gates::getRot<fp_t>(-omega, -theta, -phi)
                                : Gates::getRot<fp_t>(phi, theta, omega);

        applySingleQubitOp(arr, num_qubits, rotMat.data(), wires[0]);
    }

    static void applyCNOT(CFP_t *arr, const size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Controll qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        /* This is faster than iterate over all indices */
        for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i11 = i00 | rev_wire1_shift | rev_wire0_shift;

            std::swap(arr[i10], arr[i11]);
        }
    }

    static void applyCY(CFP_t *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        PL_ABORT("GaterOperationsLM::applyCY is not implemented yet");
    }

    static void applyCZ(CFP_t *arr, const size_t num_qubits,
                        const std::vector<size_t> &wires,
                        [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Controll qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        /* This is faster than iterate over all indices */
        for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i11 = i00 | rev_wire0_shift | rev_wire1_shift;
            arr[i11] *= -1;
        }
    }

    static void applySWAP(CFP_t *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse) {
        assert(wires.size() == 2);

        const size_t rev_wire0 = num_qubits - wires[1] - 1;
        const size_t rev_wire1 = num_qubits - wires[0] - 1; // Controll qubit

        const size_t rev_wire0_shift = static_cast<size_t>(1U) << rev_wire0;
        const size_t rev_wire1_shift = static_cast<size_t>(1U) << rev_wire1;

        const size_t rev_wire_min = std::min(rev_wire0, rev_wire1);
        const size_t rev_wire_max = std::max(rev_wire0, rev_wire1);

        const size_t parity_low = fillTrailingOnes(rev_wire_min);
        const size_t parity_high = fillLeadingOnes(rev_wire_max + 1);
        const size_t parity_middle =
            fillLeadingOnes(rev_wire_min + 1) & fillTrailingOnes(rev_wire_max);

        for (size_t k = 0; k < Util::exp2(num_qubits - 2); k++) {
            const size_t i00 = ((k << 2U) & parity_high) |
                               ((k << 1U) & parity_middle) | (k & parity_low);
            const size_t i10 = i00 | rev_wire1_shift;
            const size_t i01 = i00 | rev_wire0_shift;
            std::swap(arr[i10], arr[i01]);
        }
    }

    template <class Param_t = fp_t>
    static void applyControlledPhaseShift(CFP_t *arr, size_t num_qubits,
                                          const std::vector<size_t> &wires,
                                          [[maybe_unused]] bool inverse,
                                          Param_t angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        static_cast<void>(angle);
        PL_ABORT("GaterOperationsLM::applyControlledPhaseShift is not "
                 "implemented yet");
    }

    template <class Param_t = fp_t>
    static void applyCRX(CFP_t *arr, size_t num_qubits,
                         const std::vector<size_t> &wires,
                         [[maybe_unused]] bool inverse, Param_t angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        static_cast<void>(angle);
        PL_ABORT("GaterOperationsLM::applyCRX is not implemented yet");
    }

    template <class Param_t = fp_t>
    static void applyCRY(CFP_t *arr, size_t num_qubits,
                         const std::vector<size_t> &wires,
                         [[maybe_unused]] bool inverse, Param_t angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        static_cast<void>(angle);
        PL_ABORT("GaterOperationsLM::applyCRY is not implemented yet");
    }

    template <class Param_t = fp_t>
    static void applyCRZ(CFP_t *arr, size_t num_qubits,
                         const std::vector<size_t> &wires,
                         [[maybe_unused]] bool inverse, Param_t angle) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        static_cast<void>(angle);
        PL_ABORT("GaterOperationsLM::applyCRZ is not implemented yet");
    }

    template <class Param_t = fp_t>
    static void applyCRot(CFP_t *arr, size_t num_qubits,
                          const std::vector<size_t> &wires,
                          [[maybe_unused]] bool inverse, Param_t phi,
                          Param_t theta, Param_t omega) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        static_cast<void>(phi);
        static_cast<void>(theta);
        static_cast<void>(omega);
        PL_ABORT("GaterOperationsLM::applyCRot is not implemented yet");
    }

    static void applyToffoli(CFP_t *arr, size_t num_qubits,
                             const std::vector<size_t> &wires,
                             [[maybe_unused]] bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        PL_ABORT("GaterOperationsLM::applyTofolli is not implemented yet");
    }

    static void applyCSWAP(CFP_t *arr, size_t num_qubits,
                           const std::vector<size_t> &wires,
                           [[maybe_unused]] bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        PL_ABORT("GaterOperationsLM::applyCSWAP is not implemented yet");
    }

    static void applyGeneratorPhaseShift(CFP_t *arr, size_t num_qubits,
                                         const std::vector<size_t> &wires,
                                         [[maybe_unused]] bool inverse) {
        assert(wires.size() == 1);
        const size_t rev_wire = num_qubits - wires[0] - 1;
        const size_t wire_parity = fillTrailingOnes(rev_wire);
        const size_t wire_parity_inv = fillLeadingOnes(rev_wire + 1);

        for (size_t k = 0; k < Util::exp2(num_qubits - 1); k++) {
            const size_t i0 = ((k << 1U) & wire_parity_inv) | (wire_parity & k);
            arr[i0] = Util::ZERO<fp_t>();
        }
    }

    static void applyGeneratorCRX(CFP_t *arr, size_t num_qubits,
                                  const std::vector<size_t> &wires,
                                  [[maybe_unused]] bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        PL_ABORT("GaterOperationsLM::applyGeneratorCRX is not implemented yet");
    }

    static void applyGeneratorCRY(CFP_t *arr, size_t num_qubits,
                                  const std::vector<size_t> &wires,
                                  [[maybe_unused]] bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        PL_ABORT("GaterOperationsLM::applyGeneratorCRY is not implemented yet");
    }
    static void applyGeneratorCRZ(CFP_t *arr, size_t num_qubits,
                                  const std::vector<size_t> &wires,
                                  [[maybe_unused]] bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        PL_ABORT("GaterOperationsLM::applyGeneratorCRZ is not implemented yet");
    }
    static void
    applyGeneratorControlledPhaseShift(CFP_t *arr, size_t num_qubits,
                                       const std::vector<size_t> &wires,
                                       [[maybe_unused]] bool inverse) {
        static_cast<void>(arr);
        static_cast<void>(num_qubits);
        static_cast<void>(wires);
        PL_ABORT("GaterOperationsLM::applyGeneratorControlledPhaseShift is not "
                 "implemented yet");
    }
};

} // namespace Pennylane
